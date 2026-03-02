import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss


class Criterion(nn.Module):
    def __init__(
        self,
        w_sal=1,
        w_ss=0,
        w_nss=1,
        w_cc=1,
        w_kld=1,
        w_mse=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sal_loss = Loss(w_nss, w_cc, w_kld, w_mse, **kwargs)

        weight = torch.tensor([1, 10, 10, 10, 10, 2, 10], dtype=torch.float32)
        self.dice_loss = DiceLoss(mode="multiclass")
        self.ce_loss = nn.CrossEntropyLoss(reduction="none", weight=weight)
        self.iou_loss = JaccardLoss(mode="multiclass")

        self.w_sal = w_sal
        self.w_ss = w_ss

    def forward(
        self,
        gts,
        preds,
        fixation: torch.Tensor,
        weights=1,
    ):
        sal_loss = torch.tensor(0.0, device=gts["sal"].device)
        ss_loss = torch.tensor(0.0, device=gts["ss"].device)

        if "sal" in preds:
            sal_loss = self.sal_loss(gts["sal"], preds["sal"], fixation).mean()

        if "ss" in preds:
            ss_loss = (
                self.ce_loss(preds["ss"], gts["ss"]).mean()
                + self.iou_loss(
                    preds["ss"],
                    gts["ss"],
                ).mean()
                + self.dice_loss(preds["ss"], gts["ss"]).mean()
            )

        loss = weights * (self.w_sal * sal_loss + self.w_ss * ss_loss)

        ret = {
            "loss": loss.mean(),
            "sal_loss": sal_loss.mean(),
            "ss_loss": ss_loss.mean(),
        }
        return ret


__all__ = [Criterion]


class Loss(nn.Module):
    def __init__(
        self,
        w_nss=1,
        w_cc=1,
        w_kld=1,
        w_mse=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nss_loss = NSSLoss()
        self.kld_loss = KLDLoss()
        self.cc_loss = CCLoss()
        self.mse_loss = MSELoss()

        self.w_nss = w_nss
        self.w_kld = w_kld
        self.w_cc = w_cc
        self.w_mse = w_mse

    def forward(
        self,
        gt_saliency,
        pred_saliency,
        fixation,
    ):
        loss_nss = self.nss_loss(pred_saliency, fixation)
        loss_cc = self.cc_loss(pred_saliency, gt_saliency)
        loss_kld = self.kld_loss(pred_saliency, gt_saliency)
        loss_mse = self.mse_loss(pred_saliency, gt_saliency)

        loss_per_sample = (
            self.w_nss * loss_nss
            + self.w_cc * loss_cc
            + self.w_kld * loss_kld
            + self.w_mse * loss_mse
        )

        return loss_per_sample


class MSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(MSELoss, self).__init__()
        self.eps = eps

    def forward(self, pred_saliency, gt_saliency):
        B, _, H, W = pred_saliency.shape
        pred = pred_saliency.view(B, -1)

        gt = gt_saliency.view(B, -1)

        mse = F.mse_loss(pred, gt, reduction="none")  # (B, H*W)
        mse = mse.mean(dim=1)  # Mean over spatial dims per sample
        return mse


class SMSELoss(nn.Module):
    """MSE loss to focus only on the fixation points"""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.mseloss = nn.MSELoss(reduction="none")

    def forward(self, pred_saliency, gt_saliency, fixation):
        B, _, H, W = pred_saliency.shape

        pred = pred_saliency.view(B, -1)
        gt = gt_saliency.view(B, -1)
        fix = fixation.view(B, -1)

        loss = self.mseloss(
            pred * fix, gt * fix
        )  #  / fix.sum(dim=1).clamp_min(self.eps)
        return loss.mean(dim=1)


class NSSLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(NSSLoss, self).__init__()
        self.eps = eps

    def forward(self, pred_saliency, fixation_map):
        """
        pred_saliency: predicted saliency map (B, 1, H, W), float32
        fixation_map: binary fixation map (B, 1, H, W), values in {0, 1}
        """
        B, _, H, W = pred_saliency.shape
        pred = pred_saliency.view(B, -1)
        fix = fixation_map.view(B, -1)

        mean = pred.mean(dim=1, keepdim=True)
        std = pred.std(dim=1, keepdim=True) + self.eps
        pred_norm = (pred - mean) / std

        nss = torch.sum(pred_norm * fix, dim=1) / (fix.sum(dim=1) + self.eps)

        return -nss  # negative because we want to maximize NSS


class KLDLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(KLDLoss, self).__init__()
        self.eps = eps

    def forward(self, pred_saliency, gt_saliency):
        """
        pred_saliency: predicted saliency map (B, 1, H, W), float32
        gt_saliency: ground truth saliency (B, 1, H, W), float32
        """

        B, _, H, W = pred_saliency.shape

        pred = pred_saliency.view(B, -1)
        gt = gt_saliency.view(B, -1)

        pred_sum = pred.sum(dim=1, keepdim=True)
        pred_norm = pred / (pred_sum + self.eps)
        pred_norm = torch.clamp(pred_norm, min=self.eps)
        log_pred_norm = torch.log(pred_norm)

        gt_sum = gt.sum(dim=1, keepdim=True)
        gt_norm = gt / (gt_sum + self.eps)
        gt_norm = torch.clamp(gt_norm, min=self.eps)
        log_gt_norm = torch.log(gt_norm)

        kld = torch.sum(gt_norm * (log_gt_norm - log_pred_norm), 1)

        return kld


class CCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CCLoss, self).__init__()
        self.eps = eps

    def forward(self, pred_logits, gt_saliency):
        """
        pred_logits: predicted logits (B, 1, H, W) — raw model output
        gt_saliency: ground truth saliency (B, 1, H, W) — unnormalized map
        """
        B, _, H, W = pred_logits.shape

        # Flatten spatial dims
        pred = pred_logits.view(B, -1)
        gt = gt_saliency.view(B, -1)

        # Normalize pred using softmax (probability distribution)
        pred_sum = pred.sum(dim=1, keepdim=True)
        pred_prob = pred / (pred_sum + self.eps)

        # Normalize gt to sum to 1
        gt_sum = gt.sum(dim=1, keepdim=True)
        gt_prob = gt / (gt_sum + self.eps)

        # Standardize both
        pred_mean = pred_prob.mean(dim=1, keepdim=True)
        gt_mean = gt_prob.mean(dim=1, keepdim=True)

        pred_std = pred_prob.std(dim=1, keepdim=True) + self.eps
        gt_std = gt_prob.std(dim=1, keepdim=True) + self.eps

        pred_norm = (pred_prob - pred_mean) / pred_std
        gt_norm = (gt_prob - gt_mean) / gt_std

        # Compute correlation coefficient
        ab = torch.sum(pred_norm * gt_norm, dim=1)
        bb = torch.sum(pred_norm * pred_norm, dim=1)
        aa = torch.sum(gt_norm * gt_norm, dim=1)

        cc = ab / torch.sqrt(aa * bb + self.eps)

        return 1 - cc


if __name__ == "__main__":
    nss = NSSLoss()
    pred = torch.zeros((4, 1, 3200, 360))
    target = torch.zeros((4, 1, 3200, 360))
    print(target.sum(dim=1).shape)
    loss = nss(pred, target)
    print(loss.shape)
    print(loss)
