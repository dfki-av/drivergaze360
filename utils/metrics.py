import torch
import torch.nn as nn
from torcheval.metrics import Mean
from torcheval.metrics.toolkit import sync_and_compute


class MetricAggregator:
    def __init__(self, device):
        self.metrics = Metrics()
        self.device = device
        self.sim = Mean(device=self.device)
        self.kld = Mean(device=self.device)
        self.cc = Mean(device=self.device)
        self.nss = Mean(device=self.device)
        self.iou = Mean(device=self.device)
        self.dice = Mean(device=self.device)

    def update(self, batch_metrics, valid_count):
        self.sim.update(batch_metrics["sim"], weight=valid_count)
        self.kld.update(batch_metrics["kld"], weight=valid_count)
        self.cc.update(batch_metrics["cc"], weight=valid_count)
        self.nss.update(batch_metrics["nss"], weight=valid_count)

        if "iou" in batch_metrics:
            self.iou.update(batch_metrics["iou"], weight=valid_count)
            self.dice.update(batch_metrics["dice"], weight=valid_count)

    def compute(self):
        kld = sync_and_compute(self.kld)
        cc = sync_and_compute(self.cc)
        nss = sync_and_compute(self.nss)
        sim = sync_and_compute(self.sim)
        iou = sync_and_compute(self.iou)
        dice = sync_and_compute(self.dice)
        return {"kld": kld, "cc": cc, "nss": nss, "sim": sim, "iou": iou, "dice": dice}

    def reset(self):
        self.sim.reset()
        self.cc.reset()
        self.kld.reset()
        self.nss.reset()
        self.iou.reset()
        self.dice.reset()


class Metrics(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    @torch.no_grad()
    def forward(self, gts, preds, fixation):
        pred_saliency = preds["sal"].detach().clone()
        gt_saliency = gts["sal"].detach().clone()
        fixation = fixation.detach().clone()
        sim = self.sim(pred_saliency, gt_saliency)
        cc = self.cc(pred_saliency, gt_saliency)
        kld = self.kld(pred_saliency, gt_saliency)
        nss = self.nss(pred_saliency, fixation)

        ret = {
            "sim": sim.mean(),
            "cc": cc.mean(),
            "kld": kld.mean(),
            "nss": nss.mean(),
        }

        if "ss" in preds:
            iou, dice = self.compute_seg_metrics(preds["ss"], gts["ss"])
            ret["iou"] = iou
            ret["dice"] = dice

        return ret

    def sim(self, pred_saliency, gt_saliency):
        B, _, H, W = pred_saliency.shape

        pred = pred_saliency.view(B, -1)
        gt = gt_saliency.view(B, -1)

        pred -= pred.min(dim=1, keepdim=True).values
        gt -= gt.min(dim=1, keepdim=True).values

        pred /= pred.sum(dim=1, keepdim=True) + self.eps
        gt /= gt.sum(dim=1, keepdim=True) + self.eps

        return torch.minimum(pred, gt).sum(dim=1)

    def cc(self, pred_logits, gt_saliency):
        """
        pred_logits: predicted logits (B, 1, H, W) — raw model output
        gt_saliency: ground truth saliency (B, 1, H, W) — unnormalized map
        """
        B, _, H, W = pred_logits.shape

        # Flatten spatial dims
        pred = pred_logits.view(B, -1)
        gt = gt_saliency.view(B, -1)

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

        return cc

    def kld(self, pred_saliency, gt_saliency):
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

    def nss(self, pred_saliency, fixation_map):
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
        return nss

    def compute_seg_metrics(self, pred, target):
        iou = self.iou_score(pred, target)
        dice = self.dice_score(pred, target)
        return iou, dice

    def iou_score(self, pred, target):
        # pred: (B,C,H,W) logits or probs
        pred_lbl = pred.argmax(dim=1)  # (B,H,W)
        C = pred.shape[1]

        inter = []
        union = []
        for c in range(C):
            p = pred_lbl == c  # (B,H,W)
            t = target == c  # (B,H,W)
            i = (p & t).sum(dim=(1, 2))  # (B,)
            u = (p | t).sum(dim=(1, 2))  # (B,)
            inter.append(i)
            union.append(u)

        inter = torch.stack(inter, dim=1)  # (B,C)
        union = torch.stack(union, dim=1)  # (B,C)
        iou = (inter + self.eps) / (union + self.eps)
        return iou.mean(dim=1)  # (B,)

    def dice_score(self, pred, target):
        # pred: (B,C,H,W) logits or probs
        pred_lbl = pred.argmax(dim=1)  # (B,H,W)
        C = pred.shape[1]

        inter = []
        card = []
        for c in range(C):
            p = pred_lbl == c  # (B,H,W)
            t = target == c  # (B,H,W)
            i = (p & t).sum(dim=(1, 2))  # (B,)
            c_ = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))  # (B,)
            inter.append(i)
            card.append(c_)

        inter = torch.stack(inter, dim=1)  # (B,C)
        card = torch.stack(card, dim=1)  # (B,C)
        dice = (2 * inter + self.eps) / (card + self.eps)
        return dice.mean(dim=1)  # (B,)
