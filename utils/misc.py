import torch
import torch.distributed as dist


def clip_percentiles(klds, lower_pct=20, upper_pct=80):
    # Compute percentiles
    lower = torch.quantile(klds, lower_pct / 100.0)
    upper = torch.quantile(klds, upper_pct / 100.0)

    # Clip to percentile bounds
    clipped = torch.clamp(klds, min=lower.item(), max=upper.item())

    return clipped, lower, upper
