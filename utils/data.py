import torch


def get_bin_masks(is_img: torch.Tensor) -> torch.Tensor:
    """Generate a binary mask highlighting
    specific object instances from an input image tensor.
    Roads        =    1u,
    Sidewalks    =    2u,
    Buildings    =    3u,
    Walls        =    4u,
    Fences       =    5u,
    Poles        =    6u,
    TrafficLight =    7u,
    TrafficSigns =    8u,
    Vegetation   =    9u,
    Terrain      =   10u,
    Sky          =   11u,
    Pedestrians  =   12u,
    Rider        =   13u,
    Car          =   14u,
    Truck        =   15u,
    Bus          =   16u,
    Train        =   17u,
    Motorcycle   =   18u,
    Bicycle      =   19u,
    // custom
    Static       =   20u,
    Dynamic      =   21u,
    Other        =   22u,
    Water        =   23u,
    RoadLines    =   24u,
    Ground       =   25u,
    Bridge       =   26u,
    RailTrack    =   27u,
    GuardRail    =   28u,
    """

    red_channel = is_img[0, :, :]

    target_values = torch.tensor(
        [7, 8, 12, 13, 14, 15, 16, 17, 18, 19], device=is_img.device
    )
    mask = torch.isin(red_channel, target_values).float()
    return mask.unsqueeze(0)  # [C, H, W]


def get_seen_objects(
    is_img: torch.Tensor, is_mask: torch.Tensor, saliency: torch.Tensor
) -> torch.Tensor:
    """
    Returns the subset of `is_img` pixels that are both salient and present in the mask.

    Args:
        is_img (torch.Tensor): Input image, shape (C, H, W), channel-first RGB.
        is_mask (torch.Tensor): Binary mask, shape (H, W), indicates valid object regions.
        saliency (torch.Tensor): Saliency map, shape (1, H, W) or (H, W).

    Returns:
        torch.Tensor: Masked image of same shape as `is_img` (C, H, W),
                      where only pixels corresponding to seen objects are kept, others set to 0.
    """

    # Apply saliency to mask
    sal_mask = saliency.squeeze() * is_mask
    sal_mask[sal_mask >= 0.225] = 1
    sal_mask[sal_mask < 0.225] = 0

    # Expand to 3 channels
    sal_mask = sal_mask.repeat(3, 1, 1).to(torch.uint8)

    # Masked instances
    instances = sal_mask * is_img
    objects = instances.permute(1, 2, 0).reshape(-1, 3)
    objects = objects[objects[:, 0] != 0]  # only keep pixels where R != 0

    # Pack RGB triplets into integers
    is_packed = (
        (is_img[0, ...].to(torch.int64) << 16)
        | (is_img[1, ...].to(torch.int64) << 8)
        | (is_img[2, ...].to(torch.int64))
    )

    objects_packed = (
        (objects[:, 0].to(torch.int64) << 16)
        | (objects[:, 1].to(torch.int64) << 8)
        | (objects[:, 2].to(torch.int64))
    )

    # Check membership
    mask = torch.isin(is_packed, objects_packed)

    # Keep only matched pixels
    sal_is = torch.where(
        mask, is_img, torch.tensor(0, dtype=is_img.dtype, device=is_img.device)
    )

    return sal_is


def get_instance_masks(is_img: torch.Tensor) -> torch.Tensor:
    """
    Generate a multi-channel mask for semantic segmentation,
    with specific classes grouped into one 'vehicle' class.

    Output channels:
        0: Background
        1: Traffic lights
        2: Traffic signs
        3: Pedestrians
        4: Rider
        5: Vehicles (car + truck + bus + train + motorcycle)
        6: Bicycle
    """

    red_channel = is_img[0, :, :]  # [H, W]

    # Define mapping of semantic groups
    class_groups = {
        1: [7],  # Traffic lights
        2: [8],  # Traffic signs
        3: [12],  # Pedestrians
        4: [13],  # Rider
        5: [14, 15, 16, 17, 18],  # Vehicles
        6: [19],  # Bicycle
    }

    H, W = red_channel.shape
    mask = torch.zeros((H, W), device=is_img.device, dtype=torch.long)

    for class_id, values in class_groups.items():
        for v in values:
            mask[red_channel == v] = class_id

    return mask  # [H, W], integer class labels
