import os

import torch.nn as nn

from models import (
    DriverGaze360,
)


def build_model(args):
    """
    Model factory that returns the correct model based on config.
    """
    heads = ["sal"]
    if args.w_ss > 0:
        heads.append("ss")

    MODEL_REGISTRY = {
        "DriverGaze360": lambda: DriverGaze360(
            heads=heads,
        ),
    }
    model = args.model
    # Assert model is valid
    assert model in MODEL_REGISTRY, (
        f"Invalid model '{model}'. Choose from: {list(MODEL_REGISTRY.keys())}"
    )

    # Build and return the model
    _model = MODEL_REGISTRY[model]()
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        print(f"Using -> {model}")
        print_model_info(_model)

    return _model


def print_model_info(model: nn.Module):
    # Count total parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Count trainable parameters
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate model size in MB
    # Each parameter is typically a float32 (4 bytes)
    param_size_bytes = num_params * 4
    model_size_mb = param_size_bytes / (1024**2)

    print("Total Parameters:", num_params)
    print("Trainable Parameters:", num_trainable)
    print("Model Size (MB): ", model_size_mb)
