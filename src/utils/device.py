import logging

import torch


def get_device() -> torch.device:
    """
    Determines the best available device for training (GPU > CPU).

    Returns:
        torch.device: The selected device.
    """
    # Try CUDA GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device

    # Try Apple Silicon MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS (Apple Silicon)")
        return device

    logging.info("Using CPU")
    return torch.device("cpu")


def get_device_type() -> str:
    """Returns device type as string: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
