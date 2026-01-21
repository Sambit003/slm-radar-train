import logging

import torch


def get_device() -> torch.device:
    """
    Determines the best available device for training (TPU > GPU > CPU).

    Returns:
        torch.device: The selected device.
    """
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        logging.info(f"Using TPU: {device}")
        return device
    except ImportError:
        pass

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS (Apple Silicon)")
        return device

    logging.info("Using CPU")
    return torch.device("cpu")
