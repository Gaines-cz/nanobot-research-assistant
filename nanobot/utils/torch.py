"""PyTorch utility functions for device detection and optimization."""

import torch
from loguru import logger


def get_optimal_device() -> str:
    """获取最优可用设备：MPS > CUDA > CPU

    Returns:
        设备字符串："mps", "cuda", 或 "cpu"
    """
    if torch.backends.mps.is_available():
        logger.info("MPS device available")
        return "mps"
    elif torch.cuda.is_available():
        logger.info("CUDA device available")
        return "cuda"
    else:
        logger.info("Using CPU")
        return "cpu"
