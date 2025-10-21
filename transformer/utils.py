import torch


def is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda


def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1
