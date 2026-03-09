from types import SimpleNamespace

import torch

from src.utils.device import get_device, get_device_type


def test_get_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda index: "Fake GPU",
    )
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )

    assert get_device().type == "cuda"
    assert get_device_type() == "cuda"


def test_get_device_uses_mps_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )

    assert get_device().type == "mps"
    assert get_device_type() == "mps"


def test_get_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )

    assert get_device().type == "cpu"
    assert get_device_type() == "cpu"
