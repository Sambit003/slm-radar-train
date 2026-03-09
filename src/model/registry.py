from typing import Dict, Optional, Tuple, Type

from transformers import AutoConfig

from src.model.base import BaseMultiHeadClassifier

MODEL_REGISTRY: Dict[str, Type[BaseMultiHeadClassifier]] = {}


def register_model(*aliases: str):
    """Register a classifier class for one or more architecture aliases."""

    def decorator(cls: Type[BaseMultiHeadClassifier]):
        for alias in aliases:
            MODEL_REGISTRY[alias.lower()] = cls
        return cls

    return decorator


def get_model_class(
    model_name_or_path: str,
    *,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = True,
) -> Type[BaseMultiHeadClassifier]:
    """Resolve model class from HF config.model_type, then fallback by name."""
    model_type = ""
    try:
        cfg = AutoConfig.from_pretrained(
            model_name_or_path,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )
        model_type = str(getattr(cfg, "model_type", "")).lower()
    except Exception:
        model_type = ""

    candidates = _candidate_keys(model_name_or_path, model_type)
    for key in candidates:
        if key in MODEL_REGISTRY:
            return MODEL_REGISTRY[key]

    supported = ", ".join(sorted(MODEL_REGISTRY.keys()))
    raise ValueError(
        f"Unsupported model architecture for '{model_name_or_path}'. "
        f"Detected model_type='{model_type or 'unknown'}'. "
        f"Supported keys: {supported}"
    )


def _candidate_keys(
    model_name_or_path: str, model_type: str
) -> Tuple[str, ...]:
    lowered_name = (model_name_or_path or "").lower()
    keys = []
    if model_type:
        keys.append(model_type)
        if "-" in model_type:
            keys.extend(part for part in model_type.split("-") if part)
    if lowered_name:
        keys.append(lowered_name)
        if "/" in lowered_name:
            keys.extend(part for part in lowered_name.split("/") if part)
        if "-" in lowered_name:
            keys.extend(part for part in lowered_name.split("-") if part)

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return tuple(unique)
