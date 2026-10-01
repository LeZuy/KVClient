import json
import torch
import hashlib
import transformers

from typing import Any
from __future__ import annotations
from transformers import PreTrainedModel
from dataclasses import dataclass, asdict
from importlib.metadata import PackageNotFoundError, version

PE_FIELDS = ("rope_theta", "rope_scaling", "rope_parameters",
            "partial_rotary_factor", "rotary_pct", "max_position_embeddings",
            "original_max_position_embeddings", "sliding_window", "use_sliding_window")

@dataclass(frozen=True)
class QuantizationProfile:
    method: str
    config: dict[str, Any]

@dataclass(frozen=True)
class AttentionProfile:
    implementation: str
    backend: str | None
    version: str | None

@dataclass(frozen=True)
class PositionProfile:
    type: str
    config: dict[str, Any]

@dataclass
class Profile:
    dtype: str
    model_id: str
    torch_version: str
    schema_version: int
    transformers_version: str
    weight_checkpoint_hash: str
    attention: AttentionProfile
    positional_encoding: PositionProfile
    quantization: QuantizationProfile | None

    def canonical_bytes(self) -> bytes:
        data = asdict(self)
        canonical = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return canonical.encode("utf-8")

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

def _normalize(value: Any) -> Any:
    """Convert config values to JSON objects."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")

    if isinstance(value, dict):
        return {
            str(k): _normalize(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]

    if hasattr(value, "to_dict"):
        return _normalize(value.to_dict())

    return str(value)

def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None

def _get_text_config(model: PreTrainedModel):
    config = model.config
    if hasattr(config, "get_text_config"):
        try:
            return config.get_text_config()
        except Exception:
            pass
    return config

def _extract_model_id(model: PreTrainedModel) -> str:
    config = model.config
    model_id = getattr(config, "_name_or_path", None)
    if not model_id:
        model_id = getattr(model, "name_or_path", None)
    return str(model_id) if model_id else "unknown"

def _extract_dtype(model: PreTrainedModel) -> str:
    dtype = getattr(model, "dtype", None)
    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            return "unknown"

    return str(dtype).removeprefix("torch.")

def _extract_checkpoint_hash(model: PreTrainedModel) -> str:
    config = model.config
    commit_hash = getattr(config, "_commit_hash", None)
    if commit_hash:
        return str(commit_hash)
    return "unknown"

def _extract_positional_encoding(model: PreTrainedModel) -> dict[str, Any]:
    config = _get_text_config(model)
    result: dict[str, Any] = {}

    for field in PE_FIELDS:
        if not hasattr(config, field):
            continue

        value = getattr(config, field)
        
        if value is not None:
            result[field] = _normalize(value)

    if ("rope_theta" in result or "rope_scaling" in result or "rope_parameters" in result):
        result["type"] = "rope"
    else:
        result["type"] = "unknown"

    return result

def _extract_attention(model: PreTrainedModel) -> dict[str, Any]:
    config = _get_text_config(model)
    impl = getattr(config, "_attn_implementation", None)

    if impl is None:
        impl = getattr(config, "attn_implementation", None)

    if impl is None:
        impl = "unknown"

    impl = str(impl)

    attention = {"implementation": impl,
                "backend": None,
                "version": None}

    if impl in {"flash_attention_2", "flash_attention_3"}:
        attention["backend"] = "flash_attn"
        attention["version"] = _package_version("flash-attn")

    elif impl == "sdpa":
        # PyTorch SDPA dispatch to different kernels at runtime.
        # Record API/backend family, not exact CUDA kernel.
        attention["backend"] = "pytorch_sdpa"
        attention["version"] = torch.__version__

    elif impl == "eager":
        attention["backend"] = "transformers_eager"
        attention["version"] = transformers.__version__

    elif impl == "flex_attention":
        attention["backend"] = "pytorch_flex_attention"
        attention["version"] = torch.__version__

    return attention

def _extract_quantization(model: PreTrainedModel) -> dict[str, Any] | None:
    config = model.config

    quant_config = getattr(config, "quantization_config", None)

    if quant_config is None:
        quant_config = getattr(model, "quantization_config", None)

    if quant_config is None:
        return None

    if hasattr(quant_config, "to_dict"):
        quant_config = quant_config.to_dict()

    if isinstance(quant_config, dict):
        result = _normalize(quant_config)

        method = (result.get("quant_method") or result.get("quantization_method"))

        if method is not None:
            result["method"] = str(method)

        return result

    return {
        "method": quant_config.__class__.__name__,
        "config": _normalize(quant_config)
        }

def extract_profile(model: PreTrainedModel, 
                    *,
                    checkpoint_hash: str | None = None) -> Profile:

    detected_hash = _extract_checkpoint_hash(model)

    if checkpoint_hash is not None:
        detected_hash = checkpoint_hash

    return Profile(
            model_id=_extract_model_id(model),
            checkpoint_hash=detected_hash,
            dtype=_extract_dtype(model),
            quantization=_extract_quantization(model),
            attention=_extract_attention(model),
            positional_encoding=_extract_positional_encoding(model),
            transformers_version=transformers.__version__,
            torch_version=torch.__version__
        )