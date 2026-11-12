"""Freeze the effective backend configuration; no provider SDK or model calls here."""
import hashlib
import inspect
import json
import math
from pathlib import Path


def describe_backend(generate, *, strict=False):
    target = generate if inspect.isfunction(generate) else type(generate)
    path = inspect.getsourcefile(target)
    descriptor = dict(adapter=f"{target.__module__}.{target.__qualname__}",
                      adapter_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
                      if path and Path(path).is_file() else None)
    describe = getattr(generate, "describe", None)
    config = describe() if callable(describe) else None
    if strict:
        required = {"model_id", "model_revision", "tokenizer_revision", "precision",
                    "context_tokens", "max_new_tokens", "temperature", "top_p"}
        if not isinstance(config, dict) or not required <= set(config):
            raise ValueError("model backend describe() must declare revisions and sampling limits")
        if descriptor["adapter_sha256"] is None:
            raise ValueError("model backend adapter source must be available for freezing")
        for key in ("model_id", "model_revision", "tokenizer_revision", "precision"):
            if not isinstance(config[key], str) or not config[key].strip():
                raise ValueError(f"backend {key} must be explicit")
        for key in ("context_tokens", "max_new_tokens"):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f"backend {key} must be a positive integer")
        for key in ("temperature", "top_p"):
            if (type(config[key]) not in (int, float) or not math.isfinite(config[key])
                    or config[key] < 0 or (key == "top_p" and not 0 < config[key] <= 1)):
                raise ValueError(f"backend {key} is invalid")
    descriptor["configuration"] = config
    return json.loads(json.dumps(descriptor, sort_keys=True, allow_nan=False))
