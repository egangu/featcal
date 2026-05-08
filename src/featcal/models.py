from __future__ import annotations

from pathlib import Path

from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel


def _cache_kwargs(cache_dir: str | Path | None) -> dict:
    return {"cache_dir": str(cache_dir)} if cache_dir is not None else {}


def load_processor(model_id: str, cache_dir: str | Path | None = None) -> CLIPProcessor:
    return CLIPProcessor.from_pretrained(model_id, **_cache_kwargs(cache_dir))


def load_clip_model(model_id: str, cache_dir: str | Path | None = None) -> CLIPModel:
    model = CLIPModel.from_pretrained(model_id, **_cache_kwargs(cache_dir))
    model.eval()
    return model


def load_vision_model(
    model_id_or_path: str | Path,
    cache_dir: str | Path | None = None,
) -> CLIPVisionModel:
    model = CLIPVisionModel.from_pretrained(
        str(model_id_or_path),
        **_cache_kwargs(cache_dir),
    )
    model.eval()
    return model

