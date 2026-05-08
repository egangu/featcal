from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BaseImageProcessor, ProcessorMixin

from .constants import DATASET_SPECS


def load_task_dataset(
    task: str,
    split: str,
    *,
    cache_dir: str | Path | None = None,
):
    if task not in DATASET_SPECS:
        raise KeyError(f"Unknown task {task!r}.")
    spec = DATASET_SPECS[task]
    split_name = spec.train_split if split == "train" else spec.test_split
    kwargs = {"split": split_name, "trust_remote_code": True}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if spec.name is None:
        return load_dataset(spec.path, **kwargs)
    return load_dataset(spec.path, spec.name, **kwargs)


def _extract_image_and_label(item):
    if isinstance(item, dict):
        image = item.get("image", item.get("img"))
        label = item.get("label", item.get("fine_label"))
        if image is None or label is None:
            raise KeyError(f"Dataset item lacks image/label keys: {item.keys()}")
        return image, label
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return item[0], item[1]
    raise TypeError(f"Unsupported dataset item type: {type(item)!r}")


class CLIPImageDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        processor: ProcessorMixin | BaseImageProcessor,
    ) -> None:
        self.dataset = dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = _extract_image_and_label(self.dataset[idx])
        image = image.convert("RGB")
        pixel_values = self.processor(images=[image], return_tensors="pt")[
            "pixel_values"
        ][0]
        if isinstance(label, bool):
            label = int(label)
        return pixel_values, int(label)


def maybe_subset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    return Subset(dataset, range(min(max_samples, len(dataset))))


def make_clip_loader(
    dataset,
    processor,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    max_samples: int | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    wrapped = CLIPImageDataset(maybe_subset(dataset, max_samples), processor)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": False,
    }
    if pin_memory is not None:
        kwargs["pin_memory"] = pin_memory
    if num_workers > 0:
        kwargs["prefetch_factor"] = 1
    return DataLoader(wrapped, **kwargs)


def iter_limited_image_batches(
    loader: Iterable,
    max_examples: int,
) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    seen = 0
    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        remaining = max_examples - seen
        if remaining <= 0:
            break
        if images.shape[0] > remaining:
            images = images[:remaining]
        batches.append(images.detach().cpu())
        seen += int(images.shape[0])
    if not batches:
        raise ValueError("No calibration images were collected.")
    return batches

