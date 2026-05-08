from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from .classification import get_classnames_and_templates
from .constants import BASE_MODEL_ID
from .data import load_task_dataset, make_clip_loader
from .models import load_clip_model, load_processor
from .utils import count_parameters


@torch.no_grad()
def build_zero_shot_weights(
    *,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    task: str,
    device: torch.device,
    text_batch_size: int = 256,
) -> Tensor:
    classnames, templates = get_classnames_and_templates(task)
    texts: list[str] = []
    class_slices: list[slice] = []
    for classname in classnames:
        start = len(texts)
        texts.extend(template(classname) for template in templates)
        class_slices.append(slice(start, len(texts)))

    text_model = clip_model.text_model.to(device)
    text_projection = clip_model.text_projection.to(device)
    prompt_embeddings = []
    for start in range(0, len(texts), text_batch_size):
        batch_texts = texts[start : start + text_batch_size]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        embeddings = text_model(**inputs)[1]
        embeddings = text_projection(embeddings)
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        prompt_embeddings.append(embeddings)

    prompt_embeddings = torch.cat(prompt_embeddings, dim=0)
    zeroshot_weights = []
    for class_slice in class_slices:
        class_embedding = prompt_embeddings[class_slice].mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm(
            p=2,
            dim=-1,
            keepdim=True,
        )
        zeroshot_weights.append(class_embedding)
    return torch.stack(zeroshot_weights, dim=0)


@torch.no_grad()
def evaluate_clip_vision_model(
    *,
    vision_model: CLIPVisionModel,
    tasks: Sequence[str],
    device: torch.device,
    cache_dir: str | Path | None = None,
    datasets_cache_dir: str | Path | None = None,
    base_model_id: str = BASE_MODEL_ID,
    batch_size: int = 256,
    num_workers: int = 8,
    max_samples: int | None = None,
) -> dict:
    processor = load_processor(base_model_id, cache_dir=cache_dir)
    clip_model = load_clip_model(base_model_id, cache_dir=cache_dir)
    clip_model.vision_model = vision_model
    clip_model.to(device)
    clip_model.eval()
    vision_model.to(device)
    vision_model.eval()

    report: dict = {"model_info": count_parameters(vision_model)}
    zeroshot_cache: dict[str, Tensor] = {}
    logit_scale = clip_model.logit_scale.exp()

    for task in tqdm(tasks, desc="Evaluating tasks"):
        dataset = load_task_dataset(task, "test", cache_dir=datasets_cache_dir)
        loader = make_clip_loader(
            dataset,
            processor,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            max_samples=max_samples,
            pin_memory=(device.type == "cuda"),
        )
        text_embeds = zeroshot_cache.get(task)
        if text_embeds is None:
            text_embeds = build_zero_shot_weights(
                clip_model=clip_model,
                processor=processor,
                task=task,
                device=device,
            )
            zeroshot_cache[task] = text_embeds

        total_correct = 0
        total_examples = 0
        loss_sum = 0.0
        loss_batches = 0
        for images, targets in tqdm(loader, desc=f"Evaluating {task}", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()
            outputs = vision_model(images)
            image_embeds = clip_model.visual_projection(outputs[1])
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            logits = image_embeds @ text_embeds.t() * logit_scale
            loss = F.cross_entropy(logits, targets)
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == targets).sum().item())
            total_examples += int(targets.numel())
            loss_sum += float(loss.detach().cpu().item())
            loss_batches += 1

        report[task] = {
            "accuracy": total_correct / total_examples if total_examples else 0.0,
            "loss": loss_sum / loss_batches if loss_batches else 0.0,
        }

    accuracies = [report[task]["accuracy"] for task in tasks]
    losses = [report[task]["loss"] for task in tasks]
    report["average"] = {
        "accuracy": sum(accuracies) / len(accuracies),
        "loss": sum(losses) / len(losses),
    }
    return report

