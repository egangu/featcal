from __future__ import annotations

import argparse
import gc
from pathlib import Path

from transformers import CLIPVisionModel

from .constants import BASE_MODEL_ID, EXPERT_MODEL_IDS, TASKS
from .evaluate import evaluate_clip_vision_model
from .featcal import CLIPFeatCalibrator, FeatCalConfig, load_expert_models
from .merge import save_vision_model, task_arithmetic_merge
from .models import load_processor, load_vision_model
from .utils import resolve_device, save_json, save_report, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the CLIP ViT-B/32 TA8 Task Arithmetic + FeatCal demo."
    )
    parser.add_argument("--output-dir", default="outputs/clip-vit-b32-ta8")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--datasets-cache-dir", default=None)
    parser.add_argument("--base-model-id", default=BASE_MODEL_ID)
    parser.add_argument(
        "--expert-model-format",
        default="tanganke/clip-vit-base-patch32_{task}",
        help="HF id or local path pattern for task experts. Must contain {task}.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--ta-scaling-factor", type=float, default=0.3)
    parser.add_argument("--num-calibration-examples", type=int, default=256)
    parser.add_argument("--calibration-batch-size", type=int, default=16)
    parser.add_argument("--lambda-ratio", type=float, default=0.05)
    parser.add_argument("--anchor-blend-rho", type=float, default=2.2)
    parser.add_argument("--teacher-interp-alpha", type=float, default=0.25)
    parser.add_argument("--covariance-eps", type=float, default=1e-8)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-num-workers", type=int, default=8)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--skip-ta",
        action="store_true",
        help="Load --ta-model-path instead of constructing Task Arithmetic.",
    )
    parser.add_argument("--ta-model-path", default=None)
    parser.add_argument(
        "--skip-eval-ta",
        action="store_true",
        help="Skip evaluating the Task Arithmetic baseline.",
    )
    parser.add_argument(
        "--skip-featcal",
        action="store_true",
        help="Only construct/evaluate the Task Arithmetic baseline.",
    )
    return parser


def _selected_experts(tasks: list[str], expert_model_format: str) -> dict[str, str]:
    unknown = set(tasks) - set(EXPERT_MODEL_IDS)
    if unknown:
        raise KeyError(f"Unknown tasks: {sorted(unknown)}")
    if "{task}" not in expert_model_format:
        raise ValueError("--expert-model-format must contain {task}.")
    return {task: expert_model_format.format(task=task) for task in tasks}


def _save_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    save_json(
        {
            "backbone": "CLIP-ViT-B/32",
            "tasks": args.tasks,
            "merging_baseline": "Task Arithmetic",
            "post_merging_method": "FeatCal",
            "base_model_id": args.base_model_id,
            "expert_model_format": args.expert_model_format,
            "seed": args.seed,
            "task_arithmetic": {"scaling_factor": args.ta_scaling_factor},
            "featcal": {
                "num_calibration_examples": args.num_calibration_examples,
                "lambda_ratio": args.lambda_ratio,
                "anchor_blend_rho": args.anchor_blend_rho,
                "teacher_interp_alpha": args.teacher_interp_alpha,
                "calibrate_bias": True,
                "calibrate_layernorm": True,
            },
        },
        output_dir / "run_config.json",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    ta_dir = output_dir / "ta"
    featcal_dir = output_dir / "featcal-ta"
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_run_config(args, output_dir)

    expert_ids = _selected_experts(args.tasks, args.expert_model_format)

    if args.skip_ta:
        if args.ta_model_path is None:
            raise ValueError("--skip-ta requires --ta-model-path.")
        ta_model = CLIPVisionModel.from_pretrained(args.ta_model_path)
    else:
        ta_model = task_arithmetic_merge(
            base_model_id=args.base_model_id,
            expert_model_ids=expert_ids,
            scaling_factor=args.ta_scaling_factor,
            cache_dir=args.cache_dir,
        )
        save_vision_model(ta_model, ta_dir)

    if not args.skip_eval_ta:
        ta_report = evaluate_clip_vision_model(
            vision_model=ta_model,
            tasks=args.tasks,
            device=device,
            cache_dir=args.cache_dir,
            datasets_cache_dir=args.datasets_cache_dir,
            base_model_id=args.base_model_id,
            batch_size=args.eval_batch_size,
            num_workers=args.eval_num_workers,
            max_samples=args.max_eval_samples,
        )
        save_report(ta_report, ta_dir)

    if args.skip_featcal:
        return 0

    # The reference run launches Task Arithmetic and FeatCal as two separate
    # commands with the same seed. Reset at the start of the FeatCal stage so
    # the following model loads and calibration DataLoader shuffling see the
    # same RNG boundary as the second reference command.
    seed_everything(args.seed)

    ta_model.to(device)
    base_model = load_vision_model(args.base_model_id, cache_dir=args.cache_dir)
    experts = load_expert_models(expert_ids, cache_dir=args.cache_dir)
    processor = load_processor(args.base_model_id, cache_dir=args.cache_dir)
    config = FeatCalConfig(
        num_calibration_examples=args.num_calibration_examples,
        calibration_batch_size=args.calibration_batch_size,
        lambda_ratio=args.lambda_ratio,
        anchor_blend_rho=args.anchor_blend_rho,
        teacher_interp_alpha=args.teacher_interp_alpha,
        covariance_eps=args.covariance_eps,
        calibrate_bias=True,
        calibrate_layernorm=True,
    )
    calibrator = CLIPFeatCalibrator(
        config,
        processor=processor,
        device=device,
        cache_dir=args.cache_dir,
        datasets_cache_dir=args.datasets_cache_dir,
    )
    featcal_model = calibrator.calibrate(
        merged_model=ta_model,
        base_model=base_model,
        expert_models=experts,
        tasks=args.tasks,
    )
    save_vision_model(featcal_model, featcal_dir)

    del base_model, experts
    gc.collect()

    featcal_report = evaluate_clip_vision_model(
        vision_model=featcal_model,
        tasks=args.tasks,
        device=device,
        cache_dir=args.cache_dir,
        datasets_cache_dir=args.datasets_cache_dir,
        base_model_id=args.base_model_id,
        batch_size=args.eval_batch_size,
        num_workers=args.eval_num_workers,
        max_samples=args.max_eval_samples,
    )
    save_report(featcal_report, featcal_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
