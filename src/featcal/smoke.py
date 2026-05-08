from __future__ import annotations

import argparse
from pathlib import Path

from .run_ta8 import main as run_ta8_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a cheap two-task FeatCal smoke check."
    )
    parser.add_argument("--output-dir", default="outputs/smoke")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--datasets-cache-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    return run_ta8_main(
        [
            "--output-dir",
            str(output_dir),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--tasks",
            "mnist",
            "svhn",
            "--num-calibration-examples",
            "8",
            "--calibration-batch-size",
            "4",
            "--eval-batch-size",
            "32",
            "--eval-num-workers",
            "0",
            "--max-eval-samples",
            "64",
        ]
        + (["--cache-dir", args.cache_dir] if args.cache_dir else [])
        + (
            ["--datasets-cache-dir", args.datasets_cache_dir]
            if args.datasets_cache_dir
            else []
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

