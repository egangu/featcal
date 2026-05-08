# FeatCal: Feature Calibration for Post-Model Merging

This repository contains the minimal official demo for reproducing the CLIP
ViT-B/32 8-task Task Arithmetic + FeatCal result.

The demo is self-contained and includes the code needed to:

1. load CLIP models,
2. load the 8 task datasets,
3. construct Task Arithmetic merged weights,
4. run FeatCal calibration,
5. evaluate zero-shot classification accuracy.

## Demo Setting

- Backbone: `openai/clip-vit-base-patch32`
- Tasks: SUN397, Stanford Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD
- Merging baseline: Task Arithmetic, scaling factor `0.3`
- Post-merging method: FeatCal
- Calibration samples: `256` per task
- Reproduction hyperparameters: `lambda_ratio=0.05`,
  `anchor_blend_rho=2.2`, `teacher_interp_alpha=0.25`

The FeatCal script parameters above match the reference run used for this demo.
The paper's main-text default is `rho=2.0, alpha=0.3`; both are conservative
settings, but the numbers below use the script-aligned values.

## Expected Results

With full test-set evaluation, the expected accuracy is:

| Method | Average accuracy |
|---|---:|
| Task Arithmetic | 67.55 |
| Task Arithmetic + FeatCal | 85.47 |

Per-task reference accuracies:

| Task | TA | TA + FeatCal |
|---|---:|---:|
| SUN397 | 57.01 | 70.07 |
| Stanford Cars | 55.70 | 72.52 |
| RESISC45 | 64.75 | 87.94 |
| EuroSAT | 73.30 | 96.22 |
| SVHN | 77.93 | 95.06 |
| GTSRB | 68.50 | 93.21 |
| MNIST | 96.07 | 98.81 |
| DTD | 47.13 | 69.95 |

Small numeric differences can occur from dependency versions and dataloader
ordering, but the average should match closely when using the same seed and full
datasets.

## Clone

```bash
git clone https://github.com/egangu/featcal.git
cd featcal
```

## Install

Create a lightweight Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For CUDA machines, install the PyTorch build matching your CUDA driver first,
then run `pip install -e .`. Example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

The first run downloads CLIP weights and datasets from Hugging Face. You can set
standard cache variables before running:

```bash
export HF_HOME=$PWD/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
```

If your network requires a Hugging Face mirror or HTTP(S) proxy, set those
environment variables in your shell before launching the demo. This repository
does not hard-code any mirror or proxy address.

## Run

Full reproduction:

```bash
bash scripts/run_ta8.sh
```

Equivalent Python command:

```bash
python -m featcal.run_ta8 \
  --output-dir outputs/clip-vit-b32-ta8 \
  --device auto
```

Outputs are written to:

```text
outputs/clip-vit-b32-ta8/
  run_config.json
  ta/
    config.json
    model.safetensors
    metrics.json
    run.log
  featcal-ta/
    config.json
    model.safetensors
    metrics.json
    run.log
```

`run.log` is a single-line JSON report for compatibility with the reference
logs. `metrics.json` is the same report formatted for reading.

## Smoke Check

For a cheap two-task run over tiny calibration/evaluation subsets:

```bash
bash scripts/run_smoke.sh
```

This is only a software check. It is not meant to reproduce the paper table.

## Useful Options

Evaluate on a small subset while debugging:

```bash
python -m featcal.run_ta8 --max-eval-samples 512
```

Use a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m featcal.run_ta8 --device cuda
```

Reuse an existing Task Arithmetic checkpoint:

```bash
python -m featcal.run_ta8 \
  --skip-ta \
  --ta-model-path outputs/clip-vit-b32-ta8/ta \
  --output-dir outputs/clip-vit-b32-ta8
```

Run only the Task Arithmetic baseline:

```bash
python -m featcal.run_ta8 --skip-featcal
```

## Implementation Notes

Task Arithmetic is implemented as:

```text
theta_TA = theta_base + 0.3 * sum_i(theta_expert_i - theta_base)
```

FeatCal calibrates CLIP vision encoder layers in forward order. For each layer,
it collects current merged-model features and task-expert features on the
calibration split, then solves closed-form updates for Linear weights, Linear
biases, and LayerNorm affine parameters. Test data is used only for final
evaluation.
