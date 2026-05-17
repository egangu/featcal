<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/featcal_title_dark.svg">
    <img alt="FeatCal: Feature Calibration for Post-Merging Models" src="assets/featcal_title.svg" width="960">
  </picture>
</h1>

语言： [English](README.md) | **简体中文**

[![arXiv](https://img.shields.io/badge/arXiv-2605.13030-b31b1b.svg)](https://arxiv.org/abs/2605.13030)
[![Hugging Face Papers](https://img.shields.io/badge/Hugging%20Face-Papers-ffcc4d.svg)](https://huggingface.co/papers/2605.13030)

**FeatCal: Feature Calibration for Post-Merging Models** 的官方代码。

这个自包含 demo 复现了 CLIP ViT-B/32 8 任务 Task Arithmetic + FeatCal 结果。

## 新闻

- **2026-05-14：** 论文发布于 arXiv 和 Hugging Face Papers。
- **2026-05-08：** 发布 demo 代码，用于复现 CLIP ViT-B/32 8 任务
  Task Arithmetic + FeatCal 结果。

## 论文概览

模型合并会把多个任务专家模型合并为一个模型，但合并后的模型仍可能弱于各个专家模型。
FeatCal 从特征漂移的角度研究这一差距：即同一输入在合并模型和任务专家模型中产生的特征差异。
随后，FeatCal 使用少量校准样本，按前向顺序逐层校准合并模型。

FeatCal 使用闭式解更新模型权重。它不使用梯度下降、迭代优化、额外模块，也不引入辅助推理路径。

![FeatCal overview](assets/intro_triptych.png)

## 论文亮点

本仓库目前提供 CLIP ViT-B/32 8 任务 Task Arithmetic + FeatCal 设置的轻量级复现。
更完整的 CLIP、FLAN-T5 和 MergeBench 结果请见论文。

| 设置 | 基线 | + FeatCal |
|---|---:|---:|
| CLIP ViT-B/32 TA, 8 tasks | 67.5 | 85.5 |
| FLAN-T5-base GLUE TA | 78.9 | 85.2 |
| Llama-3.2-3B MergeBench TA | 60.1 | 62.1 |
| Llama-3.1-8B MergeBench TA | 63.5 | 65.8 |

## Demo 设置

- 骨干模型：`openai/clip-vit-base-patch32`
- 任务：SUN397, Stanford Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD
- 合并基线：Task Arithmetic，缩放因子 `0.3`
- 合并后方法：FeatCal
- 校准样本数：每个任务 `256` 个
- 复现超参数：`lambda_ratio=0.05`,
  `anchor_blend_rho=2.2`, `teacher_interp_alpha=0.25`

上面的 FeatCal 脚本参数与本 demo 使用的参考运行一致。
论文正文中的默认值为 `rho=2.0, alpha=0.3`；两组配置都较为保守，
但下方数字使用与脚本一致的配置。

## 预期结果

在完整测试集评测下，预期准确率为：

| 方法 | 平均准确率 |
|---|---:|
| Task Arithmetic | 67.55 |
| Task Arithmetic + FeatCal | 85.47 |

逐任务参考准确率：

| 任务 | TA | TA + FeatCal |
|---|---:|---:|
| SUN397 | 57.01 | 70.07 |
| Stanford Cars | 55.70 | 72.52 |
| RESISC45 | 64.75 | 87.94 |
| EuroSAT | 73.30 | 96.22 |
| SVHN | 77.93 | 95.06 |
| GTSRB | 68.50 | 93.21 |
| MNIST | 96.07 | 98.81 |
| DTD | 47.13 | 69.95 |

由于依赖版本和 dataloader 顺序不同，数值可能出现小幅差异；
但在相同随机种子和完整数据集下，平均结果应当非常接近。

## 运行效率

以下统计为 `n=256` 时 Task Arithmetic 之后的运行开销，不包含最终评测。
加速比以 Surgery 为参照。

| 方法 | 加速比（时间） | GPU 能耗 (Wh) | CPU RSS (GiB) |
|---|---:|---:|---:|
| Surgery | 1.0x (217s) | 18.1 | 69.3 |
| ProbSurgery | 1.0x (224s) | 18.3 | 87.0 |
| FeatCal | **4.1x (53s)** | **1.9** | **22.8** |

## 克隆

```bash
git clone https://github.com/egangu/featcal.git
cd featcal
```

## 安装

创建一个轻量级 Python 环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

对于 CUDA 机器，请先安装与 CUDA 驱动匹配的 PyTorch 构建版本，
再运行 `pip install -e .`。例如：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

首次运行会从 Hugging Face 下载 CLIP 权重和数据集。运行前可以设置标准缓存变量：

```bash
export HF_HOME=$PWD/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
```

如果你的网络需要 Hugging Face 镜像或 HTTP(S) 代理，请在启动 demo 前在 shell 中设置相应环境变量。
本仓库不会硬编码任何镜像或代理地址。

## 运行

完整复现：

```bash
bash scripts/run_ta8.sh
```

等价的 Python 命令：

```bash
python -m featcal.run_ta8 \
  --output-dir outputs/clip-vit-b32-ta8 \
  --device auto
```

输出将写入：

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

`run.log` 是单行 JSON 报告，用于兼容参考日志。
`metrics.json` 是同一报告的易读格式。

## 快速检查

在极小的校准和评测子集上运行一个低成本的双任务检查：

```bash
bash scripts/run_smoke.sh
```

这只用于软件检查，不用于复现论文表格。

## 常用选项

调试时在小子集上评测：

```bash
python -m featcal.run_ta8 --max-eval-samples 512
```

使用指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python -m featcal.run_ta8 --device cuda
```

复用已有的 Task Arithmetic checkpoint：

```bash
python -m featcal.run_ta8 \
  --skip-ta \
  --ta-model-path outputs/clip-vit-b32-ta8/ta \
  --output-dir outputs/clip-vit-b32-ta8
```

只运行 Task Arithmetic 基线：

```bash
python -m featcal.run_ta8 --skip-featcal
```

## 实现说明

Task Arithmetic 的实现形式为：

```text
theta_TA = theta_base + 0.3 * sum_i(theta_expert_i - theta_base)
```

FeatCal 按前向顺序校准 CLIP 视觉编码器层。对于每一层，
它会在校准 split 上收集当前合并模型特征和任务专家特征，
然后通过闭式解更新 Linear 权重、Linear bias 和 LayerNorm 仿射参数。
测试数据仅用于最终评测。

## 引用

如果你觉得本仓库有帮助，请引用：

```bibtex
@misc{gu2026featcal,
      title={FeatCal: Feature Calibration for Post-Merging Models},
      author={Yanggan Gu and Shuo Cai and Zihao Wang and Wenjun Wang and Yuanyi Wang and Pengkai Wang and Sirui Huang and Su Lu and Jianmin Wu and Hongxia Yang},
      year={2026},
      eprint={2605.13030},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.13030},
}
```
