# MambaPSA: A Mamba-based Replacement for C2PSA in YOLO26

<p align="center">
  <em>A lightweight state-space attention module for real-time object detection</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.12681"><img src="https://img.shields.io/badge/arXiv-2607.12681-b31b1b" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/Paper-IET%20ICETA%202026-blue" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/Framework-Ultralytics%20YOLO26-green" alt="Framework"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-AGPL--3.0-orange" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.x-red" alt="PyTorch"></a>
</p>

---

## Overview

**MambaPSA** replaces the self-attention-based **C2PSA** module in YOLO26 with a **Mamba (selective state-space) block**, reducing computational complexity from quadratic to linear in the number of spatial tokens while maintaining detection accuracy. This repository contains the full implementation, model configurations, and training scripts used in our paper.

> **Paper:** *MambaPSA: A Mamba-based Replacement for C2PSA in YOLO26*
> Sheng-Wei Chan, Chia-Min Lin, Hsin-Jui Pan, Ching-Yu Tsai, Chih-Hsiang Yang, Yung-Che Wang, Jen-Shiun Chiang\*
> Department of Electrical and Computer Engineering, Tamkang University, New Taipei City, Taiwan
> Submitted to **IET ICETA 2026**. (\*Corresponding author)
> 📄 Paper: **[arXiv:2607.12681](https://arxiv.org/abs/2607.12681)**


The core idea: the C2PSA block in YOLO26 applies position-sensitive self-attention over spatial tokens, incurring **O(N²)** cost that scales quadratically with token count and constrains its use in lightweight edge deployments. MambaPSA substitutes this with a selective state-space model that captures long-range dependencies in **O(N)**, making the module more efficient — particularly on CPU, where attention is a bottleneck.

---

## Key Contributions

- **MambaPSA block** — a Mamba-based replacement for the C2PSA block at the end of the YOLO26 backbone. It preserves the CSP wrapper of C2PSA but replaces the internal self-attention branch with a Mamba core (`d_state=8`, `e=1`, mono-directional scan), keeping the module approximately parameter-neutral.
- **BiViM placement study** — a bidirectional Vision Mamba block (`d_state=16`, `e=2`) inserted at the P3, P4, or P5 neck levels, revealing a **non-monotonic** relationship between placement and accuracy.
- **Linear-complexity feature aggregation** — reduces FLOPs by 12.1% while keeping accuracy on par with the attention-based baseline.
- **CPU inference speedup** — +17.6% throughput on CPU, where the quadratic cost of attention is most limiting.

---

## Results

Trained on **PASCAL VOC** (2007+2012 trainval, 16,551 images) and evaluated on **VOC 2007 test** (4,952 images, 20 classes). All models trained for 100 epochs under identical settings.

Relative to the YOLO26n baseline, **MambaPSA** reduces parameters by **2.9%** and FLOPs by **12.1%**, with negligible accuracy change (**−0.1 mAP@50:95**). Inserting a bidirectional Vision Mamba (**BiViM**) block at the **P4** neck level yields the best accuracy gain (**+0.9 mAP@50:95**).

| Model | Params | FLOPs | mAP@50:95 | Notes |
|-------|:------:|:-----:|:---------:|-------|
| YOLO26n (baseline) | — | — | — | C2PSA self-attention |
| **MambaPSA** | **−2.9%** | **−12.1%** | **−0.1** | Mamba replaces C2PSA |
| MambaPSA + BiViM @ P4 | — | — | **+0.9** | best accuracy |

**CPU throughput** (Intel Core i7-9700K): **17 → 20 FPS** (**+17.6%**) versus the baseline.

> See `runs/` for full training logs and per-model metrics.

---

## Repository Structure
---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.x with CUDA (Mamba's selective scan requires a CUDA-capable GPU)
- NVIDIA GPU (the `mamba-ssm` kernels are CUDA-only)

### Setup

```bash
# Clone the repository
git clone https://github.com/henrychan0719/yolov26-mamba.git
cd yolov26-mamba

# Install dependencies
pip install -r requirements.txt
```

> **Note:** MambaPSA relies on the `mamba-ssm` selective-scan CUDA kernels and therefore requires an NVIDIA GPU for training and GPU inference. On CPU, the block falls back gracefully.

---

## Usage

### Training

```bash
python train.py \
  --model yolo26n-mambapsa.yaml \
  --data VOC.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 32
```

Or with the Ultralytics CLI:

```bash
yolo detect train model=yolo26n-mambapsa.yaml data=VOC.yaml epochs=100 imgsz=640 batch=32 optimizer=AdamW lr0=0.001
```

> Training configuration follows the paper: AdamW optimizer, learning rate `1e-3`, batch size `32`, 100 epochs, image size `640`.

### Validation

```bash
python val.py --model yolo26n-mambapsa.yaml --weights path/to/best.pt --data VOC.yaml
```

### Inference

```python
from ultralytics import YOLO

model = YOLO("yolo26n-mambapsa.yaml")
model.load("path/to/best.pt")
results = model("path/to/image.jpg")
results[0].show()
```

---

## Model Variants

This repository includes several configurations to reproduce the ablation study:

| Config | Description |
|--------|-------------|
| `yolo26n-mambapsa.yaml` | **Main model** — Mamba replaces C2PSA |
| `yolo26n-mamba-p3.yaml` | Mamba block placed at the P3 level |
| `yolo26n-mamba-p4.yaml` | Mamba block placed at the P4 level (best accuracy) |
| `yolo26n-mamba.yaml` | Mamba applied in the backbone |
| `yolo26n-dualmamba.yaml` | Dual (two-branch) Mamba variant |

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{mambapsa2026,
  title         = {MambaPSA: A Mamba-based Replacement for C2PSA in YOLO26},
  author        = {Chan, Sheng-Wei and Lin, Chia-Min and Pan, Hsin-Jui and
                   Tsai, Ching-Yu and Yang, Chih-Hsiang and Wang, Yung-Che and
                   Chiang, Jen-Shiun},
  journal       = {arXiv preprint arXiv:2607.12681},
  year          = {2026},
  eprint        = {2607.12681},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

---

## Acknowledgements

- Built on [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics).
- Mamba selective state-space model based on [state-spaces/mamba](https://github.com/state-spaces/mamba).
- Supported by the National Science and Technology Council (NSTC), Taiwan, under grant **114-2221-E-032-011-**.

---

## License

This project is released under the **AGPL-3.0 License**, consistent with the Ultralytics YOLO framework. See [LICENSE](LICENSE) for details.

---

## Contact

For questions regarding the paper or implementation, please open an issue or contact the corresponding author:

**Jen-Shiun Chiang** — `chiang@mail.tku.edu.tw`
Department of Electrical and Computer Engineering, Tamkang University, New Taipei City, Taiwan.
