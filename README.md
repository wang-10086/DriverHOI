<div align="center">

# 🚄 DriverHOI

### **Train Driver Human–Object Interaction Recognition Based on Graph Parsing Neural Networks**

<p>
  <a href="#"><img src="https://img.shields.io/badge/Paper-RESS-b31b1b.svg" alt="Paper"></a>
  <a href="#dataset"><img src="https://img.shields.io/badge/Dataset-DriverHOI3D-blue" alt="Dataset"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-97CA00.svg" alt="License"></a>
</p>

**[Kun Wang](mailto:23120254@bjtu.edu.cn)&ensp;·&ensp;[Haifeng Bao](mailto:hfbao@bjtu.edu.cn)*&ensp;·&ensp;[Weining Fang](mailto:wnfang@bjtu.edu.cn)**

State Key Laboratory of Advanced Rail Autonomous Operation, Beijing Jiaotong University

</div>

> 📄 Official implementation of the **DriverHOI** model from our paper submitted to *Reliability Engineering & System Safety*.

---

## Overview

Targeting the spatially dense layout of display and control devices on train driving consoles, we designed the **DriverHOI** model based on **Graph Parsing Neural Networks (GPNN)**. By fusing 3D hand poses and device geometric priors, the model constructs a "Driver–Device" heterogeneous graph and performs iterative message passing, achieving joint inference of driving actions and interaction objects.

---

## Key Results

Performance under **10-fold Leave-One-Subject-Out (LOSO) cross-validation**:

<div align="center">

| Metric | Score |
|:------:|:-----:|
| **Overall Accuracy** | **94.0%** |
| Action Accuracy | 98.3% |
| Interaction Accuracy | 99.1% |
| Device Top-1 Accuracy | 94.6% |
| Device Top-3 Accuracy | 99.8% |

</div>

---

## Dataset

The **DriverHOI3D** dataset was collected in a high-fidelity 1:1 train driving simulator. It contains **1,856 samples** from **10 subjects**, covering **47 valid action–device interaction pairs**.

| Item | Detail |
|------|--------|
| **Subjects** | 10 participants |
| **Samples** | 1,856 |
| **Actions** | 4 types — Point, Press, Push, Swing |
| **Devices** | 31 console devices + 1 "No Interaction" |
| **Camera Views** | 4 synchronized views per sample |
| **Annotations** | RGB image, 3D hand keypoints (21 joints/hand), camera parameters |

<details>
<summary><b>📂 Data Directory Structure</b> (click to expand)</summary>

```
DriverHOI3D/
├── calibration/
│   ├── intri.yml
│   └── extri.yml
├── subject1/
│   ├── point/
│   │   ├── device0/
│   │   │   ├── MBP25030012/       # Camera view 1
│   │   │   ├── MBP25030014/       # Camera view 2
│   │   │   ├── MBP25030016/       # Camera view 3
│   │   │   ├── MBP25030017/       # Camera view 4
│   │   │   └── post_json/         # 3D keypoint annotations
│   │   ├── device1/
│   │   └── ...
│   ├── press/
│   ├── push/
│   └── swing/
├── subject2/
└── ...
```

</details>

---

## Installation

```bash
git clone https://github.com/wang-10086/DriverHOI.git
cd DriverHOI
pip install -r requirements.txt
```
---

## Usage

### 1️⃣ Configuration

Edit `config.py` to set your dataset path:

```python
DATA_ROOT = "/path/to/DriverHOI3D"
```

### 2️⃣ Training (Single Fold)

```bash
python train.py --test_subj subject1 --exp_name subject1 --model_type DriverHOI --num_epochs 50 --lr 1e-3
```

### 3️⃣ 10-Fold LOSO Cross-Validation

```bash
python run_cv.py --model_type DriverHOI --job_name driverhoi_exp01
```

This sequentially trains 10 folds, each time leaving one subject out for testing.

### 4️⃣ Evaluation

**Single fold** — edit `CKPT_PATH` and `TEST_SUBJ` in `test.py` before running:

```bash
python test.py
```

**Batch evaluation (all folds)** — edit `JOB_NAME` in `test_cv.py` to match the training job name:

```bash
python test_cv.py
```

---

## Ablation Study

The framework supports four ablation configurations to verify the contribution of each input feature:

<div align="center">

| Mode | Pose | Geom | Visual | Act Acc | Top-1 Acc | Overall Acc |
|:----:|:----:|:----:|:------:|:-------:|:---------:|:-----------:|
| **Baseline** | ✅ | ✅ | ❌ | 98.3% | 94.6% | **94.0%** |
| No_Pose | ❌ | ✅ | ❌ | 89.7% | 94.1% | 84.9% |
| No_Geom | ✅ | ❌ | ❌ | 97.7% | 56.6% | 55.5% |
| With_Visual | ✅ | ✅ | ✅ | 92.8% | 93.9% | 88.1% |

</div>

```bash
python run_cv.py --model_type DriverHOI --ablation no_pose --job_name ablation_no_pose
python run_cv.py --model_type DriverHOI --ablation no_geom --job_name ablation_no_geom
python run_cv.py --model_type DriverHOI --ablation with_visual --job_name ablation_visual
```

---

## Comparison Models

We compare three interaction reasoning mechanisms under the same feature encoding:

<div align="center">

| Model | Mechanism | Act Acc | Top-1 Acc | Overall Acc |
|:-----:|:---------:|:-------:|:---------:|:-----------:|
| TransHOI | Attention-based | 97.2% | 92.9% | 90.6% |
| SCG-HOI | SCG-based (static graph) | 96.8% | 91.6% | 88.4% |
| **DriverHOI** | **GPNN (dynamic graph)** | **98.3%** | **94.6%** | **94.0%** |

</div>

```bash
python run_cv.py --model_type TransHOI  --job_name transhoi_exp01
python run_cv.py --model_type SCG-HOI   --job_name scghoi_exp01
```

---

## License

This project is released for **academic research purposes**.