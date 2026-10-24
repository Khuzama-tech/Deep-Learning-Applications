<div align="center">

# 🧠 Project 2, Week 5: Scene Classification with Transfer Learning
### **Introduction to Applied AI | BS 7th Semester | Department of Electronics**
**Student Name:** Khuzaima Ahmed | **Platform:** Kaggle Notebook (GPU T4)


</div>

---

## 📌 Project Overview
This repository contains the complete implementation and deliverables for **Lab 5 (Project 2: Deep Learning Applications)**. The core objective of this project is to build, train, evaluate, and fine-tune a convolutional neural network (CNN) image classifier using **PyTorch** and a pre-trained **ResNet-18** architecture on the Intel Image Classification dataset. 

---

## 📊 Key Performance Metrics & Results

| Training Phase / Evaluation | Details & Architecture Configuration | Performance Result |
| :--- | :--- | :--- |
| **Dataset** | Intel Image Classification (6 Classes: *Buildings, Forest, Glacier, Mountain, Sea, Street*) | 11,944 Train / 3,000 Test |
| **Base Model** | ResNet-18 pretrained on ImageNet (Backbone Frozen) | Initialized |
| **Feature Extraction** | Trained 3,078 parameters (Linear Head only) for 3 Epochs | **~85.1%** Validation Accuracy |
| **Fine-Tuning** | Unfrozen `layer4` (~8.4M parameters) with differential learning rates (`1e-4` & `1e-3`) | **~88.5%** Validation Accuracy |
| **Final Test Evaluation** | Evaluated once on the unseen test split using best model weights | **89.2%** Test Accuracy |

---

## 🔍 Detailed Lab Structure & Workflow

1. **The Data & Tensor Representation:** Verified CUDA hardware acceleration, loaded the Intel Image Classification dataset, inspected sample tensors (`150x150x3`, `uint8`, range `0-255`), and anticipated structural confusions between urban classes.
2. **Convolutions & Parameters by Hand:** Manually verified 2D spatial convolution filters using synthetic matrices and real grayscale image edge detection, alongside parameter scaling proofs comparing dense layers vs. convolutional weight-sharing.
3. **Data Pipelines & Augmentation:** Configured ImageNet normalization statistics (`MEAN, STD`) and applied rigorous training-only augmentations (`RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`) with synchronized data loaders.
4. **Feature Extraction:** Built modular training/evaluation loops (`run_epoch`, `fit`), froze the ResNet-18 feature extractor backbone, and trained the 3,078-parameter classification head.
5. **Fine-Tuning & Learning Curves:** Unfrozen the final residual block (`layer4`) with differential learning rates and plotted convergence trajectories, confirming stable loss reduction without severe overfitting.
6. **Error Analysis & Gallery:** Evaluated test performance via classification reports, confusion matrices, and an error gallery. The weakest classes were `buildings` and `street`, most frequently confused due to overlapping urban architectural facades.
7. **Model Serialization:** Exported trained model weights (`resnet18_intel.pt`, ~44 MB) to disk for downstream deployment readiness.

---

## ⚠️ Most Instructive Error Analysis
> *"An image depicting a tight urban alleyway architectural facade labeled under `street` was misclassified as `buildings` with high confidence, illustrating genuine visual overlap where even human judgment faces semantic ambiguity between dense structural settings."*

---

## 🚀 Next: Week 6 Project Proposal
* **Domain:** Industrial Manufacturing Defects Inspection (Quality Control).
* **Problem & Beneficiaries:** Automated surface defect detection (such as cracks, surface scratches, corrosion, or casting flaws) on manufactured components to assist quality assurance engineers and production lines in catching defects early, reducing waste, and ensuring high hardware reliability.
* **Candidate Dataset:** *Severstal: Steel Defect Detection* or *NEU Surface Defect Database* (Kaggle), featuring thousands of labeled high-resolution industrial surface images categorized into distinct defect classes under an open-source license.
* **Scope:** 4 target defect classes, leveraging transfer learning with ResNet-18/50.

---

## 🏭 Week 6: Casting Defect Detection with ResNet-18

---

### 📌 Overview & Value Proposition

| Key Aspect | Details |
| :--- | :--- |
| **Who Benefits** | **Quality Control Engineers & Plant Operators** in **Foundry & Automotive Component Manufacturing Settings**. |
| **Why It Matters** | Prevents structural failures in cast pump impellers and reduces scrap waste costs. Shipping a defective component downstream causes exponential financial cost, potential assembly line shutdown, or field failure risks. |

---

### 📊 Dataset Details & Audit

| Parameter | Specification / Audit Result |
| :--- | :--- |
| **Dataset** | **Casting Product Quality Dataset** ([Kaggle Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product), Open/CC BY-SA 4.0); **7,300+ total grayscale images** across **2 classes** (`def_front` vs. `ok_front`); labeled by **Quality Assurance Inspectors / Domain Experts**. |
| **Data Audit** | • **Imbalance Ratio**: ~$1.3:1$ (`def_front` vs `ok_front`)<br>• **Exact Duplicates**: `128`<br>• **Test Near-Duplicates in Train**: `327` of `715`]<br>• **Split Strategy**: Split by **part / batch**, **0 shared groups** between train and val|
| **Imbalance Handling** | **Weighted Loss / Class Weights** using cross-entropy with weights `{'def_front': 0.883, 'ok_front': 1.154}`. |

---

### 📈 Evaluation & Test Results

> **Note**: Test set evaluated strictly **once** after pipeline calibration.

| Metric | Target / Measured Value |
| :--- | :--- |
| **Macro-F1** | **0.987** |
| **Test AUC** | **0.999** |
| **Sensitivity, Specificity & PPV** | Sensitivity **0.991**, Specificity **1.000**, and PPV **1.000** at threshold **0.185** (chosen on val for recall $\ge 0.95$) |

---

### 🔍 Interpretability & Risk Assessment

| Aspect | Analysis |
| :--- | :--- |
| **Grad-CAM Focus** | Visual heatmaps confirm the model looks directly at **surface casting defect regions** on the casting body. |
| **Shortcut Risk** | Checked and **none found**; model is not learning background fixtures, casting ring edges, or lighting gradients. |

---

### ⚠️ Limitations & Deployment Constraints

- **Data Source & Scope**: Industrial images captured under specific foundry lighting and camera setups.
- **Not For Use**: Not for structural load testing or non-metal castings, nor for safety-critical structural failure prediction without human operator sign-off.

---


```text
├── README.md                                             # Comprehensive project documentation & model card
├── week-5-transfer-learning.ipynb                         # Week 5 transfer learning baseline notebook
└── week06-industrial-manufacturing-defects-inspection.ipynb # Week 6 industrial QC & defect detection notebook
