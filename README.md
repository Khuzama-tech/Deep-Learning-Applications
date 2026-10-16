<div align="center">

# 🧠 Project 2, Week 5: Scene Classification with Transfer Learning


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
* **Domain:** Plant Leaf Disease Detection in Agriculture.
* **Problem & Beneficiaries:** Automated early-stage plant pathology classification to assist farmers and agricultural extension workers in maximizing crop yield and minimizing chemical overuse.
* **Candidate Dataset:** *PlantVillage Dataset* (Kaggle), featuring thousands of labeled high-resolution images spanning multiple crop species and healthy/diseased leaf categories under an open-source license.
* **Scope:** 4-6 target classes, leveraging transfer learning with ResNet-18/50.

---

## 🛠️ Repository Structure
```text
├── notebook/
│   └── Week_5_Transfer_Learning.ipynb   # Complete Kaggle notebook with code and markdown cells
├── models/
│   └── resnet18_intel.pt                # Saved model checkpoint weights (~44 MB)
└── README.md                            # Comprehensive project documentation
