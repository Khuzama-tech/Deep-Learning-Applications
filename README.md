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
* **Candidate Dataset:** *Severstal: Steel Defect Detection* or *casting product image data for quality inspection* (Kaggle), featuring thousands of labeled high-resolution industrial surface images categorized into distinct defect classes under an open-source license.
* **Scope:** 4 target defect classes, leveraging transfer learning with ResNet-18/50.

---


## 🏭 Week 6: Casting Defect Detection with ResNet-18

---

### 📌 Overview & Value Proposition

| Key Aspect | Details |
| :--- | :--- |
| **Who Benefits** | **Quality Control Engineers & Plant Operators** in foundry and automotive component manufacturing settings. |
| **Why It Matters** | Helps identify defective casting products, reduce scrap waste, and improve quality control. Detecting defective components before they move to the next production stage can reduce financial losses and potential manufacturing problems. |
| **Model** | ResNet-18 pretrained on ImageNet and fine-tuned for binary casting defect classification. |

---

### 📊 Dataset Details & Audit

| Parameter | Specification / Audit Result |
| :--- | :--- |
| **Dataset** | **Casting Product Quality Dataset** ([Kaggle Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)) |
| **Total Images** | 6,633 grayscale images |
| **Classes** | 2 (`def_front` and `ok_front`) |
| **Defective Images** | 3,758 |
| **Non-Defective Images** | 2,875 |
| **Imbalance Ratio** | Approximately 1.3:1 |
| **Image Dimensions** | 300 × 300 pixels |
| **Exact Duplicates** | 128 |
| **Test Near-Duplicates in Train** | 327 out of 715 test images |
| **Split Strategy** | Group-aware train/validation split with 0 shared groups between training and validation sets. |

#### Dataset Split

| Dataset | Number of Images |
| :--- | ---: |
| Training | 5,527 |
| Validation | 1,106 |
| Testing | 715 |
| **Total** | **7,348** |

**Note:** The split counts above are from a separate group-aware split output and do not add up to the original dataset count of 6,633. Verify the split source and dataset version before publishing these counts together.

#### Imbalance Handling

Class weights were calculated using weighted cross-entropy loss.

```python
class_weights = {
    'def_front': 0.883,
    'ok_front': 1.153
}
```

These weights help balance the contribution of both classes during model training.

---

### 🧠 Model Training: ResNet-18

- **Architecture:** ResNet-18
- **Pretrained Weights:** ImageNet
- **Trainable Parameters:** 8,394,754
- **Training Strategy:** Two-phase transfer learning.

#### Training Phases

| Phase | Training Approach | Validation Macro-F1 |
| :--- | :--- | ---: |
| Phase 1 – Epoch 1 | Train classification head | 0.906 |
| Phase 1 – Epoch 2 | Train classification head | 0.949 |
| Phase 2 – Epoch 1 | Fine-tune last ResNet block | 0.989 |
| Phase 2 – Epoch 2 | Fine-tune last ResNet block | 0.982 |
| Phase 2 – Epoch 3 | Fine-tune last ResNet block | 0.995 |

**Final Validation Macro-F1: 0.9954**

---

### 📈 Evaluation & Test Results

The original test classification report produced the following results:

| Metric | Defective (`def_front`) | Non-Defective (`ok_front`) |
| :--- | ---: | ---: |
| **Precision** | 1.000 | 0.989 |
| **Recall** | 0.993 | 1.000 |
| **F1-Score** | 0.997 | 0.994 |
| **Support** | 453 | 262 |

#### Overall Test Performance

| Metric | Measured Value |
| :--- | ---: |
| **Test Accuracy** | 0.996 |
| **Macro Precision** | 0.994 |
| **Macro Recall** | 0.997 |
| **Macro-F1** | **0.995** |
| **Weighted-F1** | 0.996 |

---

### 🎯 Additional Threshold Evaluation

A separate threshold evaluation produced the following results:

| Metric | Measured Value |
| :--- | ---: |
| **Validation Threshold** | 0.4830 |
| **Validation Recall** | 1.000 |
| **Test AUC** | 0.4682 |
| **Sensitivity** | 1.000 |
| **Specificity** | 0.000 |
| **PPV** | 0.6336 |
| **True Positives (TP)** | 453 |
| **False Negatives (FN)** | 0 |
| **True Negatives (TN)** | 0 |
| **False Positives (FP)** | 262 |

**Important:** This separate threshold evaluation conflicts with the original classification report. It indicates that every test image was classified as defective. Therefore, the model's class mapping, checkpoint, preprocessing, test loader, and threshold selection must be checked before reporting threshold-based performance as final.

---

### 🔍 Interpretability & Risk Assessment

| Aspect | Analysis |
| :--- | :--- |
| **Grad-CAM Target Layer** | `layer4.1.conv2` |
| **Visualization Samples** | 6 examples: 3 correctly classified and 3 incorrectly classified. |
| **Interpretability** | Grad-CAM was used to visualize the image regions influencing model predictions. |
| **Shortcut Risk** | Heatmaps should be inspected to determine whether the model focuses on actual casting defects rather than background, borders, or lighting artifacts. |

**Note:** Grad-CAM visualizations were generated, but the heatmap results need visual inspection before concluding that shortcut learning is absent.

---

### ⚠️ Limitations & Deployment Constraints

- **Data Source & Scope:** The dataset contains industrial casting images captured under specific imaging conditions.
- **Duplicate Images:** Exact and near-duplicate images were identified, so data leakage must be carefully controlled.
- **Generalization:** Performance on other manufacturing environments, camera setups, and lighting conditions has not yet been established.
- **Human Verification:** Predictions should be reviewed by quality control personnel before making production decisions.
- **Not For Use:** This model is not a replacement for structural load testing or engineering safety certification.

---

### 🏁 Conclusion

In Week 6, a pretrained ResNet-18 model was developed for binary casting defect classification. The original test classification report achieved **99.6% accuracy and a 0.995 macro-F1 score**.

The project also explored class weighting, transfer learning, threshold evaluation, and Grad-CAM interpretability. However, the conflicting threshold evaluation results must be investigated before considering the model ready for industrial deployment.


---


# 🏭 Week 7: Text and Multimodal AI
### Industrial Manufacturing Defects Inspection

> Applying NLP, OCR, Named Entity Recognition, and Deep Learning to extract meaningful information from industrial inspection reports.

## 📌 Project Overview

In Week 7, I explored how AI can understand and extract information from text and industrial inspection documents. I implemented text classification, Named Entity Recognition (NER), regex-based field extraction, OCR error correction, and privacy masking.

I also integrated my Week 6 ResNet-18 image classification model with text-based industrial evidence to develop a multimodal inspection workflow.

## 1. 📊 Text Classification

**Dataset:** SMS Spam Collection  
**Total Texts:** 5,572

| Feature Extraction | Macro-F1 | Number of Features |
|---|---:|---:|
| Word 1–2 grams | 0.974 | 12,128 |
| Character 2–5 grams | 0.982 | 36,449 |

**Classification Results:**
- Accuracy: 98.8%
- Macro-F1: 0.974
- Spam Precision: 0.979
- Spam Recall: 0.933

**Top Spam-like Features:**
`call, txt, free, text, reply, uk, www, mobile, claim, stop`

**Top Ham-like Features:**
`my, me, ok, that, it, but, ll, gt, lt, can, what`

Character n-grams performed better than word n-grams because they can recognize unusual spellings and modified spam words.

## 2. 🏷️ Named Entity Recognition (NER)

I applied spaCy NER to my Industrial Manufacturing Defects Inspection domain.

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| spaCy Out-of-the-Box | 0.571 | 0.500 | 0.533 |
| spaCy + EntityRuler | 0.875 | 0.875 | 0.875 |

**Improvement:** 0.342 in F1-score.

The EntityRuler improved the recognition of domain-specific entities, including:
- PERSON
- ORG
- GPE / LOC
- DATE
- PRODUCT
- BATCH_ID
- DEFECT

This demonstrated the importance of custom entity patterns for industrial inspection reports.

## 3. 🔍 Regex-Based Field Extraction

I implemented regex-based extraction to identify important information from industrial and business documents.

**Successfully extracted fields:**
- CNIC numbers
- Mobile numbers
- Email addresses
- Financial amounts
- Dates
- Invoice numbers

**Test Results:**
- CNIC and mobile extraction: True
- Amount and date extraction: True
- Email and invoice extraction: True
- Long order number: False
- Room number and floor: False

The negative tests helped verify that unrelated numbers were not incorrectly extracted as personal identifiers.

## 4. 📄 OCR and Error Correction

OCR was used to extract text from documents and identify common character recognition errors.

| OCR Example | Raw CER | Corrected CER |
|---|---:|---:|
| Invoice N0. 4417 | 0.062 | 0.000 |
| Total: PKR 12,5O0 | 0.059 | 0.000 |
| Due 3O/09/2O26 | 0.143 | 0.000 |

Correction rules successfully fixed common OCR errors, including:
- `N0.` → `No.`
- `5O0` → `500`
- `3O/09/2O26` → `30/09/2026`

**Additional industrial document OCR test:**
- Raw CER: 0.008
- After correction: 0.008

## 5. 🏭 Industrial Domain Text Processing

I created domain-specific sentences related to industrial manufacturing and casting defects.

**Extracted information included:**
- People: Engineer Ali, Technician Ahmed, Engineer Sara
- Organizations: ABC Manufacturing, XYZ Industries
- Locations: Lahore, Karachi
- Batch ID: 4417
- Product ID: PI-204
- Defects: Porosity, shrinkage, cold shut, and cracks
- AI model: ResNet-18

These examples were used to develop a domain-specific NER dataset and practice industrial information extraction.

## 6. 🔐 Privacy and Personal Information Masking

Sensitive information was masked before logging and further processing.

| Original Information | Masked Output |
|---|---|
| CNIC number | [CNIC] |
| Mobile number | [PHONE] |
| Email address | [EMAIL] |

This approach helps protect personal information while preserving useful industrial inspection evidence.

## 7. 🤖 Multimodal AI Pipeline

I integrated my Week 6 ResNet-18 image model with text-based industrial evidence extraction.

**Image Model:**
- Architecture: ResNet-18
- Classes: `def_front`, `ok_front`
- Model loaded successfully.

**Industrial Text Evidence:**

| Field | Extracted Value |
|---|---|
| Batch ID | BATCH-2026-101 |
| Part ID | IM-2045 |
| Machine ID | MCH-07 |
| Temperature | 180 C |
| Dimensions | 25.5 mm |
| Weight | 2.5 kg |
| Date | 30/10/2026 |
| Rejection Rate | 12% |

**Detected Defects:**
- Porosity
- Crack
- Shrinkage

**Urgent Term:** Crack

**Final Decision:** `REFER`

**Important Note:** The image model loaded successfully, but the test image `casting_sample.jpg` was not found. Therefore, image probabilities were unavailable, and this example's decision was based on text evidence rather than completed multimodal fusion.

## 8. 📦 Final Pipeline Output

The system generated a structured JSON record containing inspection evidence.

```json
{
  "image": "casting_sample.jpg",
  "image_probabilities": {
    "status": "unavailable",
    "message": "Casting image file not found."
  },
  "note_masked": "Casting Batch BATCH-2026-101, Part IM-2045. Inspection detected porosity and shrinkage defects. A surface crack was also observed. Machine MCH-07, temperature 180 C, dimension 25.5 mm, rejection rate 12%. Contact: [MOBILE]",
  "text_evidence": {
    "detected_defects": [
      "porosity",
      "crack",
      "shrinkage"
    ],
    "urgent_terms": [
      "crack"
    ]
  },
  "decision": "REFER"
}
```

## 🎯 Key Learnings

- Implemented text classification using word and character n-grams.
- Achieved 0.982 macro-F1 with character n-grams.
- Improved spaCy NER F1 from 0.533 to 0.875 using EntityRuler.
- Extracted structured information using regex.
- Corrected common OCR recognition errors.
- Applied CNIC, phone, and email masking.
- Integrated ResNet-18 with industrial text evidence extraction.

## 🏁 Conclusion

Week 7 helped me understand how NLP, OCR, NER, and Deep Learning can work together in industrial inspection applications.

By combining image classification with text-based evidence extraction, I developed the foundation of an AI-assisted Industrial Manufacturing Defects Inspection system that can identify potential defects, extract important manufacturing information, and support quality control decisions.

**Next Goal:** Improve the multimodal pipeline by testing real casting images and combining image and text probabilities for more reliable inspection decisions.

---

### 🛠️ Technologies Used

`Python` `NLP` `spaCy` `TF-IDF` `Scikit-learn` `Regex` `OCR` `PyTorch` `ResNet-18` `Multimodal AI`

### 🔖 Tags

#IndustrialAI #Manufacturing #NLP #spaCy #OCR #NER #DeepLearning #ResNet18 #MultimodalAI #QualityControl #LearningInPublic

├── README.md                                                   # Comprehensive project documentation & model card
├── week-5-transfer-learning.ipynb                              # Week 5 transfer learning baseline notebook
├── week06-industrial-manufacturing-defects-inspection.ipynb   # Week 6 industrial QC & defect detection notebook
└── week07-text-and-multimodal-ai.ipynb                         # Week 7 text processing, NLP, OCR & multimodal AI
