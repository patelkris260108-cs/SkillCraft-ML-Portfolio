# Task 3: Dogs vs. Cats — Dataset Setup & Acquisition Guide

## SkillCraft Technology — Machine Learning Internship

This guide provides instructions for manually acquiring and placing the official Kaggle Dogs vs. Cats dataset.

---

### 1. Dataset Origin & Overview
- **Competition**: Kaggle Dogs vs. Cats (Kernels Edition / Computer Vision Benchmark)
- **Competition URL**: [https://www.kaggle.com/c/dogs-vs-cats/data](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Source Files on Kaggle**:
  - `train.zip` (~543 MB compressed): Contains 25,000 labeled JPEG images (`cat.0.jpg` ... `cat.12499.jpg` and `dog.0.jpg` ... `dog.12499.jpg`).
  - `test1.zip` (~271 MB): Unlabeled test images (not used for model evaluation).
  - `sampleSubmission.csv`: Benchmark submission template.

---

### 2. Manual Download Steps (User Action Required)

Please follow these steps on your computer:

1. **Navigate to the Kaggle Dataset Page**:
   Open your web browser and go to:
   ```text
   https://www.kaggle.com/c/dogs-vs-cats/data
   ```
2. **Authenticate & Accept Rules**:
   Sign in to your Kaggle account. If prompted, review and accept the competition rules to unlock data downloads.
3. **Download `train.zip`**:
   In the **Data Explorer** panel, locate and download `train.zip`.
   *(Note: `test1.zip` is not required for training or benchmark evaluation).*
4. **Place the Archive in the Project Directory**:
   Move or copy the downloaded `train.zip` file directly into:
   ```text
   C:\Users\HP\my new project (skillcraft)\SkillCraft-ML-Portfolio\task3-dogs-vs-cats\data\raw\train.zip
   ```
   *Do NOT extract the archive manually. The project dataset validator and ingestion pipeline inspect and extract only what is needed in a memory-safe, bounded manner.*
5. **Resume in Antigravity**:
   Return to your Antigravity conversation and prompt:
   > *"I have placed train.zip at task3-dogs-vs-cats/data/raw/train.zip. Please proceed with dataset validation."*

---

### 3. Security & Safety Principles
- **No API Tokens**: You will never be asked to enter, commit, or provide Kaggle API credentials, passwords, or `kaggle.json` files.
- **Archive Preservation**: The original `train.zip` archive is treated as strictly read-only and will never be overwritten, modified, or deleted by project code.
- **Memory Safety**: The validation tool reads image metadata and streams image headers in bounded chunks, preventing large memory spikes on machines with limited free RAM.

---

### 4. Target Benchmark Configuration
Once `train.zip` is validated, the pipeline selects a balanced **4,000-image benchmark subset**:
- **Cats**: 2,000 valid images
- **Dogs**: 2,000 valid images
- **Split Ratio**: 70% Training (2,800 images), 15% Validation (600 images), 15% Test (600 images)
- **Seed**: `random_state=42`
- **Integrity**: Mutually exclusive partitions recorded in a reproducible manifest.
