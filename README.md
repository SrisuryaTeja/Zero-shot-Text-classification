# 🤖 Zero-Shot Text Classification

A modular framework for building and evaluating **zero-shot text classification** models using Bi-Encoder and Poly-Encoder architectures.

This project explores contrastive learning–based approaches for zero-shot classification and evaluates multiple transformer backbones including DistilBERT and MiniLM, with and without hard negative mining.

---

## 📖 Overview

Zero-shot text classification allows models to classify text into categories **without seeing labeled examples for those categories during training**.

Instead of training a traditional classifier head, this project:

- Encodes text and labels into embedding space
- Computes similarity between text and candidate labels
- Ranks labels using similarity scores
- Evaluates using ranking metrics (Precision@K, Recall@K, F1@K, MRR)

Two architectures are implemented:

- **Bi-Encoder**
- **Poly-Encoder**

---

## 🚀 Quick Start

### ✅ Prerequisites

- Python **3.11**
- (Optional) CUDA-enabled GPU for faster training/inference

---

### 📦 Installation

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/SrisuryaTeja/Zero-shot-Text-classification.git
cd Zero-shot-Text-classification
```

#### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

- Linux/macOS:
```bash
source venv/bin/activate
```

- Windows:
```bash
.\venv\Scripts\activate
```

#### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Data Preparation

The `data/` directory currently contains:

- **1916 generated samples**

If you would like to generate your own dataset:

```bash
python src/generate_data.py
```

You can modify generation parameters inside the script.

---

## ⚙️ Configuration

Model and inference parameters are defined in:

- `biencoder-config.yaml`
- `polyencoder-config.yaml`

You can configure:

- `model_name`
- `max_length`
- `batch_size`
- `learning_rate`
- `device` (`cpu` or `cuda`)
- training steps
- hard negative settings

---

## 🏗️ Training

### 🔹 Train Bi-Encoder

```bash
python -m scripts.biencoder_train
```

### 🔹 Train Poly-Encoder

```bash
python -m scripts.polyencoder_train
```

---

## 📁 Project Structure

```
Zero-shot-Text-classification/
├── .gitignore
├── README.md
├── biencoder-config.yaml
├── polyencoder-config.yaml
├── dataset.py
├── data/
├── models/
├── scripts/
│   ├── biencoder_train.py
│   ├── polyencoder_train.py
│   ├── evaluate.py
|   |── generate_data.py
├── requirements.txt
```

---

# 📊 Evaluation Results

Evaluation is performed using ranking-based metrics:

- **Precision@5**
- **Recall@5**
- **F1@5**
- **MRR (Mean Reciprocal Rank)**

---

| Model                                     | Precision@5 | Recall@5 | F1@5  | MRR   |
|-------------------------------------------|------------|----------|-------|-------|
| Bi-Encoder (DistilBERT)                   | 0.140      | 0.650    | 0.229 | 0.534 |
| Bi-Encoder (all-MiniLM-L6-v2)             | 0.142      | 0.658    | 0.232 | 0.553 |
| Bi-Encoder (all-MiniLM-L6-v2 + Hard Neg.) | 0.146      | 0.676    | 0.238 | 0.589 |
| Poly-Encoder                              | 0.144      | 0.663    | 0.234 | 0.536 |


Hard negative mining significantly improves:

- Precision
- Recall
- F1
- MRR

---
## 📈 Key Observations

- MiniLM outperforms DistilBERT.
- Hard negative mining significantly improves ranking metrics.
- Bi-Encoder + Hard Negatives achieves the best overall performance.
- MRR improvement shows better top-rank label prediction quality.
