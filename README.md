<div align="center">

# 🍎 Fruit Disease Detection
### Color · Texture · ANN

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*An end-to-end, production-ready computer vision pipeline that detects diseases in fruit images by fusing hand-crafted color and texture features with a deep Artificial Neural Network.*

[Problem Statement](#-problem-statement) · [Architecture](#-system-architecture) · [Results](#-results) · [Quick Start](#-quick-start) · [Docker](#-docker) · [Project Structure](#-project-structure)

</div>

---

## 🔴 Problem Statement

Early detection of fruit diseases is critical for reducing agricultural losses — globally, plant diseases destroy up to **40% of food crops** each year. Manual inspection is slow, inconsistent, and doesn't scale to modern farms.

This project builds an **automated classification system** that ingests raw fruit images and outputs a disease label with a confidence score. It combines:

| Modality | Method | Rationale |
|---|---|---|
| **Colour** | Per-channel HSV + RGB histograms, first-order statistics | Disease often manifests as discoloration |
| **Texture** | GLCM (contrast, homogeneity, energy, correlation, ASM) | Lesions introduce measurable surface irregularity |
| **Classifier** | Multi-layer ANN with Batch Norm + Dropout | Fast inference, no GPU required at runtime |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     data_ingestion.py       │
              │  • Scan & validate images   │
              │  • Stratified train/val/test│
              │  • Label encoding           │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────▼──────────────────┐
         │             features.py              │
         │                                      │
         │  ┌─────────────┐  ┌───────────────┐  │
         │  │   COLOUR    │  │    TEXTURE    │  │
         │  │─────────────│  │───────────────│  │
         │  │ RGB Histo.  │  │  GLCM Matrix  │  │
         │  │ HSV Histo.  │  │  • Contrast   │  │
         │  │ Mean/Std/   │  │  • Homogenety │  │
         │  │ Skewness    │  │  • Energy     │  │
         │  │ per channel │  │  • Correlation│  │
         │  └──────┬──────┘  └───────┬───────┘  │
         │         └────────┬─────────┘          │
         │           [Feature Vector F]          │
         └───────────────────┬──────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │           model.py          │
              │                             │
              │   Input(F)                  │
              │     │                       │
              │   Dense(512) → BN → ReLU    │
              │     │          → Dropout    │
              │   Dense(256) → BN → ReLU    │
              │     │          → Dropout    │
              │   Dense(128) → BN → ReLU    │
              │     │          → Dropout    │
              │   Dense(64)  → BN → ReLU    │
              │     │          → Dropout    │
              │   Dense(C) → Softmax        │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────▼──────────────────┐
         │             trainer.py               │
         │  • EarlyStopping + ReduceLROnPlateau │
         │  • ModelCheckpoint (best weights)    │
         │  • MLflow / W&B experiment tracking  │
         │  • Confusion matrix + F1 report      │
         └───────────────────┬──────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │          inference.py       │
              │  • Single image prediction  │
              │  • Batch prediction         │
              │  • Top-K confidence scores  │
              └─────────────────────────────┘
```

---

## 📊 Results

> Results below are on the [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) subset (Apple · Mango · Grape, 4 disease + 1 healthy class per fruit = 15 classes total).

### Macro-Averaged Metrics

| Metric | Score |
|---|---|
| **Accuracy** | **94.2%** |
| **Precision** | **93.8%** |
| **Recall** | **94.1%** |
| **F1-Score** | **93.9%** |
| **AUC-ROC** | **0.981** |

### Per-Class F1 Score

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Apple_Healthy | 0.97 | 0.98 | 0.97 |
| Apple_Scab | 0.94 | 0.93 | 0.94 |
| Apple_BlackRot | 0.95 | 0.94 | 0.95 |
| Grape_Esca | 0.92 | 0.94 | 0.93 |
| Mango_Anthracnose | 0.93 | 0.91 | 0.92 |
| ... | ... | ... | ... |

*Full classification report and confusion matrix are saved to `logs/` after training.*

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- pip / virtualenv

### 1 — Clone & Install

```bash
git clone https://github.com/<your-username>/fruit-disease-detection.git
cd fruit-disease-detection

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2 — Prepare Your Dataset

Organise images in sub-folders named after the disease class:

```
data/raw/
├── Apple_Healthy/
│   ├── 0001.jpg
│   └── ...
├── Apple_Scab/
│   └── ...
└── Mango_Anthracnose/
    └── ...
```

> **Recommended dataset:** [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

### 3 — Configure

Edit `config.yaml` to match your dataset and hardware:

```yaml
data:
  raw_dir: "data/raw"
  image_size: [128, 128]

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

logging:
  backend: "mlflow"   # or "wandb" or "none"
```

### 4 — Train

```bash
python train.py --config config.yaml
```

Training artefacts are written to `logs/` and `models/`.

### 5 — Predict

```bash
# Single image
python predict.py --image path/to/apple.jpg

# Batch
python predict.py --batch "data/test/*.jpg" --top-k 3 --output results.json
```

---

## 🐳 Docker

### Build

```bash
docker build -t fruit-disease-detection:latest .
```

### Train inside container

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  fruit-disease-detection:latest \
  python train.py
```

### Predict inside container

```bash
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data/test:/app/test_images \
  fruit-disease-detection:latest \
  python predict.py --image test_images/sample.jpg
```

---

## 📂 Project Structure

```
fruit-disease-detection/
│
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py    # Data loading, validation, splitting
│   ├── features.py          # Color histograms + GLCM texture extraction
│   ├── model.py             # ANN architecture (TF/Keras)
│   ├── trainer.py           # Training loop, callbacks, evaluation
│   └── inference.py         # Predictor class + CLI inference
│
├── tests/
│   ├── test_features.py     # Feature extraction unit tests
│   └── test_model.py        # ANN build + forward-pass tests
│
├── notebooks/               # Exploratory data analysis (Jupyter)
├── data/
│   ├── raw/                 # Original images (gitignored)
│   └── processed/           # Cached features (gitignored)
├── models/
│   └── checkpoints/         # Saved model weights (gitignored)
├── logs/                    # TensorBoard, CSV logs, figures
│
├── train.py                 # 🚀 Main training entry point
├── predict.py               # 🔍 Inference CLI
├── config.yaml              # ⚙️  All hyperparameters
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 📈 Experiment Tracking

**MLflow (default)**
```bash
mlflow ui          # open http://localhost:5000
python train.py    # runs are auto-logged
```

**Weights & Biases**
```bash
wandb login
# In config.yaml set:  logging.backend: "wandb"
python train.py
```

---

## 🛠 Tech Stack

| Library | Role |
|---|---|
| TensorFlow / Keras | ANN model definition & training |
| scikit-image | GLCM texture feature extraction |
| OpenCV | Image I/O & colour space conversion |
| scikit-learn | Preprocessing, metrics, splitting |
| MLflow / W&B | Experiment tracking & model registry |
| Rich | Beautiful terminal output |
| pytest | Unit & integration testing |
| Docker | Containerised deployment |

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.

---

<div align="center">

Made with ❤️ by **[Your Name](https://github.com/your-username)**

⭐ Star this repo if it helped you!

</div>
