# Intelligent staging and adaptive intervention of chronic infected wounds via a deep-learning-assisted wearable sensing-to-therapeutic patch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the deep learning code implementation for the wound staging classification framework associated with the manuscript submitted to *Nature Communications*. The system employs a **ResNet-18 ensemble** with **5-fold cross-validation** and **Grad-CAM interpretability** to achieve automated wound-stage classification with **>90% accuracy**.

**Code Author**: Qilun Yin (second author of the manuscript).

---

## Abstract

Chronic infected wounds are sustained by microbial colonization and biofilm formation, which prolong inflammation, impair vascularization and delay tissue regeneration. Their management is hindered by a diagnostic–therapeutic mismatch: wound-state assessment is often episodic, subjective or delayed, whereas antimicrobial treatment is non-specific and increasingly compromised by drug resistance. Here we report a wearable sensing-to-therapeutic patch, termed **wSTP**, for deep-learning-assisted precise wound staging and adaptive optical therapy in a single conformal platform. The wSTP combines a flexible oxygen saturation (sO₂)-monitoring module with photosensitizer-loaded biodegradable microneedles and wirelessly interfaces with a smartphone for data acquisition, image analysis and therapeutic control. Continuous sO₂ trajectories provide metabolic information on the hypoxia–infection cycle, while wound images capture morphological features of inflammation, infection, biofilm formation and regeneration. Fusion of these multimodal inputs using a **ResNet-18-based framework** enables automated wound-stage classification with an accuracy exceeding **90%**. Guided by the identified stage, integrated 660 nm μ-LEDs activate photodynamic therapy with microneedle-delivered photosensitizers to disrupt biofilms and eliminate bacteria, or provide photobiomodulation to regulate inflammation and promote repair.

---

## Key Features

- **5-Fold Cross-Validated Ensemble**: Five independently trained ResNet-18 models with soft-voting inference for robust and low-variance predictions.
- **Weighted Random Sampling**: Addresses class imbalance by ensuring balanced mini-batch exposure across wound categories.
- **Transfer Learning**: ImageNet pre-trained backbone with semi-freezing fine-tuning and differential learning rates.
- **Grad-CAM Interpretability**: Visual heatmaps highlighting the regions driving model predictions, enabling clinical trust.
- **t-SNE Visualization**: Feature-space analysis demonstrating clear inter-class separability.
- **Nature-style Publication Figures**: All outputs generate publication-ready vector graphics (PDF, 300 DPI).

---

## Repository Structure

```
wSTP-wound-classification/
├── train_kfold.py              # 5-fold cross-validation training pipeline
├── single_inference.py         # Single-image inference with Grad-CAM visualization
├── inference_ensemble.py       # Batch evaluation: t-SNE, Grad-CAM, confidence analysis
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended; CPU-only mode is supported)

### Setup

```bash
git clone https://github.com/Yinqilun/wSTP-wound-classification.git
cd wSTP-wound-classification
pip install -r requirements.txt
```

---

## Usage

### 1. Training (5-Fold Cross-Validation)

Train the ResNet-18 ensemble using stratified 5-fold cross-validation with weighted sampling:

```bash
python train_kfold.py
```

The script will:
- Perform stratified data splitting into 5 folds
- Train one ResNet-18 model per fold with semi-freezing and differential learning rates
- Save trained weights as `best_model_fold1.pth` through `best_model_fold5.pth`
- Evaluate each fold on its held-out validation set

**Configuration** (edit within `train_kfold.py`):
- `DATA_ROOT`: Path to your wound image dataset (organized by class subdirectories)
- `NUM_CLASSES`: Number of wound categories (default: 4)
- `NUM_EPOCHS`, `BATCH_SIZE`, `LR`: Training hyperparameters

### 2. Single-Image Inference (Clinical Use)

Run inference on a single wound image with Grad-CAM heatmap visualization:

```bash
python single_inference.py
```

**Before running**, update the configuration in `single_inference.py`:
```python
TARGET_IMG_PATH = "./path/to/your/wound_image.jpg"
MODEL_PATH = "best_model_fold5.pth"  # or any fold's weight
```

**Output**: A side-by-side figure showing the original image with prediction/confidence and the Grad-CAM activation map.

### 3. Batch Evaluation & Visualization

Perform comprehensive evaluation on a test dataset:

```bash
python inference_ensemble.py
```

**Configuration** (edit within `inference_ensemble.py`):
- `DATA_ROOT`: Path to test dataset
- `MODEL_PATH`: Path to a trained model checkpoint

**Outputs**:
- `advanced_viz_tsne_high_contrast.pdf` — t-SNE feature embedding visualization
- `advanced_viz_gradcam.pdf` — Grad-CAM heatmaps for randomly sampled images
- `advanced_viz_confidence.pdf` — Confidence distribution histogram (correct vs. incorrect predictions)

---

## Model Architecture

| Component | Specification |
|-----------|--------------|
| Backbone | ResNet-18 (ImageNet pre-trained) |
| Classifier Head | Dropout(0.5) → Linear(512, 4) |
| Input Size | 224 × 224 × 3 |
| Normalization | ImageNet mean/std |
| Ensemble Strategy | Soft voting (mean of 5-fold output probabilities) |

### Training Strategy

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Feature Extractor LR | 1 × 10⁻⁵ |
| Classifier Head LR | 1 × 10⁻⁴ |
| Loss Function | Cross-Entropy Loss |
| Data Augmentation | Random flip, rotation (±30°), slight brightness/contrast jitter |
| Sampling | Weighted random sampling (inverse class frequency) |

---

## Results

- **Overall Accuracy**: Exceeding 90% on independent test set (ensemble inference)
- **Class Balance**: Weighted sampling effectively mitigates long-tail distribution
- **Interpretability**: Grad-CAM activations consistently localize to wound regions rather than background

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions regarding the code, please open an issue on this repository.
