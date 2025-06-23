# 🧠 LEMON EEG CNN Reimplementation

## Overview

This project reimplements the core methodology of the 2022 MDPI paper: *"EEG Signals Based Internet Addiction Diagnosis Using CNN with 5-Layer Architecture"*. It uses the LEMON (Leipzig Mind‑Brain‑Body) EEG dataset to classify internet addiction from resting-state brain activity.

The goal is to reproduce and improve upon the reported \~87.6% classification accuracy by modernizing the pipeline using PyTorch, MNE, and MLOps tools such as Weights & Biases.

---

## 🔗 Dataset

- **Source**: LEMON (Leipzig Mind‑Brain‑Body, MPI CBS)
- **Data Type**: 62-channel resting-state EEG
- **Participants**: Subset of 49 subjects (24 high IAT scores, 25 low IAT scores)
- **Sampling Rate**: 1000 Hz (to be downsampled to 250 Hz)
- **Format**: BIDS-compliant EEG (EDF or FIFF format)

---

## 📊 Objective

Classify subjects as either:

- **Internet Addicted (IA)**
- **Non-Addicted (Control)**

Using:

- **Input**: Resting-state EEG signals (spectral or raw time series)
- **Output**: Binary classification via CNN

---

## 🧪 Architecture (Baseline CNN)

- 5 Convolutional Layers
- ReLU Activations
- Max Pooling after each conv layer
- Flatten → Dense → Softmax
- Input shape: `(batch_size, 62, freq_bins)` for spectral mode

---

## 🔧 Tools & Libraries

- **Preprocessing**: `mne`, `numpy`, `scipy`
- **Modeling**: `PyTorch`, `torchvision`
- **MLOps**: `Weights & Biases`
- **Packaging**: `poetry` or `pip`, `Docker` (optional)

---

## 🛠️ Directory Structure

```
lemon-cnn-eeg/
├── data/              # Raw + preprocessed EEG
├── notebooks/         # EDA & sanity checks
├── scripts/
│   ├── preprocess.py  # MNE-based EEG preprocessing
│   ├── train.py       # CNN training loop
│   └── evaluate.py    # Evaluation metrics
├── models/            # CNN architecture definition
├── configs/           # YAML/JSON configs for experiments
├── wandb/             # W&B logs and run artifacts
├── requirements.txt   # Dependencies
└── README.md
```

---

## ✅ Goals

- Download and prepare LEMON EEG data
- Implement preprocessing: filtering, downsampling, FFT
- Window and segment EEG into 8s samples with 2s overlap
- Extract spectral power features (full and band-limited)
- Implement 5-layer CNN in PyTorch
- Train model on spectral input to reproduce ~87.6% accuracy
- Track experiments with Weights & Biases
- Document and refactor pipeline for reproducibility
- Prepare baseline for further multimodal extensions

---

## 🔁 Future Extensions

- Use FSDP or DeepSpeed for distributed training
- Extend with behavioral data (e.g., app usage)
- Test transformer-based architectures
- Convert pipeline into a dockerized CLI

---

## 📬 Contact

If you're interested in collaborating, contributing, or want support replicating the model, feel free to reach out!

