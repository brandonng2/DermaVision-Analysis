# Dermascope-VLM

Skin disease classification on HAM10000 using CNNs, Transformers, and Vision-Language Models. Compares a custom CNN, fine-tuned ResNet-50, and fine-tuned Swin-T against zero-shot CLIP and DermLIP baselines. Includes sequential ablations over optimizer, learning rate, and class imbalance strategy for each supervised model.

## Project Overview

HAM10000 is a 10,015-image dermatoscopy dataset spanning 7 classes (melanoma, melanocytic nevi, basal cell carcinoma, actinic keratosis, benign keratosis, dermatofibroma, vascular lesions). The dataset is heavily imbalanced — melanocytic nevi account for ~67% of samples — making class imbalance handling a key design consideration.

This project trains and evaluates five models:

| Model         | Type        | Training       |
| ------------- | ----------- | -------------- |
| Custom CNN    | CNN         | From scratch   |
| ResNet-50     | CNN         | Fine-tuned     |
| Swin-T Tiny   | Transformer | Fine-tuned     |
| CLIP ViT-B/32 | VLM         | Zero-shot only |
| DermLIP       | Medical VLM | Zero-shot only |

Each supervised model goes through a sequential ablation (optimizer → learning rate → imbalance strategy) before a final best-config training run. Results are evaluated on a held-out test set using accuracy, macro AUC, micro AUC, and per-class AUC.

## Project Structure

```
Dermascope-VLM/
├── data/
│   ├── images/                  # HAM10000 images (.jpg)
│   ├── masks/                   # HAM10000 masks (.jpg)
│   └── GroundTruth.csv          # Class labels
│
├── notebooks/
│   ├── dataset.ipynb            # Data exploration and class distribution
│   ├── evaluate_best_image_models.ipynb  # Compare best CNN / ResNet-50 / Swin-T
│   ├── ablation_cnn.ipynb       # CNN ablation analysis
│   ├── ablation_resnet50.ipynb  # ResNet-50 ablation analysis
│   └── ablation_swin_t.ipynb    # Swin-T ablation analysis
│
├── results/
│   ├── cnn/
│   │   ├── best_model/          # Checkpoint + logs from best config
│   │   └── ablation/            # Per-axis ablation runs
│   ├── resnet50/
│   │   ├── best_model/
│   │   └── ablation/
│   └── swin_t/
│       ├── best_model/
│       └── ablation/
│
├── scripts/
│   └── download_data.sh         # Downloads and unzips HAM10000 from Kaggle
│
├── src/
│   ├── dataset.py               # HAM10000Dataset, data loaders, transforms, class weights
│   ├── train.py                 # Shared train/eval loop used by all supervised models
│   ├── utils.py                 # Metrics, plotting, Grad-CAM
│   │
│   ├── cnn/
│   │   ├── custom_cnn.py        # 4-block CNN architecture
│   │   ├── train_cnn.py         # Final training run (best config)
│   │   └── ablation_cnn.py      # Sequential ablation over optimizer / LR / imbalance
│   │
│   ├── resnet50/
│   │   ├── resnet50.py          # ResNet-50 with custom head
│   │   ├── train_resnet50.py    # Final training run
│   │   └── ablation_resnet50.py
│   │
│   └── swin_t/
│       ├── swin_t.py            # Swin-T Tiny with custom dropout head
│       ├── train_swin_t.py      # Final training run
│       └── ablation_swin_t.py
│
├── environment.yml
└── README.md
```

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Dermascope-VLM.git
cd Dermascope-VLM
```

### 2. Set up Kaggle API credentials

The dataset is hosted on Kaggle, so you need a Kaggle account and API key:

1. Go to **Kaggle Account → API** and create a new API token.
2. You can either:
   - Place `kaggle.json` in `~/.kaggle/kaggle.json` and run:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```
   - Or set environment variables:
     ```bash
     export KAGGLE_USERNAME=your_username
     export KAGGLE_KEY=your_api_key
     ```

### 3. Create Conda environment

```bash
conda env create -f environment.yml
conda activate derma
```

### 4. Download HAM10000 dataset

```bash
bash scripts/download_data.sh
```

After running this, `data/` should contain:

```
data/
├── images/
├── masks/
└── GroundTruth.csv
```

## Running

### Ablation (run first to find best hyperparameters)

```bash
python src/cnn/ablation_cnn.py
python src/resnet50/ablation_resnet50.py
python src/swin_t/ablation_swin_t.py
```

Each ablation script runs three sequential axes — optimizer, learning rate, imbalance strategy — and writes a `summary.json` with the best config to `results/<model>/ablation/`.

### Final training (update best config in train script first)

```bash
python src/cnn/train_cnn.py
python src/resnet50/train_resnet50.py
python src/swin_t/train_swin_t.py
```

### Evaluation

Open `notebooks/evaluate_best_image_models.ipynb` to compare all three models on the test set. Per-model ablation analysis is in `notebooks/ablation_<model>.ipynb`.

## Key Design Decisions

- `src/train.py` and `src/utils.py` are shared across all supervised models — one fix applies everywhere
- Val split is 10% of training data, stratified by class — val loss is tracked each epoch for early stopping and ablation selection
- Class weights computed from training set targets only, never from val or test
- Ablations are sequential: each axis fixes the best result from the previous axis before sweeping the next
