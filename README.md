# Dermascope-VLM
 
## Installation & Setup
 
Follow these steps to set up the environment and get the HAM10000 dataset ready for training.
 
### 1. Clone the repository
 
```bash
git clone https://github.com/yourusername/Dermascope-VLM.git
cd Dermascope-VLM
```
 
### 2. Set up Kaggle API credentials
 
The dataset is hosted on Kaggle, so you need a Kaggle account and API key:
 
1. Go to **Kaggle Account → API** and create a new API token.
2. You can either:
   - Place `kaggle.json` in `~/.kaggle/kaggle.json` (Linux/Mac) and run:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```
   - Or set environment variables in your shell:
     ```bash
     export KAGGLE_USERNAME=your_username
     export KAGGLE_KEY=your_api_key
     ```
 
This ensures the Kaggle CLI can authenticate and download the dataset.
 
### 3. Create Conda environment
 
This project uses Conda to manage dependencies. The required packages are listed in `environment.yml`.
 
```bash
# Create environment from YAML file
conda env create -f environment.yml
 
# Activate the environment
conda activate derma
```
 
### 4. Download HAM10000 dataset
 
We provide a script to automatically download and unzip the dataset:
 
```bash
bash scripts/download_data.sh
```
 
- This will download the dataset and unzip it into the `data/` folder.
- You do not need to manually run `chmod +x` if using `bash scripts/download_data.sh`.
 
After running this, you should see:
 
```
data/
├── images/
├── masks/        # if segmentation task
└── metadata.csv
```

## Project Structure
 
## Running
 
  
## Models
 
| Model         | Type        | Training          |
|---------------|-------------|-------------------|
| Custom CNN    | CNN         | From scratch      |
| ResNet-50     | CNN         | Fine-tuned        |
| Swin-T Tiny   | Transformer | Fine-tuned        |
| CLIP ViT-B/32 | VLM         | Zero-shot only    |
| DermLIP       | Medical VLM | Zero-shot only    |
 
## Ablations
 
- **LR sweep**: `ABLATION_MODE = True` in `train_resnet.py` — sweeps [1e-3, 1e-4, 1e-5, 1e-6]
- **Class imbalance**: Toggle `USE_WEIGHTED` in any `train_*.py`
- **Prompt sensitivity**: Automatically run in `eval_clip.py`
 
## Key Design Decisions
 
- `train.py` is shared across all supervised models — one fix applies everywhere
- `val_loader` is split from train set (10%) so val loss is tracked properly each epoch
- CLIP uses its own normalization (`clip_transform` in `dataset.py`) — separate from supervised models
- Class weights computed from **training** set targets (not test set)