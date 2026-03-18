import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_loaders, get_class_weights, NUM_CLASSES, CLASS_NAMES
from train import fit
from swin_t import build_swin_tiny
from utils import plot_history, plot_confusion_matrix, visualize_predictions, print_classification_report, compute_macro_auc

# ── Config (best config from ablation) ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "configs" / "swin_t.json", "r") as f:
    CFG = json.load(f)

MODEL_NAME = CFG["model_name"]
OUTPUT_DIR = str(PROJECT_ROOT / CFG["paths"]["best_output_dir"])
BEST_MODEL_PATH = str(PROJECT_ROOT / CFG["paths"]["best_model_path"])
NUM_EPOCHS = int(CFG["training"]["num_epochs"])
OPTIMIZER = CFG["training"]["optimizer"]
LR = float(CFG["training"]["lr"])
IMBALANCE = CFG["training"]["imbalance"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Saving to: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
train_loader, val_loader, test_loader = get_loaders(use_sampler=(IMBALANCE == "sampler"))

# ── Model ─────────────────────────────────────────────────────────────────────
model = build_swin_tiny(num_classes=NUM_CLASSES).to(device)

if OPTIMIZER == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

if IMBALANCE == "weighted":
    criterion = nn.CrossEntropyLoss(weight=get_class_weights(device))
else:
    criterion = nn.CrossEntropyLoss()

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"Training Swin-T — {OPTIMIZER} | lr={LR} | {IMBALANCE}")
history = fit(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device, output_dir=OUTPUT_DIR)

# ── Evaluate ──────────────────────────────────────────────────────────────────
plot_history({"Swin-T": history}, NUM_EPOCHS, output_dir=OUTPUT_DIR, model_name=MODEL_NAME)
plot_confusion_matrix(model, test_loader, device, CLASS_NAMES, title="Swin-T", output_dir=OUTPUT_DIR, model_name=MODEL_NAME)
print_classification_report(model, test_loader, device, CLASS_NAMES, output_dir=OUTPUT_DIR, model_name=MODEL_NAME)
visualize_predictions(model, test_loader, device, CLASS_NAMES, output_dir=OUTPUT_DIR, model_name=MODEL_NAME)

auc = compute_macro_auc(model, test_loader, device, NUM_CLASSES)

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), BEST_MODEL_PATH)
print(f"Saved: {BEST_MODEL_PATH}")