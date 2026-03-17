import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_loaders, get_class_weights, NUM_CLASSES, CLASS_NAMES, RESULTS_DIR
from train import fit
from custom_cnn import custom_CNN
from utils import plot_history, plot_confusion_matrix, visualize_predictions, print_classification_report, compute_macro_auc
import json

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "cnn"
OUTPUT_DIR  = os.path.join("results", MODEL_NAME, "final")

NUM_EPOCHS = 15
LR = 1e-3
USE_WEIGHTED = False      # toggle for class-imbalance ablation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Saving results to: {OUTPUT_DIR}")

# ── Data ──────────────────────────────────────────────────────────────────────
train_loader, val_loader, test_loader = get_loaders()

# ── Model ─────────────────────────────────────────────────────────────────────
model = custom_CNN(num_classes=NUM_CLASSES).to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

if USE_WEIGHTED:
    criterion = nn.CrossEntropyLoss(weight=get_class_weights(device))
else:
    criterion = nn.CrossEntropyLoss()

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"Training {MODEL_NAME}...")
history = fit(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device, output_dir=OUTPUT_DIR)

# ── Evaluate ──────────────────────────────────────────────────────────────────
plot_history({"CNN": history}, NUM_EPOCHS, output_dir=OUTPUT_DIR, model_name=MODEL_NAME)

plot_confusion_matrix(model, test_loader, device, CLASS_NAMES, title="CNN", output_dir=OUTPUT_DIR, model_name=MODEL_NAME)

print_classification_report(model, test_loader, device, CLASS_NAMES, output_dir=OUTPUT_DIR, model_name=MODEL_NAME)

visualize_predictions(model, test_loader, device, CLASS_NAMES, output_dir=OUTPUT_DIR, model_name=MODEL_NAME)

# ── Save weights ──────────────────────────────────────────────────────────────
weights_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.pth")
torch.save(model.state_dict(), weights_path)
print(f"Saved weights: {weights_path}")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
