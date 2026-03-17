import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))  
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_loaders, get_class_weights, NUM_CLASSES, CLASS_NAMES
from train import fit
from custom_cnn import custom_CNN
from utils import plot_history, plot_confusion_matrix, visualize_predictions, print_classification_report, compute_macro_auc

# ── Config (best config from ablation) ───────────────────────────────────────
MODEL_NAME = "cnn"
OUTPUT_DIR = os.path.join("results", MODEL_NAME, "best_model")
NUM_EPOCHS = 40
OPTIMIZER  = "adam"
LR         = 1e-4
IMBALANCE  = "sampler"  # best from ablation: highest macro F1, recovers minority classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Saving to: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
train_loader, val_loader, test_loader = get_loaders(use_sampler=(IMBALANCE == "sampler"))

# ── Model ─────────────────────────────────────────────────────────────────────
model = custom_CNN(num_classes=NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"Training CNN — {OPTIMIZER} | lr={LR} | {IMBALANCE}")
history = fit(model, train_loader, val_loader, optimizer, criterion,
              NUM_EPOCHS, device, output_dir=OUTPUT_DIR)

# ── Evaluate ──────────────────────────────────────────────────────────────────
plot_history({"CNN": history}, NUM_EPOCHS, output_dir=OUTPUT_DIR, model_name=MODEL_NAME)
plot_confusion_matrix(model, test_loader, device, CLASS_NAMES, title="CNN", output_dir=OUTPUT_DIR, model_name=MODEL_NAME)
print_classification_report(model, test_loader, device, CLASS_NAMES, output_dir=OUTPUT_DIR, model_name=MODEL_NAME)
visualize_predictions(model, test_loader, device, CLASS_NAMES, output_dir=OUTPUT_DIR, model_name=MODEL_NAME)

auc = compute_macro_auc(model, test_loader, device, NUM_CLASSES)
print(f"Macro AUC: {auc:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_cnn.pth"))
print(f"Saved: {OUTPUT_DIR}/cnn.pth")