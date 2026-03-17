import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))  
import torch
import torch.nn as nn
import torch.optim as optim
import json

from dataset import get_loaders, get_class_weights, NUM_CLASSES, CLASS_NAMES
from train import fit
from custom_cnn import custom_CNN
from utils import plot_history, plot_confusion_matrix, visualize_predictions, print_classification_report, compute_macro_auc

# ── Config ────────────────────────────────────────────────────────────────────
# Sequential ablation — each axis uses best result from previous axis
# Axis 1: Optimizer  (fixed lr=1e-3, no weighting)
# Axis 2: LR sweep   (best optimizer, no weighting)
# Axis 3: Imbalance  (best optimizer + best LR)

NUM_EPOCHS = 15
MODEL_NAME = "cnn"
BASE_OUTPUT = os.path.join("results", MODEL_NAME, "ablation")

OPTIMIZERS = ["sgd", "adam"]
LR_VALUES  = [1e-3, 1e-4, 1e-5, 1e-6]
IMBALANCES = ["none", "weighted", "sampler"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_optimizer(model, opt_name, lr):
    if opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)


def build_criterion(imbalance, device):
    if imbalance == "weighted":
        return nn.CrossEntropyLoss(weight=get_class_weights(device))
    return nn.CrossEntropyLoss()


def run_single(optimizer_name, lr, imbalance, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_loader, val_loader, test_loader = get_loaders(
        use_sampler=(imbalance == "sampler")
    )

    model = custom_CNN(num_classes=NUM_CLASSES).to(device)
    optimizer = build_optimizer(model, optimizer_name, lr)
    criterion = build_criterion(imbalance, device)
    tag = f"{optimizer_name}_lr{lr}_{imbalance}"
    run_label = f"CNN | {optimizer_name} | lr={lr} | {imbalance}"
    print(f"\n── {run_label} ──")

    history = fit(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device, output_dir=output_dir)

    plot_history({run_label: history}, NUM_EPOCHS, output_dir=output_dir, model_name=tag)
    plot_confusion_matrix(model, test_loader, device, CLASS_NAMES, title=run_label, output_dir=output_dir, model_name=tag)
    print_classification_report(model, test_loader, device, CLASS_NAMES, output_dir=output_dir, model_name=tag)
    visualize_predictions(model, test_loader, device, CLASS_NAMES, output_dir=output_dir, model_name=tag)

    auc = compute_macro_auc(model, test_loader, device, NUM_CLASSES)
    best_val_acc = max(history["val_acc"])

    torch.save(model.state_dict(), os.path.join(output_dir, f"{tag}.pth"))
    print(f"Saved → {output_dir}/{tag}.pth")

    return {
        "optimizer": optimizer_name,
        "lr": lr,
        "imbalance": imbalance,
        "best_val_acc": round(best_val_acc, 4),
        "macro_auc": round(auc, 4),
    }


# ── Axis 1: Optimizer ─────────────────────────────────────────────────────────
print("=" * 60)
print("Axis 1: Optimizer (lr=1e-3, no weighting)")
print("=" * 60)
opt_results = []
for opt in OPTIMIZERS:
    result = run_single(
        optimizer_name=opt, lr=1e-3, imbalance="none",
        output_dir=os.path.join(BASE_OUTPUT, "optimizer", opt)
    )
    opt_results.append(result)

best_opt = max(opt_results, key=lambda x: x["best_val_acc"])["optimizer"]
print(f"\nBest optimizer: {best_opt}")

# ── Axis 2: LR sweep ─────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"Axis 2: LR sweep (optimizer={best_opt}, no weighting)")
print("=" * 60)
lr_results = []
for lr in LR_VALUES:
    result = run_single(
        optimizer_name=best_opt, lr=lr, imbalance="none",
        output_dir=os.path.join(BASE_OUTPUT, "lr", str(lr))
    )
    lr_results.append(result)

best_lr = max(lr_results, key=lambda x: x["best_val_acc"])["lr"]
print(f"\nBest LR: {best_lr}")

# ── Axis 3: Imbalance strategy ────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"Axis 3: Imbalance strategy (optimizer={best_opt}, lr={best_lr})")
print("=" * 60)
imb_results = []
for imb in IMBALANCES:
    result = run_single(
        optimizer_name=best_opt, lr=best_lr, imbalance=imb,
        output_dir=os.path.join(BASE_OUTPUT, "imbalance", imb)
    )
    imb_results.append(result)

best_imb = max(imb_results, key=lambda x: x["best_val_acc"])["imbalance"]
print(f"\nBest imbalance strategy: {best_imb}")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = {
    "axis1_optimizer": opt_results,
    "axis2_lr": lr_results,
    "axis3_imbalance": imb_results,
    "best_config": {
        "optimizer": best_opt,
        "lr": best_lr,
        "imbalance": best_imb,
    }
}
summary_path = os.path.join(BASE_OUTPUT, "summary.json")
os.makedirs(BASE_OUTPUT, exist_ok=True)
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved: {summary_path}")
print(f"\nBest config → optimizer={best_opt} | lr={best_lr} | imbalance={best_imb}")
print("Update train_cnn.py with these values and run it for the final model.")