import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from dataset import MEAN, STD


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _unnormalize(images):
    """Undo dataset normalization for display. Returns clamped [0,1] tensor."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    return (images.cpu() * std + mean).clamp(0, 1)

# ── Predictions ───────────────────────────────────────────────────────────────
def get_predictions(model, loader, device):
    """Return (all_labels, all_preds, all_probs) as numpy arrays."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs  = inputs.to(device)
            outputs = model(inputs)
            probs   = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ── Classification report ─────────────────────────────────────────────────────
def print_classification_report(model, loader, device, classes, output_dir=None, model_name="model"):
    labels, preds, _ = get_predictions(model, loader, device)
    report = classification_report(labels, preds, target_names=classes, zero_division=0)
    print(report)
    if output_dir:
        make_output_dir(output_dir)
        fpath = os.path.join(output_dir, f"{model_name}_classification_report.txt")
        with open(fpath, "w") as f:
            f.write(report)
        print(f"Saved classification report: {fpath}")


# ── Macro AUC ─────────────────────────────────────────────────────────────────
def compute_macro_auc(model, loader, device, num_classes):
    labels, _, probs = get_predictions(model, loader, device)
    label_matrix = np.eye(num_classes)[labels]
    return roc_auc_score(label_matrix, probs, average='macro', multi_class='ovr')


# ── Confusion matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(model, loader, device, classes, title="", output_dir=None, model_name="model"):
    labels, preds, _ = get_predictions(model, loader, device)
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='rocket_r',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix — {title or model.__class__.__name__}")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_dir:
        make_output_dir(output_dir)
        fpath = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix: {fpath}")

    plt.show()
    plt.close()


# ── Training curves ───────────────────────────────────────────────────────────
def plot_history(histories: dict, num_epochs: int, output_dir=None, model_name="model"):
    epochs = range(1, num_epochs + 1)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    for name, h in histories.items():
        axs[0].plot(epochs, h["train_loss"], label=f"{name} train")
        axs[0].plot(epochs, h["val_loss"], label=f"{name} val", linestyle="--")
        axs[1].plot(epochs, h["train_acc"], label=f"{name} train")
        axs[1].plot(epochs, h["val_acc"], label=f"{name} val", linestyle="--")

    for ax, title, ylabel in zip(axs, 
                                 ["Training & Validation Loss", "Training & Validation Accuracy"],
                                 ["Loss", "Accuracy (%)"],
                                ):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    if output_dir:
        make_output_dir(output_dir)
        fpath = os.path.join(output_dir, f"{model_name}_training_curves.png")
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved training curves: {fpath}")

    plt.show()
    plt.close()


# ── Prediction visualization ──────────────────────────────────────────────────
def visualize_predictions(model, loader, device, classes, num_images=8, output_dir=None, model_name="model"):
    model.eval()
    images, labels = next(iter(loader))
    images_dev = images[:num_images].to(device)
    labels = labels[:num_images]

    with torch.no_grad():
        _, preds = torch.max(model(images_dev), 1)

    imgs = _unnormalize(images[:num_images])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.transpose(imgs[i].numpy(), (1, 2, 0)))
        color = 'green' if preds[i] == labels[i] else 'red'
        ax.set_xlabel(f"T: {classes[labels[i]]}\nP: {classes[preds[i]]}",
                      fontsize=8, color=color, labelpad=6)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(hspace=0.5, wspace=0.4)

    if output_dir:
        make_output_dir(output_dir)
        fpath = os.path.join(output_dir, f"{model_name}_sample_predictions.png")
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved sample predictions: {fpath}")

    plt.show()
    plt.close()


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]),
                                mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def show_gradcam(model, target_layer, loader, device, classes, num_images=4, title="", output_dir=None, model_name="model"):
    cam_extractor = GradCAM(model, target_layer)
    images, labels = next(iter(loader))

    fig, axes = plt.subplots(2, num_images, figsize=(4 * num_images, 8))
    for i in range(num_images):
        img_tensor = images[i].unsqueeze(0).to(device).requires_grad_(True)
        heatmap = cam_extractor(img_tensor)
        img_np = np.transpose(_unnormalize(images[i]).numpy(), (1, 2, 0))

        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"True: {classes[labels[i]]}", fontsize=9)
        axes[0, i].axis('off')

        axes[1, i].imshow(img_np)
        axes[1, i].imshow(heatmap, cmap='jet', alpha=0.4)
        axes[1, i].set_title("Grad-CAM", fontsize=9)
        axes[1, i].axis('off')

    plt.suptitle(title or model.__class__.__name__, fontsize=12)
    plt.tight_layout()

    if output_dir:
        make_output_dir(output_dir)
        fpath = os.path.join(output_dir, f"{model_name}_gradcam.png")
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved Grad-CAM: {fpath}")

    plt.show()
    plt.close()