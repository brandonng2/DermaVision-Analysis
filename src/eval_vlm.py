import os
import sys
import json
import torch
import numpy as np
import open_clip
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dataset import get_splits, CLASS_NAMES, NUM_CLASSES

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH  = PROJECT_ROOT / "configs" / "eval_vlm.json"


def _load_vlm_config():
    with open(CONFIG_PATH) as f:
            return json.load(f)

_cfg = _load_vlm_config()
IMAGE_DIR = PROJECT_ROOT / _cfg["paths"]["image_dir"]
OUTPUT_DIR = PROJECT_ROOT / _cfg["paths"]["output_dir"]
BATCH_SIZE = _cfg["loader"]["batch_size"]
NUM_WORKERS = _cfg["loader"]["num_workers"]

MODELS = {
    name: (cfg["model_id"], cfg["pretrained"])
    for name, cfg in _cfg["models"].items()
}

PROMPT_TEMPLATES = {
    name: (lambda c, t=tmpl: t.format(c=c))
    for name, tmpl in _cfg["prompt_templates"].items()
}

HAM_CLASSNAMES = _cfg["class_names"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class VLMTestDataset(Dataset):
    def __init__(self, df, preprocess):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.labels = df[CLASS_NAMES].values.argmax(axis=1).astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["image"]
        image = self.preprocess(
            Image.open(IMAGE_DIR / f"{img_id}.jpg").convert("RGB")
        )
        return image, self.labels[idx]


# ── Helpers ───────────────────────────────────────────────────────────────────
def encode_text_embeddings(model, tokenizer, template):
    texts = tokenizer([template(c) for c in HAM_CLASSNAMES]).to(device)
    with torch.no_grad():
        feats = model.encode_text(texts)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats


def zero_shot_eval(model, tokenizer, loader, template):
    model.eval()
    text_features = encode_text_embeddings(model, tokenizer, template)
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            img_feats = model.encode_image(images)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            probs = (img_feats @ text_features.T).softmax(dim=-1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(probs.argmax(axis=1))
            all_probs.extend(probs)

    labels = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    label_matrix = np.eye(NUM_CLASSES)[labels]

    return {
        "accuracy": (preds == labels).mean() * 100,
        "macro_auc": roc_auc_score(label_matrix, probs, average="macro", multi_class="ovr"),
        "micro_auc": roc_auc_score(label_matrix, probs, average="micro", multi_class="ovr"),
        "per_class_auc": roc_auc_score(label_matrix, probs, average=None,    multi_class="ovr").tolist(),
        "report": classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0),
        "labels": labels,
        "preds": preds,
    }


def save_confusion_matrix(labels, preds, title, output_dir):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="rocket_r",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix — {title}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fpath = output_dir / "confusion_matrix.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()


def save_prompt_results(result, tmpl_name, template_str, model_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Classification report
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Template: {tmpl_name} — \"{template_str}\"\n\n")
        f.write(result["report"])

    # Confusion matrix
    save_confusion_matrix(
        result["labels"], result["preds"],
        title=f"{model_name} / {tmpl_name}",
        output_dir=output_dir,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _, _, test_df = get_splits()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, (model_id, pretrained) in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")

        if pretrained:
            model, _, preprocess = open_clip.create_model_and_transforms(model_id, pretrained=pretrained)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(model_id)
        tokenizer = open_clip.get_tokenizer(model_id)
        model = model.to(device)

        loader = DataLoader(
            VLMTestDataset(test_df, preprocess),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0),
        )

        model_dir = OUTPUT_DIR / model_name.lower()
        sweep_results = {}

        print(f"\n{'Template':<20}  {'Accuracy':>10}  {'Macro AUC':>10}  {'Micro AUC':>10}")
        print(f"{'-'*56}")

        for tmpl_name, template in PROMPT_TEMPLATES.items():
            result = zero_shot_eval(model, tokenizer, loader, template)
            sweep_results[tmpl_name] = result

            print(f" {tmpl_name:<20}  {result['accuracy']:>9.2f}%"
                  f" {result['macro_auc']:>10.4f}  {result['micro_auc']:>10.4f}")

            # Save per-prompt folder: results/vlm/clip/dermoscopy/
            save_prompt_results(
                result, tmpl_name,
                template_str=_cfg["prompt_templates"][tmpl_name],
                model_name=model_name,
                output_dir=model_dir / tmpl_name,
            )

        # ── summary.json ─────────────────────────────────────────────────────
        best_name = max(sweep_results, key=lambda k: sweep_results[k]["accuracy"])

        summary = {
            "prompt_sweep": {
                name: {
                    "template": _cfg["prompt_templates"][name],
                    "accuracy": round(r["accuracy"], 4),
                    "macro_auc": round(r["macro_auc"], 4),
                    "micro_auc": round(r["micro_auc"], 4),
                    "per_class_auc": {
                        cls: round(r["per_class_auc"][i], 4)
                        for i, cls in enumerate(CLASS_NAMES)
                    },
                }
                for name, r in sweep_results.items()
            },
            "best_config": {
                "template": best_name,
                "template_str": _cfg["prompt_templates"][best_name],
                "accuracy": round(sweep_results[best_name]["accuracy"], 4),
                "macro_auc": round(sweep_results[best_name]["macro_auc"], 4),
                "micro_auc": round(sweep_results[best_name]["micro_auc"], 4),
                "per_class_auc": {
                    cls: round(sweep_results[best_name]["per_class_auc"][i], 4)
                    for i, cls in enumerate(CLASS_NAMES)
                },
            },
        }

        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nBest template: '{best_name}' ({sweep_results[best_name]['accuracy']:.2f}%)")
        print(f"Saved: {model_dir}")