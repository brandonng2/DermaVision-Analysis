import os
import sys
import json
import torch
import numpy as np
import open_clip
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dataset import get_splits, CLASS_NAMES, NUM_CLASSES

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "eval_vlm.json"

def _load_vlm_config():
    """Load VLM evaluation config from JSON."""
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {CONFIG_PATH}. Using defaults.")
        return {
            "paths": {"image_dir": "data/images", "output_dir": "results/vlm"},
            "loader": {"batch_size": 128, "num_workers": 4},
            "models": {
                "CLIP": {"model_id": "ViT-B-32", "pretrained": "openai"},
                "DermLIP": {"model_id": "hf-hub:redlessone/DermLIP_ViT-B-16", "pretrained": None},
            },
            "prompt_templates": {
                "dermoscopy": "a dermoscopy image of {class}",
                "dermatoscopic": "a dermatoscopic image of {class}",
                "skin_lesion": "a skin lesion showing {class}",
                "this_is": "this is a skin image of {class}",
                "photo_of": "a photo of {class}",
                "clinical": "a clinical image of {class}, a skin disease",
            },
            "class_names": ["melanoma", "melanocytic nevi", "basal cell carcinoma",
                          "actinic keratosis", "benign keratosis", "dermatofibroma", "vascular lesions"],
        }

_cfg = _load_vlm_config()

IMAGE_DIR = PROJECT_ROOT / _cfg["paths"]["image_dir"]
OUTPUT_DIR = PROJECT_ROOT / _cfg["paths"]["output_dir"]
BATCH_SIZE = _cfg["loader"]["batch_size"]
NUM_WORKERS = _cfg["loader"]["num_workers"]

# Build MODELS dict from config
MODELS = {}
for model_name, model_cfg in _cfg["models"].items():
    MODELS[model_name] = (model_cfg["model_id"], model_cfg["pretrained"])

# Build PROMPT_TEMPLATES dict with lambda functions
PROMPT_TEMPLATES = {}
for tmpl_name, tmpl_str in _cfg["prompt_templates"].items():
    # Convert template string like "a dermoscopy image of {class}" to lambda
    PROMPT_TEMPLATES[tmpl_name] = lambda c, t=tmpl_str: t.format(c=c)

HAM_CLASSNAMES = _cfg["class_names"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Dataset ───────────────────────────────────────────────────────────────────
class VLMTestDataset(Dataset):
    def __init__(self, df, preprocess):
        self.df        = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.labels    = df[CLASS_NAMES].values.argmax(axis=1).astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["image"]
        image  = self.preprocess(
            Image.open(IMAGE_DIR / f"{img_id}.jpg").convert("RGB")
        )
        return image, self.labels[idx]


# ── Text embeddings ───────────────────────────────────────────────────────────
def encode_text_embeddings(model, tokenizer, template):
    texts = tokenizer([template(c) for c in HAM_CLASSNAMES]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features  # [num_classes, embed_dim]


# ── Zero-shot eval for one template ──────────────────────────────────────────
def zero_shot_eval(model, tokenizer, loader, template):
    model.eval()
    text_features = encode_text_embeddings(model, tokenizer, template)

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            img_features = model.encode_image(images)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            logits = img_features @ text_features.T  # [B, num_classes]
            probs  = logits.softmax(dim=-1).cpu().numpy()
            preds  = probs.argmax(axis=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    probs  = np.array(all_probs)
    label_matrix = np.eye(NUM_CLASSES)[labels]

    return {
        "accuracy":      (preds == labels).mean() * 100,
        "macro_auc":     roc_auc_score(label_matrix, probs, average="macro", multi_class="ovr"),
        "micro_auc":     roc_auc_score(label_matrix, probs, average="micro", multi_class="ovr"),
        "per_class_auc": roc_auc_score(label_matrix, probs, average=None,   multi_class="ovr"),
        "report":        classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0),
        "labels":        labels,
        "preds":         preds,
        "probs":         probs,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _, _, test_df = get_splits()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, (model_id, pretrained) in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")

        if pretrained:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_id, pretrained=pretrained
            )
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(model_id)
        tokenizer = open_clip.get_tokenizer(model_id)

        model = model.to(device)

        # Build loader once — reused across all templates
        loader = DataLoader(
            VLMTestDataset(test_df, preprocess),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0),
        )

        # ── Prompt sweep ──────────────────────────────────────────────────────
        sweep_results = {}
        print(f"\n  {'Template':<20}  {'Accuracy':>10}  {'Macro AUC':>10}  {'Micro AUC':>10}")
        print(f"  {'-'*56}")

        for tmpl_name, template in PROMPT_TEMPLATES.items():
            result = zero_shot_eval(model, tokenizer, loader, template)
            sweep_results[tmpl_name] = result
            print(f"  {tmpl_name:<20}  {result['accuracy']:>9.2f}%"
                  f"  {result['macro_auc']:>10.4f}  {result['micro_auc']:>10.4f}")

        # ── Save all results ──────────────────────────────────────────────────
        best_name   = max(sweep_results, key=lambda k: sweep_results[k]["accuracy"])
        best_result = sweep_results[best_name]

        model_dir = OUTPUT_DIR / model_name.lower()
        model_dir.mkdir(parents=True, exist_ok=True)

        # Full sweep summary
        summary_path = model_dir / "prompt_sweep.txt"
        with open(summary_path, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"{'Template':<20}  {'Accuracy':>10}  {'Macro AUC':>10}  {'Micro AUC':>10}\n")
            f.write("-" * 56 + "\n")
            for tmpl_name, r in sweep_results.items():
                marker = " <-- best" if tmpl_name == best_name else ""
                f.write(f"{tmpl_name:<20}  {r['accuracy']:>9.2f}%"
                        f"  {r['macro_auc']:>10.4f}  {r['micro_auc']:>10.4f}{marker}\n")

        # Best config classification report
        report_path = model_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Best template: {best_name} — "
                    f"\"{PROMPT_TEMPLATES[best_name](HAM_CLASSNAMES[0])}\"\n\n")
            f.write(best_result["report"])

        # Per-class AUC for best template
        print(f"\n  Best template: '{best_name}' ({best_result['accuracy']:.2f}%)")
        print(f"\n  Per-class AUC ({best_name}):")
        for i, cls in enumerate(CLASS_NAMES):
            print(f"    {cls:<25}  {best_result['per_class_auc'][i]:.4f}")

        print(f"\n  Saved: {model_dir}")