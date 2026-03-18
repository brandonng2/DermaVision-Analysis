import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
NUM_CLASSES = len(CLASS_NAMES)
IDX2LABEL   = {i: cls for i, cls in enumerate(CLASS_NAMES)}
IDX2FULLNAME = {
    0: 'Melanoma',
    1: 'Melanocytic Nevi',
    2: 'Basal Cell Carcinoma',
    3: 'Actinic Keratosis',
    4: 'Benign Keratosis',
    5: 'Dermatofibroma',
    6: 'Vascular Lesions',
}

IMAGE_SIZE = 224
BATCH_SIZE = 32

# TODO: replace with output of src/compute_stats.py
MEAN = [0.763, 0.546, 0.570]
STD  = [0.141, 0.152, 0.169]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset.json"


def _load_dataset_config():
    with open(DATASET_CONFIG_PATH, "r") as f:
        return json.load(f)


_cfg = _load_dataset_config()
IMAGE_SIZE = int(_cfg.get("image", {}).get("size", IMAGE_SIZE))
BATCH_SIZE = int(_cfg.get("loader", {}).get("batch_size", BATCH_SIZE))
MEAN = _cfg.get("image", {}).get("mean", MEAN)
STD = _cfg.get("image", {}).get("std", STD)
IMAGE_DIR = PROJECT_ROOT / _cfg["paths"]["image_dir"]
CSV_PATH = PROJECT_ROOT / _cfg["paths"]["csv_path"]
RESULTS_DIR = PROJECT_ROOT / _cfg["paths"]["results_dir"]
DEFAULT_VAL_SIZE = float(_cfg.get("split", {}).get("val_size", 0.1))
DEFAULT_TEST_SIZE = float(_cfg.get("split", {}).get("test_size", 0.1))
DEFAULT_RANDOM_STATE = int(_cfg.get("split", {}).get("random_state", 42))

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.labels = df[CLASS_NAMES].values.argmax(axis=1).astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image']
        image  = Image.open(IMAGE_DIR / f"{img_id}.jpg").convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(self.labels[idx])


def get_splits(val_size=DEFAULT_VAL_SIZE, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE):
    df = pd.read_csv(CSV_PATH)
    labels = df[CLASS_NAMES].values.argmax(axis=1)

    train_df, temp_df, _, temp_labels = train_test_split(
        df, labels, test_size=(val_size + test_size),
        stratify=labels, random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test_size / (val_size + test_size),
        stratify=temp_labels, random_state=random_state,
    )
    return train_df, val_df, test_df


def get_loaders(batch_size=BATCH_SIZE, num_workers=None, use_sampler=False):
    if num_workers is None:
        num_workers = int(_cfg.get("loader", {}).get("num_workers", 4))

    train_df, val_df, test_df = get_splits()

    train_ds = HAM10000Dataset(train_df, train_transform)

    if use_sampler:
        class_counts = np.bincount(train_ds.labels, minlength=NUM_CLASSES)
        sample_weights = 1.0 / class_counts[train_ds.labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    loader_kwargs = dict(
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(HAM10000Dataset(val_df,  val_transform), **loader_kwargs)
    test_loader = DataLoader(HAM10000Dataset(test_df, val_transform), **loader_kwargs)

    return train_loader, val_loader, test_loader


def get_class_weights(device):
    train_df, _, _ = get_splits()
    targets = train_df[CLASS_NAMES].values.argmax(axis=1)
    weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=targets)
    return torch.tensor(weights, dtype=torch.float).to(device)


def show_distribution(df, name):
    label_counts = df[CLASS_NAMES].sum().astype(int)
    labels = [IDX2FULLNAME[i] for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(12, 5))
    color = 'purple' if name == "Testing" else 'steelblue'
    bars = ax.bar(labels, label_counts, color=color)

    for bar, count in zip(bars, label_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, str(count), ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(f"Class Distribution in {name} Set")
    ax.set_ylabel("Number of Images")
    ax.set_xlabel("Class")
    ax.set_ylim(0, max(label_counts) * 1.1)
    plt.tight_layout()
    plt.show()


def imshow(imgs, labels):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    imgs = (imgs * std + mean).clamp(0, 1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.transpose(imgs[i].numpy(), (1, 2, 0)))
        ax.set_xlabel(IDX2FULLNAME[labels[i].item()], fontsize=12, labelpad=6)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()
