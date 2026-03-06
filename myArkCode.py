"""
Ark chest X-ray classification benchmark.

Trains 3 architectures × 3 datasets × 5 runs = 45 total experiments.
- Architectures: swin_base, convnext_base, internimage (convnext_xlarge proxy)
- Datasets: ChestXray14, CheXpert, MIMIC
- FP16 AMP training, Adam @ 1e-3, batch size 64
- Per-epoch train AUC + test AUC
- 20 epochs, 5 runs per combo → mean±std AUC summary table

Usage:
    python myArkCode.py                  # random init
    python myArkCode.py --pretrained     # ImageNet-1K init
"""

import sys
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.cuda.amp import GradScaler, autocast

import timm
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Make Ark dataloader importable
# ---------------------------------------------------------------------------
_DATALOADER_DIR = os.path.join(os.path.dirname(__file__), "Ark_Plus", "Finetuning")
if _DATALOADER_DIR not in sys.path:
    sys.path.insert(0, _DATALOADER_DIR)

from dataloader import (
    ChestXray14,
    CheXpert,
    MIMIC,
    build_transform_classification,
)

# ---------------------------------------------------------------------------
# Constants — fill in DATA_DIRS to point at your image roots
# ---------------------------------------------------------------------------

# Root directories where the actual images live (edit these)
DATA_DIRS = {
    "ChestXray14": "/path/to/ChestXray14/images",
    "CheXpert":    "/path/to/CheXpert",
    "MIMIC":       "/path/to/MIMIC-CXR",
}

# Split list files (already present in the repo)
_DS = os.path.join(os.path.dirname(__file__), "Ark_Plus", "dataset")

TRAIN_LISTS = {
    "ChestXray14": os.path.join(_DS, "ChestXray14", "Xray14_train_official.txt"),
    "CheXpert":    os.path.join(_DS, "CheXpert",    "CheXpert_train_official.csv"),
    "MIMIC":       os.path.join(_DS, "MIMIC",       "mimic-cxr-2.0.0-train.csv"),
}
VAL_LISTS = {
    "ChestXray14": os.path.join(_DS, "ChestXray14", "Xray14_val_official.txt"),
    "CheXpert":    os.path.join(_DS, "CheXpert",    "CheXpert_valid_official.csv"),
    "MIMIC":       os.path.join(_DS, "MIMIC",       "mimic-cxr-2.0.0-validate.csv"),
}
TEST_LISTS = {
    "ChestXray14": os.path.join(_DS, "ChestXray14", "Xray14_test_official.txt"),
    "CheXpert":    os.path.join(_DS, "CheXpert",    "CheXpert_test_official.csv"),
    "MIMIC":       os.path.join(_DS, "MIMIC",       "mimic-cxr-2.0.0-test.csv"),
}

DATASET_DISEASES = {
    "ChestXray14": [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
    ],
    "CheXpert": [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
        "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
        "Fracture", "Support Devices",
    ],
    "MIMIC": [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
        "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
        "Fracture", "Support Devices",
    ],
}

MODELS = ["swin_base", "convnext_base", "internimage"]
DATASETS = ["ChestXray14", "CheXpert", "MIMIC"]

DEFAULT_NUM_EPOCHS = 20
DEFAULT_NUM_RUNS   = 5
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR         = 1e-3
IMG_SIZE           = 256
CROP_SIZE          = 224
NUM_WORKERS        = 4
NORMALIZATION      = "imagenet"

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """
    Build a classification model via timm.

    model_name options:
      - 'swin_base'     : Swin Transformer Base (window=7, 224)
      - 'convnext_base' : ConvNeXt Base
      - 'internimage'   : ConvNeXt-XLarge used as InternImage proxy
                          (InternImage requires DCNv3 which is not in timm;
                           ConvNeXt-XL is a strong conv-based stand-in)
    """
    if model_name == "swin_base":
        model = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif model_name == "convnext_base":
        model = timm.create_model(
            "convnext_base.fb_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif model_name == "internimage":
        # InternImage (DCNv3-based) is not available in timm.
        # We use ConvNeXt-XLarge as a strong conv-based proxy.
        model = timm.create_model(
            "convnext_xlarge.fb_in22k_ft_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model

# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_datasets(dataset_name: str):
    """Return (train_dataset, val_dataset, test_dataset) for the given dataset."""
    data_dir   = DATA_DIRS[dataset_name]
    train_list = TRAIN_LISTS[dataset_name]
    val_list   = VAL_LISTS[dataset_name]
    test_list  = TEST_LISTS[dataset_name]

    train_tf = build_transform_classification(
        normalize=NORMALIZATION, mode="train",
        crop_size=CROP_SIZE, resize=IMG_SIZE,
    )
    val_tf = build_transform_classification(
        normalize=NORMALIZATION, mode="valid",
        crop_size=CROP_SIZE, resize=IMG_SIZE,
    )
    # Use test_augment=False → simple center-crop (faster, no TenCrop)
    test_tf = build_transform_classification(
        normalize=NORMALIZATION, mode="test",
        crop_size=CROP_SIZE, resize=IMG_SIZE,
        test_augment=False,
    )

    num_class = len(DATASET_DISEASES[dataset_name])

    if dataset_name == "ChestXray14":
        train_ds = ChestXray14(data_dir, train_list, augment=train_tf, num_class=num_class)
        val_ds   = ChestXray14(data_dir, val_list,   augment=val_tf,   num_class=num_class)
        test_ds  = ChestXray14(data_dir, test_list,  augment=test_tf,  num_class=num_class)

    elif dataset_name == "CheXpert":
        train_ds = CheXpert(data_dir, train_list, augment=train_tf, num_class=num_class,
                            uncertain_label="LSR-Ones", unknown_label=0)
        val_ds   = CheXpert(data_dir, val_list,   augment=val_tf,   num_class=num_class,
                            uncertain_label="LSR-Ones", unknown_label=0)
        test_ds  = CheXpert(data_dir, test_list,  augment=test_tf,  num_class=num_class,
                            uncertain_label="Ones",     unknown_label=0)

    elif dataset_name == "MIMIC":
        train_ds = MIMIC(data_dir, train_list, augment=train_tf, num_class=num_class,
                         uncertain_label="LSR-Ones", unknown_label=0)
        val_ds   = MIMIC(data_dir, val_list,   augment=val_tf,   num_class=num_class,
                         uncertain_label="LSR-Ones", unknown_label=0)
        test_ds  = MIMIC(data_dir, test_list,  augment=test_tf,  num_class=num_class,
                         uncertain_label="Ones",     unknown_label=0)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_ds, val_ds, test_ds

# ---------------------------------------------------------------------------
# Train / Evaluate helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """One epoch of FP16 training. Returns mean loss."""
    model.train()
    total_loss = []

    for images, labels in loader:
        images = images.float().to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        with autocast(dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss.append(loss.item())

    return sum(total_loss) / len(total_loss)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate model. Returns dict with 'loss' and 'auc' (macro, multi-label).
    Handles both 4-D (center-crop) and 5-D (TenCrop) inputs.
    """
    model.eval()
    total_loss = []
    all_scores = []
    all_targets = []

    for images, labels in loader:
        labels_dev = labels.float().to(device)

        # Handle TenCrop: (B, n_crops, C, H, W) → (B*n_crops, C, H, W)
        if images.dim() == 5:
            bs, n_crops, c, h, w = images.size()
            images_in = images.view(-1, c, h, w).float().to(device)
        else:
            bs, n_crops = images.size(0), 1
            images_in = images.float().to(device)

        with autocast(dtype=torch.float16):
            outputs = model(images_in)
            # Average over crops for loss
            outputs_mean = outputs.view(bs, n_crops, -1).mean(1)
            loss = criterion(outputs_mean, labels_dev)
            scores = torch.sigmoid(outputs_mean)

        total_loss.append(loss.item())
        all_scores.append(scores.float().cpu().numpy())
        all_targets.append(labels.numpy())

    y_score = np.concatenate(all_scores,  axis=0)
    y_true  = np.concatenate(all_targets, axis=0)

    try:
        auc = roc_auc_score(y_true, y_score, average="macro")
    except ValueError:
        auc = float("nan")

    return {
        "loss": sum(total_loss) / len(total_loss),
        "auc":  float(auc),
    }

# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_single_run(
    model_name: str,
    dataset_name: str,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    pretrained: bool = False,
    device: torch.device = torch.device("cuda"),
) -> float:
    """
    Train one model on one dataset for num_epochs.
    Prints per-epoch train AUC + test AUC.
    Returns final test AUC.
    """
    num_classes = len(DATASET_DISEASES[dataset_name])

    train_ds, val_ds, test_ds = build_datasets(dataset_name)

    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    test_loader = data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    model = build_model(model_name, num_classes, pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        train_eval = evaluate(model, train_loader, criterion, device)
        test_eval  = evaluate(model, test_loader,  criterion, device)

        print(
            f"  [{model_name} | {dataset_name}] "
            f"Epoch {epoch:3d}/{num_epochs}  "
            f"Train Loss={train_loss:.4f}  "
            f"Train AUC={train_eval['auc']:.4f}  |  "
            f"Test AUC={test_eval['auc']:.4f}"
        )

    return test_eval["auc"]

# ---------------------------------------------------------------------------
# Multi-run per (model, dataset) combo
# ---------------------------------------------------------------------------

def run_combo(
    model_name: str,
    dataset_name: str,
    num_runs: int = DEFAULT_NUM_RUNS,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    pretrained: bool = False,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """5 independent runs → mean±std AUC."""
    aucs = []

    for run_idx in range(num_runs):
        print(f"\n=== {model_name} | {dataset_name} — Run {run_idx + 1}/{num_runs} ===")
        auc = train_single_run(
            model_name, dataset_name,
            num_epochs=num_epochs, batch_size=batch_size,
            lr=lr, pretrained=pretrained, device=device,
        )
        aucs.append(auc)
        print(f"  Run {run_idx + 1} final — Test AUC={auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)
    print(f"\n{model_name} | {dataset_name}: AUC={mean_auc:.4f}±{std_auc:.4f}")

    return {"aucs": aucs, "mean_auc": mean_auc, "std_auc": std_auc}

# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------

def runner(
    num_runs: int = DEFAULT_NUM_RUNS,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    pretrained: bool = False,
    device: torch.device = torch.device("cuda"),
):
    """Train all 3 models × 3 datasets, then print summary table."""
    all_results = {}

    for model_name in MODELS:
        for dataset_name in DATASETS:
            key = f"{model_name}|{dataset_name}"
            result = run_combo(
                model_name, dataset_name,
                num_runs=num_runs, num_epochs=num_epochs,
                batch_size=batch_size, lr=lr,
                pretrained=pretrained, device=device,
            )
            all_results[key] = result

    # Summary table
    init_str = "ImageNet-1K" if pretrained else "Random"
    print("\n" + "=" * 72)
    print(f"SUMMARY — Ark Chest X-ray (init={init_str}, {num_runs} runs, {num_epochs} epochs)")
    print("=" * 72)
    print(f"{'Model':20s}  {'Dataset':15s}  {'AUC (mean±std)':20s}")
    print("-" * 72)
    for model_name in MODELS:
        for dataset_name in DATASETS:
            key = f"{model_name}|{dataset_name}"
            res = all_results[key]
            print(
                f"{model_name:20s}  {dataset_name:15s}  "
                f"{res['mean_auc']:.4f}±{res['std_auc']:.4f}"
            )
    print("=" * 72)

    return all_results

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ark chest X-ray classification benchmark")
    parser.add_argument("--pretrained", action="store_true",
                        help="Load ImageNet-1K pretrained weights (default: random init)")
    parser.add_argument("--epochs",     type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--runs",       type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--device",     type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runner(
        num_runs=args.runs,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained=args.pretrained,
        device=device,
    )
