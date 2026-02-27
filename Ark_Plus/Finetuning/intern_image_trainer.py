import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import timm

from dataloader import ChestXray14, build_transform_classification
from utils import meanAUC


# Fixed configuration (intentionally no CLI knobs except --pretrained)
NUM_RUNS = 10
NUM_CLASSES = 14
EPOCHS = 20
BATCH_SIZE = 32
NUM_WORKERS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
IMG_SIZE = 256
INPUT_SIZE = 224
NORMALIZATION = "imagenet"

# Dataset wiring (list files from repo; image dir can be set by env var)
DATA_DIR = os.environ.get("ARK_CHESTXRAY14_DATA_DIR", "")
TRAIN_LIST = os.path.join("Ark_Plus", "dataset", "ChestXray14", "Xray14_train_official.txt")
TEST_LIST = os.path.join("Ark_Plus", "dataset", "ChestXray14", "Xray14_test_official.txt")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(pretrained: bool, device: torch.device) -> nn.Module:
    model_name = "internimage_t_1k_224"
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=NUM_CLASSES)
    except Exception as e:
        available = timm.list_models("*internimage*")
        raise RuntimeError(
            f"Failed to create '{model_name}'. Available timm InternImage models: {available}. Original error: {e}"
        )

    model = model.to(device)
    return model


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.float().to(device)
            targets = targets.float().to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            y_true.append(targets.cpu().numpy())
            y_pred.append(probs.cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    auc, _ = meanAUC(y_true, y_pred)
    return auc


def train_one_run(run_idx: int, pretrained: bool, device: torch.device, train_loader: DataLoader, test_loader: DataLoader):
    set_seed(1000 + run_idx)

    model = build_model(pretrained=pretrained, device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for _ in range(EPOCHS):
        model.train()
        for images, targets in train_loader:
            images = images.float().to(device)
            targets = targets.float().to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    train_auc = collect_predictions(model, train_loader, device)
    test_auc = collect_predictions(model, test_loader, device)
    return train_auc, test_auc


def main():
    parser = argparse.ArgumentParser(description="InternImage-only trainer (10 full runs)")
    parser.add_argument(
        "--pretrained",
        type=lambda x: str(x).lower() in {"1", "true", "t", "yes", "y"},
        default=True,
        help="Whether to initialize InternImage from ImageNet-1K pretrained weights (true/false)",
    )
    args = parser.parse_args()

    if DATA_DIR == "":
        raise ValueError(
            "Set ARK_CHESTXRAY14_DATA_DIR to your ChestXray14 images directory before running."
        )

    if not os.path.exists(TRAIN_LIST) or not os.path.exists(TEST_LIST):
        raise FileNotFoundError(
            f"Expected list files at '{TRAIN_LIST}' and '{TEST_LIST}'."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ChestXray14(
        images_path=DATA_DIR,
        file_path=TRAIN_LIST,
        augment=build_transform_classification(
            normalize=NORMALIZATION,
            mode="train",
            crop_size=INPUT_SIZE,
            resize=IMG_SIZE,
        ),
    )
    test_dataset = ChestXray14(
        images_path=DATA_DIR,
        file_path=TEST_LIST,
        augment=build_transform_classification(
            normalize=NORMALIZATION,
            mode="test",
            crop_size=INPUT_SIZE,
            resize=IMG_SIZE,
            test_augment=False,
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Using device: {device}")
    print(f"InternImage pretrained: {args.pretrained}")
    print(f"Running {NUM_RUNS} full training runs...")

    train_aucs, test_aucs = [], []
    for run in range(NUM_RUNS):
        train_auc, test_auc = train_one_run(run, args.pretrained, device, train_loader, test_loader)
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)
        print(f"Run {run + 1}/{NUM_RUNS} | Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f}")

    train_aucs = np.array(train_aucs)
    test_aucs = np.array(test_aucs)

    print("\n===== Summary over 10 runs =====")
    print(f"Train AUC mean/std: {train_aucs.mean():.4f} / {train_aucs.std():.4f}")
    print(f"Test  AUC mean/std: {test_aucs.mean():.4f} / {test_aucs.std():.4f}")


if __name__ == "__main__":
    main()
