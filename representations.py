from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gdown
import zipfile
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm.auto import tqdm


BASE_DATA_PATH = Path(".") / "data"
ZIP_PATH = BASE_DATA_PATH / "part1.zip"
DATA_PATH = BASE_DATA_PATH / "part1"


FRUIT_CLASS_NAMES = [
    "granny_smith",
    "strawberry",
    "orange",
    "lemon",
    "fig",
    "pineapple",
    "banana",
    "jackfruit",
    "custard_apple",
    "pomegranate",
]

datas = {}

from sklearn.metrics import confusion_matrix
import seaborn as sns


def download_data():
    BASE_DATA_PATH.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        file_id = "1qNFjQIBck90I41aiJMpZR70NZRHQtEqE"
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            output=str(ZIP_PATH),
            quiet=True,
        )

    if not DATA_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(str(BASE_DATA_PATH))


def load_data(split_name: str):
    download_data()
    images = np.load(DATA_PATH / f"{split_name}_images.npy")
    labels = np.load(DATA_PATH / f"{split_name}_labels.npy")
    return images, labels


def get_data(split_name: str):
    if split_name not in datas:
        datas[split_name] = load_data(split_name)
    return datas[split_name]


class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, split_name: str, transform: Optional[Callable] = None):
        self.images, self.labels = get_data(split_name)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self):
        return int(np.max(self.labels) + 1)


def visualize_samples(
    split_name: str,
    num_rows: int = 2,
    figsize: tuple[int, int] = (10, 5),
    seed: int = 0,
):
    dataset = FruitDataset(split_name)
    rng = np.random.RandomState(seed)
    _, axes = plt.subplots(num_rows, dataset.num_classes // num_rows, figsize=figsize)
    for class_idx in range(dataset.num_classes):
        row = class_idx // (dataset.num_classes // num_rows)
        col = class_idx % (dataset.num_classes // num_rows)
        class_indices = np.where(dataset.labels == class_idx)[0]
        random_idx = int(rng.choice(class_indices))
        img, _ = dataset[random_idx]
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{FRUIT_CLASS_NAMES[class_idx]}")
        axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(f"part1_{split_name}_samples.png")
    plt.close()


def create_feature_extractor(model_name: str, pretrained_cfg: str, device: str = "mps"):
    model = timm.create_model(model_name, pretrained=True, pretrained_cfg=pretrained_cfg)
    model.transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.head = nn.Identity()
    return model.eval().to(device)


def get_features(
    split_name: str,
    feature_extractor: torch.nn.Module,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "mps",
):
    dataset = FruitDataset(split_name, transform=feature_extractor.transform)
    feature_extractor.eval()
    features = None

    features_list = []

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"Extracting {split_name} features"):
            images = images.to(device, non_blocking=True)
            feats = feature_extractor(images)

            # usually return (B, D) but good to check
            if feats.ndim > 2:
                feats = feats.flatten(1)

            features_list.append(feats.detach().cpu())

    features = torch.cat(features_list, dim=0).numpy()

    return features, dataset.labels, dataset.num_classes


class FeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, num_classes: int):
        assert features.ndim == 2, f"Expected (N, D), got {features.shape}"
        self.features = features.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.num_classes = int(num_classes)

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.features[index]).float()
        y = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.labels)

    @classmethod
    def create(cls, split_name: str, feature_extractor: torch.nn.Module, **kwargs):
        features, labels, num_classes = get_features(split_name, feature_extractor, **kwargs)
        return cls(features, labels, num_classes)


def visualize_features_tsne(
    features_dataset,
    title: str = "Features t-SNE",
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 0,
):
    from sklearn.manifold import TSNE

    tsne = TSNE(perplexity=perplexity, max_iter=n_iter, random_state=random_state)
    features_2d = tsne.fit_transform(features_dataset.features)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=features_dataset.labels,
        cmap="tab10",
        alpha=0.7,
    )
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=FRUIT_CLASS_NAMES,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(f"part1_{title.replace(' ', '_').lower()}.png")
    plt.close()

def visualize_class_overlap(
    features_dataset,
    class_a: int = 6,  # banana
    class_b: int = 3,  # lemon
    method_name: str = "mae",
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 0,
    ):
        from sklearn.manifold import TSNE
        from scipy.stats import gaussian_kde

        mask = np.isin(features_dataset.labels, [class_a, class_b])
        features_subset = features_dataset.features[mask]
        labels_subset = features_dataset.labels[mask]

        tsne = TSNE(perplexity=perplexity, max_iter=n_iter, random_state=random_state)
        features_2d = tsne.fit_transform(features_subset)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for cls_idx, color, name in [(class_a, '#FFE135', 'banana'), (class_b, '#FFA500', 'lemon')]:
            mask_cls = labels_subset == cls_idx
            axes[0].scatter(
                features_2d[mask_cls, 0],
                features_2d[mask_cls, 1],
                c=color,
                edgecolors='black',
                linewidths=0.5,
                alpha=0.7,
                s=50,
                label=name,
            )
        axes[0].legend()
        axes[0].set_title(f'{method_name.upper()}: Banana vs Lemon')
        axes[0].set_xlabel('t-SNE Dimension 1')
        axes[0].set_ylabel('t-SNE Dimension 2')

        # density contours
        for idx, (cls_idx, name, ax) in enumerate([
            (class_a, 'banana', axes[1]),
            (class_b, 'lemon', axes[2])
        ]):
            mask_cls = labels_subset == cls_idx
            points = features_2d[mask_cls]
            
            # KDE
            kde = gaussian_kde(points.T)
            x_grid = np.linspace(features_2d[:, 0].min(), features_2d[:, 0].max(), 100)
            y_grid = np.linspace(features_2d[:, 1].min(), features_2d[:, 1].max(), 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            
            ax.contourf(X, Y, Z, levels=20, cmap='YlOrRd')
            ax.scatter(points[:, 0], points[:, 1], c='black', s=10, alpha=0.3)
            ax.set_title(f'{name.capitalize()} density')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')

        plt.tight_layout()
        plt.savefig(f'part1_{method_name}_banana_lemon_overlap.png')
        plt.close()


def train_linear_probe(
    features_dataset: FeaturesDataset,
    num_epochs: int = 32,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    num_workers: int = 4,
    device: str = "mps",
):
    linear_probe = None
    epoch_losses = []

    d = features_dataset.features.shape[1]
    c = features_dataset.num_classes
    linear_probe = nn.Linear(d, c).to(device)
    opt = torch.optim.AdamW(linear_probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    loader = torch.utils.data.DataLoader(
        features_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    epoch_losses = []
    linear_probe.train()
    for _ in tqdm(range(num_epochs), desc="Linear probe training"):
        running = 0.0
        n = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = linear_probe(x)
            loss = criterion(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = x.size(0)
            running += loss.item() * bs
            n += bs

        epoch_losses.append(running / max(n, 1))

    return linear_probe, epoch_losses


def train_finetune_probe(
    split_name: str,
    feature_extractor: torch.nn.Module,
    pretrained_linear_probe: torch.nn.Module = None,
    num_epochs: int = 2,
    batch_size: int = 32,
    feature_lr: float = 1e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    device: str = "mps",
):
    
    dataset = FruitDataset(split_name, transform=feature_extractor.transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    backbone = feature_extractor
    backbone.train()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        feat = backbone(dummy)
        if feat.ndim > 2:
            feat = feat.flatten(1)
        feat_dim = feat.shape[1]

    head = nn.Linear(feat_dim, dataset.num_classes).to(device)

    if pretrained_linear_probe is not None:
        head.load_state_dict(pretrained_linear_probe.state_dict())

    model = nn.Sequential(backbone, head).to(device)

    opt = torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": feature_lr},
            {"params": head.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []
    for _ in tqdm(range(num_epochs), desc="Finetune training"):
        model.train()
        running = 0.0
        n = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = images.size(0)
            running += loss.item() * bs
            n += bs

        epoch_losses.append(running / max(n, 1))

    return model, epoch_losses




def plot_losses(losses, name):
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training loss for {name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"part1_{name}_training_loss.png")
    plt.show()
    plt.close()


def evaluate_linear(
    linear_probe: torch.nn.Module,
    features_dataset: FeaturesDataset,
    batch_size: int = 512,
    num_workers: int = 4,
    device: str = "mps",
):
    loader = torch.utils.data.DataLoader(
        features_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    linear_probe.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = linear_probe(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

def evaluate_finetune(
    model: torch.nn.Module,
    split_name: str,
    feature_extractor: torch.nn.Module,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "mps",
):
    dataset = FruitDataset(split_name, transform=feature_extractor.transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

