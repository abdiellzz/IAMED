"""Carga de datos con exactamente las transformaciones del cuaderno.

Referencia: ``IAdoc.ipynb``, celdas 8, 12 y 21. Las transformaciones NO se han
cambiado: entrenamiento con ``RandomRotation(10)``, normalización con
``mean=0.5, std=0.5`` y evaluación sin aumento.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

SEED = 42
DATASET = "pneumoniamnist"
BATCH_SIZE = 64
DATA_DIR = os.environ.get("MEDMNIST_DIR", "./datos_medmnist")
NOMBRES_CLASES = ["normal", "pneumonia"]


def construir_transforms(num_canales: int = 1):
    """Devuelve ``(transform_train, transform_eval)`` iguales a las del cuaderno."""
    media = [0.5] * num_canales
    desv = [0.5] * num_canales

    transform_train = transforms.Compose(
        [
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=media, std=desv),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=media, std=desv),
        ]
    )
    return transform_train, transform_eval


def cargar_splits(
    split: str = "train",
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    root: str = DATA_DIR,
    num_workers: int = 0,
    entrenamiento: bool | None = None,
) -> DataLoader:
    """``DataLoader`` de un split oficial de MedMNIST.

    ``entrenamiento`` decide si se aplica aumento de datos; por defecto se
    infiere del nombre del split (``train`` → con aumento).
    """
    import medmnist
    from medmnist import INFO

    if entrenamiento is None:
        entrenamiento = split == "train"

    os.makedirs(root, exist_ok=True)
    info = INFO[DATASET]
    clase = getattr(medmnist, info["python_class"])
    num_canales = info["n_channels"]

    transform_train, transform_eval = construir_transforms(num_canales)
    ds = clase(
        split=split,
        transform=transform_train if entrenamiento else transform_eval,
        download=True,
        root=root,
    )

    generador = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=entrenamiento,
        generator=generador if entrenamiento else None,
        num_workers=num_workers,
    )


def cargar_loaders(batch_size: int = BATCH_SIZE, seed: int = SEED, root: str = DATA_DIR):
    """Los tres loaders (train, val, test) en una línea."""
    return tuple(
        cargar_splits(s, batch_size=batch_size, seed=seed, root=root)
        for s in ("train", "val", "test")
    )


def resumen_splits(root: str = DATA_DIR) -> dict[str, int]:
    """Tamaños de los tres splits (sin descargar nada si ya están en disco)."""
    import medmnist
    from medmnist import INFO

    info = INFO[DATASET]
    clase = getattr(medmnist, info["python_class"])
    os.makedirs(root, exist_ok=True)
    return {
        s: len(clase(split=s, download=True, root=root)) for s in ("train", "val", "test")
    }
