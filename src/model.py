"""Arquitecturas del cuaderno, sin cambios en la definición.

``CNNSencilla`` y ``construir_resnet18_adaptada`` son extracciones literales de
``IAdoc.ipynb`` (celdas 14 y 33). No se ha tocado ningún hiperparámetro: los
nombres de los módulos se mantienen para que los ``state_dict`` guardados por el
cuaderno sigan siendo compatibles.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASES = 2
NUM_CANALES = 1


class CNNSencilla(nn.Module):
    """CNN pequeña del cuaderno: 3 bloques conv + clasificador (240 962 parámetros)."""

    def __init__(self, num_clases: int = NUM_CLASES, num_canales: int = NUM_CANALES):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(num_canales, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_clases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clasificador(self.extractor(x))


def construir_resnet18_adaptada(
    num_clases: int = NUM_CLASES,
    preentrenado: bool = True,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """ResNet-18 adaptada a 1 canal de entrada y ``num_clases`` de salida.

    Igual que en el cuaderno: ``conv1`` acepta 1 canal promediando los pesos
    originales, y ``fc`` se sustituye por una capa lineal nueva.
    """
    import torchvision.models as models

    if preentrenado:
        pesos = models.ResNet18_Weights.DEFAULT
        modelo = models.resnet18(weights=pesos)
    else:
        modelo = models.resnet18(weights=None)

    conv1_original = modelo.conv1
    conv1_nueva = nn.Conv2d(
        in_channels=1,
        out_channels=conv1_original.out_channels,
        kernel_size=conv1_original.kernel_size,
        stride=conv1_original.stride,
        padding=conv1_original.padding,
        bias=conv1_original.bias is not None,
    )
    with torch.no_grad():
        conv1_nueva.weight.copy_(conv1_original.weight.mean(dim=1, keepdim=True))
    modelo.conv1 = conv1_nueva

    modelo.fc = nn.Linear(modelo.fc.in_features, num_clases)
    return modelo.to(device)


def construir_modelo(
    nombre: str = "cnn",
    num_clases: int = NUM_CLASES,
    preentrenado: bool = True,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Factoría: ``"cnn"`` o ``"resnet18"``."""
    nombre = nombre.lower()
    if nombre in ("cnn", "cnn_base", "cnnsencilla"):
        return CNNSencilla(num_clases).to(device)
    if nombre in ("resnet18", "resnet"):
        return construir_resnet18_adaptada(num_clases, preentrenado, device)
    raise ValueError(f"Modelo desconocido: {nombre!r}. Usa 'cnn' o 'resnet18'.")


def contar_parametros(modelo: nn.Module, solo_entrenables: bool = True) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad or not solo_entrenables)


def pesos_por_clase(etiquetas, num_clases: int = NUM_CLASES) -> torch.Tensor:
    """Pesos inversos a la frecuencia, igual que en la celda 26 del cuaderno.

    Es la cuenta que el cuaderno hace con ``np.bincount``: ``w = 1/conteo``
    normalizado para que sume ``num_clases``.
    """
    import numpy as np

    conteo = np.bincount(np.asarray(etiquetas).flatten(), minlength=num_clases)
    w = 1.0 / conteo
    return torch.tensor(w / w.sum() * num_clases, dtype=torch.float32)
