"""Bucle de entrenamiento y evaluación, extraído de las celdas 16, 20 y 26.

La lógica es la del cuaderno: selección del checkpoint por **pérdida de
validación** (no por exactitud), y evaluación con ``softmax``.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn


def preparar_lote(imagenes, etiquetas, device):
    x = imagenes.to(device)
    y = etiquetas.view(-1).long().to(device)
    return x, y


def ejecutar_epoca(modelo, loader, criterio, device, optimizador=None):
    """Una época. Devuelve ``(perdida_media, exactitud)``."""
    entrenando = optimizador is not None
    modelo.train(entrenando)
    perdida_total, aciertos, total = 0.0, 0, 0
    with torch.set_grad_enabled(entrenando):
        for imagenes, etiquetas in loader:
            x, y = preparar_lote(imagenes, etiquetas, device)
            salida = modelo(x)
            perdida = criterio(salida, y)
            if entrenando:
                optimizador.zero_grad()
                perdida.backward()
                optimizador.step()
            perdida_total += perdida.item() * x.size(0)
            aciertos += (salida.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return perdida_total / total, aciertos / total


@torch.no_grad()
def predecir(modelo, loader, device):
    """Devuelve ``(y_true, y_pred, y_prob)`` como arrays de NumPy."""
    modelo.eval()
    y_true, y_pred, y_prob = [], [], []
    for imagenes, etiquetas in loader:
        x, y = preparar_lote(imagenes, etiquetas, device)
        probs = torch.softmax(modelo(x), dim=1)
        y_prob.append(probs.cpu().numpy())
        y_pred.append(probs.argmax(dim=1).cpu().numpy())
        y_true.append(y.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)


def entrenar(
    modelo,
    train_loader,
    val_loader,
    device,
    epochs: int = 15,
    lr: float = 1e-3,
    weight: torch.Tensor | None = None,
    verbose: bool = True,
):
    """Entrena guardando el mejor estado por pérdida de validación.

    Devuelve ``(modelo, historial, mejor_epoca)``.
    """
    criterio = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)

    historial = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    mejor_val_loss, mejor_estado, mejor_epoca = float("inf"), None, 0

    for epoca in range(1, epochs + 1):
        t_loss, t_acc = ejecutar_epoca(modelo, train_loader, criterio, device, optimizador)
        v_loss, v_acc = ejecutar_epoca(modelo, val_loader, criterio, device)

        historial["train_loss"].append(t_loss)
        historial["train_acc"].append(t_acc)
        historial["val_loss"].append(v_loss)
        historial["val_acc"].append(v_acc)

        if v_loss < mejor_val_loss:
            mejor_val_loss = v_loss
            mejor_estado = copy.deepcopy(modelo.state_dict())
            mejor_epoca = epoca

        if verbose:
            print(
                f"Época {epoca:2d}/{epochs} | train: pérdida {t_loss:.4f}, acc {t_acc:.3f} | "
                f"val: pérdida {v_loss:.4f}, acc {v_acc:.3f}"
            )

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return modelo, historial, mejor_epoca


@torch.no_grad()
def umbral_optimo(y_true, probs, rejilla=None) -> float:
    """Umbral que maximiza la exactitud balanceada **en validación**.

    Es el mismo barrido del cuaderno (``np.linspace(0.05, 0.95, 91)``). Se llama
    solo con el split de validación; nunca con el de prueba.
    """
    from sklearn.metrics import balanced_accuracy_score

    if rejilla is None:
        rejilla = np.linspace(0.05, 0.95, 91)
    return max(
        rejilla,
        key=lambda u: balanced_accuracy_score(y_true, (probs >= u).astype(int)),
    )


def metricas_prueba(y_true, probs, umbral: float = 0.5) -> dict:
    """Métricas de prueba con umbral fijo (las de la tabla del README)."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (probs >= umbral).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "umbral": float(umbral),
        "auc": float(roc_auc_score(y_true, probs)),
        "exactitud": float(accuracy_score(y_true, pred)),
        "exactitud_balanceada": float(balanced_accuracy_score(y_true, pred)),
        "sensibilidad": float(recall_score(y_true, pred, zero_division=0)),
        "especificidad": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
