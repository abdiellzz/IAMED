"""Entrena una CNN o ResNet-18 en PneumoniaMNIST (versión CLI del cuaderno).

Ejemplos::

    python -m src.train --model cnn --epochs 1 --output artifacts/modelo_cnn.pt
    python -m src.train --model cnn --usar-pesos
    python -m src.train --model resnet18 --epochs 15 --sin-preentrenado

Los valores por defecto son los del cuaderno (semilla 42, 15 épocas, lr 1e-3,
batch 64). El umbral se calcula SOLO con validación y se guarda en el
checkpoint junto a los pesos.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from .data import BATCH_SIZE, SEED, cargar_loaders
from .engine import entrenar, metricas_prueba, predecir, umbral_optimo
from .model import construir_modelo, contar_parametros, pesos_por_clase


def fijar_semilla(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Entrena el clasificador PneumoniaMNIST.")
    p.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam lr (ResNet recomienda 1e-4)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--usar-pesos", action="store_true", help="Pérdida ponderada por desbalance")
    p.add_argument("--sin-preentrenado", action="store_true", help="ResNet desde cero")
    p.add_argument("--datos", default=None, help="Directorio de los .npz de MedMNIST")
    p.add_argument("--output", default=None, help="Ruta del checkpoint .pt")
    p.add_argument("--cpu", action="store_true", help="Forzar CPU")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(
        args.device
        or ("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    )
    fijar_semilla(args.seed)

    print(f"Dispositivo: {device}")
    root_kwargs = {"root": args.datos} if args.datos else {}
    train_loader, val_loader, test_loader = cargar_loaders(
        batch_size=args.batch_size, seed=args.seed, **root_kwargs
    )

    lr = args.lr
    if args.model == "resnet18" and lr == 1e-3:
        lr = 1e-4  # el cuaderno usa 1e-4 para el ajuste fino
        print("ResNet-18: ajusto lr a 1e-4 (como en el cuaderno).")

    modelo = construir_modelo(
        args.model, preentrenado=not args.sin_preentrenado, device=device
    )
    print(f"Modelo {args.model}: {contar_parametros(modelo):,} parámetros entrenables")

    weight = None
    if args.usar_pesos:
        etiquetas = train_loader.dataset.labels
        weight = pesos_por_clase(etiquetas)
        print(f"Pesos por clase: {weight.tolist()}")

    inicio = time.time()
    modelo, historial, mejor_epoca = entrenar(
        modelo, train_loader, val_loader, device,
        epochs=args.epochs, lr=lr, weight=weight,
    )
    duracion = time.time() - inicio

    # Umbral SOLO con validación (nunca con prueba).
    yv, _, pv = predecir(modelo, val_loader, device)
    umbral = umbral_optimo(yv, pv[:, 1])

    yt, _, pt = predecir(modelo, test_loader, device)
    metricas = metricas_prueba(yt, pt[:, 1], umbral=0.5)
    metricas["exactitud_balanceada_umbral_val"] = float(
        metricas_prueba(yt, pt[:, 1], umbral=umbral)["exactitud_balanceada"]
    )
    metricas["umbral_elegido_en_val"] = float(umbral)

    print(f"\nMejor época (mínima pérdida de val): {mejor_epoca} | {duracion:.1f} s")
    print(f"Prueba | AUC {metricas['auc']:.3f} | "
          f"bal@0.50 {metricas['exactitud_balanceada']:.3f} | "
          f"bal@{umbral:.2f} {metricas['exactitud_balanceada_umbral_val']:.3f}")

    salida = args.output or f"artifacts/modelo_{args.model}.pt"
    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    torch.save(
        {
            "estado_modelo": modelo.state_dict(),
            "arquitectura": args.model,
            "num_clases": 2,
            "canales": 1,
            "umbral": umbral,
            "seed": args.seed,
            "metricas_prueba": metricas,
        },
        salida,
    )
    print(f"Checkpoint: {salida}")

    with open(os.path.splitext(salida)[0] + "_metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
