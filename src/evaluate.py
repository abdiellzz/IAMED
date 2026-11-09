"""Evalúa un checkpoint guardado y, opcionalmente, clasifica una imagen.

Ejemplos::

    # métricas completas sobre el split de prueba
    python -m src.evaluate --checkpoint artifacts/modelo_cnn.pt

    # predecir una imagen suelta (PNG/JPG de 28x28 en escala de grises)
    python -m src.evaluate --checkpoint artifacts/modelo_cnn.pt --imagen rx.png

    # exportar las métricas a JSON para comparar corridas
    python -m src.evaluate --checkpoint artifacts/modelo_cnn.pt --json salida.json
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
from torchvision import transforms

from .data import NOMBRES_CLASES, cargar_splits
from .engine import metricas_prueba, predecir
from .model import construir_modelo


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evalúa un checkpoint PneumoniaMNIST.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--imagen", default=None, help="Ruta de una imagen 28x28 para predecir")
    p.add_argument("--json", default=None, help="Guardar las métricas en este JSON")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def _device(args):
    return torch.device(
        args.device or ("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    )


def cargar_checkpoint(ruta: str, device):
    """Reconstruye el modelo a partir del checkpoint (``weights_only=True``)."""
    ckpt = torch.load(ruta, map_location=device, weights_only=True)
    arquitectura = ckpt.get("arquitectura", "cnn")
    modelo = construir_modelo(
        arquitectura,
        num_clases=ckpt.get("num_clases", 2),
        preentrenado=False,
        device=device,
    )
    modelo.load_state_dict(ckpt["estado_modelo"])
    modelo.eval()
    return modelo, ckpt


def predecir_imagen(modelo, ruta: str, device) -> dict:
    """Clasifica una imagen suelta con las transformaciones de evaluación."""
    from PIL import Image

    img = Image.open(ruta).convert("L")
    if img.size != (28, 28):
        print(f"Aviso: la imagen es {img.size}, se redimensiona a 28x28.", file=sys.stderr)
        img = img.resize((28, 28))

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(modelo(x), dim=1)[0].cpu().tolist()

    clase = max(range(len(probs)), key=probs.__getitem__)
    return {
        "clase": NOMBRES_CLASES[clase] if clase < len(NOMBRES_CLASES) else str(clase),
        "indice": clase,
        "probabilidades": probs,
        "probabilidad_neumonia": probs[1] if len(probs) > 1 else None,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    device = _device(args)
    modelo, ckpt = cargar_checkpoint(args.checkpoint, device)
    print(f"Checkpoint: {args.checkpoint} | arquitectura: {ckpt.get('arquitectura', 'cnn')}")

    if args.imagen:
        r = predecir_imagen(modelo, args.imagen, device)
        print(f"\nImagen: {args.imagen}")
        print(f"  Predicción: {r['clase']} ({r['probabilidades'][r['indice']]*100:.1f} %)")
        print(f"  P(neumonía): {r['probabilidad_neumonia']:.4f}")
        print("\n  Aviso: ejercicio educativo, sin validez clínica.")
        return 0

    loader = cargar_splits(args.split, batch_size=args.batch_size, entrenamiento=False)
    y, _, probs = predecir(modelo, loader, device)
    m = metricas_prueba(y, probs[:, 1], umbral=0.5)

    umbral_val = ckpt.get("umbral")
    if umbral_val is not None:
        m["bal_umbral_guardado"] = metricas_prueba(
            y, probs[:, 1], umbral=float(umbral_val)
        )["exactitud_balanceada"]
        m["umbral_guardado"] = float(umbral_val)

    print(f"\n=== {args.split.upper()} (n={len(y)}) ===")
    print(f"AUC                  : {m['auc']:.3f}")
    print(f"Exactitud            : {m['exactitud']:.3f}")
    print(f"Exactitud balanceada : {m['exactitud_balanceada']:.3f}")
    print(f"Sensibilidad         : {m['sensibilidad']:.3f}")
    print(f"Especificidad        : {m['especificidad']:.3f}")
    print(f"F1                   : {m['f1']:.3f}")
    if "bal_umbral_guardado" in m:
        print(f"Bal @ umbral {m['umbral_guardado']:.2f}  : {m['bal_umbral_guardado']:.3f}")
    cm = m["matriz_confusion"]
    print(f"Matriz               : TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print("\nAviso: ejercicio educativo, sin validez clínica.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
        print(f"Métricas guardadas en {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
