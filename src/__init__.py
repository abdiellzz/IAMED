"""Utilidades reproducibles del clasificador PneumoniaMNIST.

Este paquete extrae, sin reescribirla, la lógica que ya está validada en
``IAdoc.ipynb``: arquitecturas, carga de datos, bucle de entrenamiento y
evaluación. El cuaderno sigue siendo la narrativa; esto es la versión
reutilizable.

Uso típico::

    python -m src.train --model cnn --epochs 1
    python -m src.evaluate --checkpoint artifacts/modelo_cnn.pt
"""

__version__ = "0.1.0"

__all__ = ["data", "model", "train", "evaluate"]
