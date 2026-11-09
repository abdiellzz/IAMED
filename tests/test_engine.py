"""Tests de métricas, umbral y bucle de entrenamiento con datos sintéticos."""

import numpy as np
import pytest
import torch
from sklearn.metrics import balanced_accuracy_score

from src.engine import metricas_prueba, predecir, umbral_optimo


def test_metricas_con_prediccion_perfecta():
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    m = metricas_prueba(y, probs, umbral=0.5)
    assert m["exactitud"] == 1.0
    assert m["auc"] == 1.0
    assert m["sensibilidad"] == 1.0
    assert m["especificidad"] == 1.0
    assert m["matriz_confusion"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_metricas_reproducen_la_cnn_base_del_readme():
    """TN=132 FP=102 FN=1 TP=389 → exactitud 0.835, bal 0.781, espec 0.564."""
    y = np.array([0] * 234 + [1] * 390)
    probs = np.concatenate(
        [
            np.array([0.6] * 102 + [0.2] * 132),  # 102 FP, 132 TN
            np.array([0.2] * 1 + [0.9] * 389),    # 1 FN, 389 TP
        ]
    )
    m = metricas_prueba(y, probs, umbral=0.5)
    assert m["matriz_confusion"] == {"tn": 132, "fp": 102, "fn": 1, "tp": 389}
    assert m["exactitud"] == pytest.approx(0.835, abs=1e-3)
    assert m["exactitud_balanceada"] == pytest.approx(0.781, abs=1e-3)
    assert m["especificidad"] == pytest.approx(0.564, abs=1e-3)


def test_exactitud_balanceada_castiga_el_desbalance():
    """Predecir siempre la mayoritaria (390/624) da exactitud alta y bal 0.5."""
    y = np.array([0] * 234 + [1] * 390)
    probs = np.full(624, 0.9)  # todo neumonía
    m = metricas_prueba(y, probs, umbral=0.5)
    assert m["exactitud"] == pytest.approx(0.625, abs=1e-3)
    assert m["exactitud_balanceada"] == pytest.approx(0.5, abs=1e-3)


def test_umbral_optimo_maximiza_la_metrica():
    rng = np.random.default_rng(0)
    y = (rng.random(300) > 0.5).astype(int)
    probs = np.clip(y * 0.5 + 0.25 + rng.normal(0, 0.1, 300), 0, 1)
    u = umbral_optimo(y, probs)
    bal_u = balanced_accuracy_score(y, (probs >= u).astype(int))
    for otro in (0.2, 0.5, 0.8):
        assert bal_u >= balanced_accuracy_score(y, (probs >= otro).astype(int))


def test_umbral_optimo_dentro_de_la_rejilla_del_cuaderno():
    rng = np.random.default_rng(1)
    y = (rng.random(100) > 0.5).astype(int)
    probs = rng.random(100)
    assert 0.05 <= umbral_optimo(y, probs) <= 0.95


def test_predecir_devuelve_tres_arrays_alineados():
    from torch.utils.data import DataLoader, TensorDataset

    from src.model import CNNSencilla

    ds = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 2, (10, 1)))
    loader = DataLoader(ds, batch_size=4)
    y, pred, probs = predecir(CNNSencilla(), loader, torch.device("cpu"))
    assert y.shape == (10,)
    assert pred.shape == (10,)
    assert probs.shape == (10, 2)
    assert set(np.unique(pred)).issubset({0, 1})
