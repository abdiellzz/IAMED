"""Tests de las arquitecturas: formas, número de parámetros e invariancias.

No requieren GPU, ni datos de MedMNIST, ni conexión: solo tensores sintéticos.
"""

import numpy as np
import pytest
import torch

from src.model import (
    CNNSencilla,
    construir_modelo,
    construir_resnet18_adaptada,
    contar_parametros,
    pesos_por_clase,
)


def test_cnn_forma_de_salida():
    modelo = CNNSencilla()
    x = torch.randn(4, 1, 28, 28)
    salida = modelo(x)
    assert salida.shape == (4, 2)


def test_cnn_numero_de_parametros_coincide_con_el_cuaderno():
    """El cuaderno imprime 240 962 parámetros entrenables (celda 14)."""
    assert contar_parametros(CNNSencilla()) == 240_962


def test_cnn_acepta_lote_de_1():
    modelo = CNNSencilla().eval()
    with torch.no_grad():
        assert modelo(torch.randn(1, 1, 28, 28)).shape == (1, 2)


def test_cnn_salida_no_es_nan():
    modelo = CNNSencilla().eval()
    with torch.no_grad():
        salida = modelo(torch.randn(8, 1, 28, 28))
    assert torch.isfinite(salida).all()


def test_softmax_suma_uno():
    """La inferencia del README aplica softmax; debe dar una distribución."""
    modelo = CNNSencilla().eval()
    with torch.no_grad():
        probs = torch.softmax(modelo(torch.randn(3, 1, 28, 28)), dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)
    assert (probs >= 0).all() and (probs <= 1).all()


@pytest.mark.parametrize("nombre", ["cnn", "cnn_base", "CNNSencilla"])
def test_factoria_cnn(nombre):
    assert isinstance(construir_modelo(nombre, device="cpu"), CNNSencilla)


def test_factoria_rechaza_modelo_desconocido():
    with pytest.raises(ValueError):
        construir_modelo("vit_gigante")


def test_resnet18_adaptada_acepta_un_canal():
    """ResNet-18 con conv1 de 1 canal y fc de 2 clases (sin descargar pesos)."""
    modelo = construir_resnet18_adaptada(num_clases=2, preentrenado=False, device="cpu")
    assert modelo.conv1.in_channels == 1
    assert modelo.fc.out_features == 2
    with torch.no_grad():
        assert modelo(torch.randn(2, 1, 28, 28)).shape == (2, 2)


def test_pesos_por_clase_suman_dos_y_penalizan_la_minoritaria():
    """Conteo real del cuaderno: 1214 normal / 3494 neumonía."""
    etiquetas = np.array([0] * 1214 + [1] * 3494)
    w = pesos_por_clase(etiquetas)
    assert w.shape == (2,)
    assert pytest.approx(float(w.sum()), abs=1e-5) == 2.0
    assert w[0] > w[1]  # la clase minoritaria pesa más
    assert pytest.approx(float(w[0]), abs=1e-3) == 1.484
    assert pytest.approx(float(w[1]), abs=1e-3) == 0.516
