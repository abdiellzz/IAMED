# Atajos para el uso diario. `make help` lista todo.
.DEFAULT_GOAL := help
PY ?= python3
MODEL ?= cnn
EPOCHS ?= 15
CKPT ?= artifacts/modelo_$(MODEL).pt

.PHONY: help install install-dev test lint train evaluate clean

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Instala las dependencias del cuaderno
	$(PY) -m pip install -r requirements.txt

install-dev:  ## Instala dependencias + herramientas de desarrollo
	$(PY) -m pip install -r requirements-dev.txt

test:  ## Ejecuta los tests
	$(PY) -m pytest tests -q

lint:  ## Comprueba el estilo
	$(PY) -m ruff check src tests

train:  ## Entrena un modelo (MODEL=cnn|resnet18, EPOCHS=15)
	$(PY) -m src.train --model $(MODEL) --epochs $(EPOCHS) --output $(CKPT)

evaluate:  ## Evalúa el checkpoint en el split de prueba
	$(PY) -m src.evaluate --checkpoint $(CKPT) --split test

clean:  ## Borra artefactos y cachés
	rm -rf artifacts .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
