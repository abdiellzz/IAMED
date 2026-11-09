# Clasificador de radiografías pediátricas con PyTorch (MedMNIST) — proyecto educativo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abdiellzz/IAMED/blob/main/IAdoc.ipynb)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> ## ⚠️ No tiene validez clínica
> Este repositorio es un **ejercicio educativo y retrospectivo**. Los modelos aquí
> descritos **no son dispositivos médicos**, **no están validados** y **no deben usarse
> para diagnosticar, descartar ni priorizar** a ningún paciente. Se entrenan con
> imágenes de **28 × 28 píxeles** procedentes de un único conjunto de datos público,
> sin validación externa ni prospectiva.

---

## Qué es esto (resumen en 60 segundos)

Clasificador binario (**normal** vs **neumonía**) sobre **PneumoniaMNIST** (MedMNIST v2),
un conjunto de radiografías de tórax **pediátricas** reducidas a 28 × 28 píxeles en
escala de grises.

Se comparan **cuatro enfoques** bajo el mismo split oficial (n = 624 en prueba):

1. CNN pequeña entrenada desde cero.
2. La misma CNN con **pérdida ponderada** por desbalance de clases.
3. **AutoML** (búsqueda bayesiana de hiperparámetros con Optuna/TPE + pruning).
4. **ResNet-18 preentrenada en ImageNet** con ajuste fino (*transfer learning*).

Además de las métricas, el cuaderno incluye lo que suele faltar en los portafolios:
**intervalos de confianza por bootstrap**, **test exacto de McNemar** con corrección de
Bonferroni, **intervalos de semilla** (3 ejecuciones) y una discusión explícita de los
fallos metodológicos encontrados.

| | |
|---|---|
| **Datos** | PneumoniaMNIST (MedMNIST v2) — train 4 708 · val 524 · test 624 |
| **Tarea** | Clasificación binaria (`normal` = 0, `neumonía` = 1) |
| **Desbalance en test** | 234 normal / 390 neumonía → un modelo trivial que siempre diga "neumonía" acierta **0.625** |
| **Modelo CNN** | 240 962 parámetros |
| **Modelo ResNet-18** | 11 171 266 parámetros (todas entrenables, ajuste fino) |
| **Entorno de ejecución** | Google Colab, GPU Tesla T4, 15 épocas por corrida |
| **Reproducibilidad** | Semilla fija (42), `cudnn.deterministic=True`, split oficial de MedMNIST |

---

## Resultados

Conjunto de **prueba** (n = 624). Exactitud balanceada con **umbral fijo u = 0.50**.
Intervalos de confianza al 95 % por **bootstrap** (2 000 remuestras sobre el conjunto de
prueba). Todos los valores están copiados de las salidas reales del cuaderno
(`IAdoc.ipynb`, celdas 36 y 37).

| Modelo | AUC | Exact. bal. (u = 0.50) | IC 95 % bootstrap |
|---|---|---|---|
| CNN base | 0.959 | 0.781 | [0.750 – 0.813] |
| CNN + pesos | 0.968 | 0.848 | [0.818 – 0.878] |
| AutoML (Optuna) | 0.969 | 0.892 | [0.866 – 0.918] |
| ResNet-18 | **0.976** | 0.869 | [0.841 – 0.897] |

> Nota de transparencia: el cuaderno imprime **0.868** para ResNet-18 en su sección propia
> (celda 34) y **0.869** en la comparativa final (celda 36). Esta tabla usa la salida de la
> comparativa final, que es también la que alimenta el bootstrap y McNemar; la diferencia de
> una milésima proviene de las dos salidas del cuaderno y no se ha retocado.

**Prueba exacta de McNemar** (mismo conjunto de prueba, u = 0.50):

| Comparación | Aciertos solo A | Aciertos solo B | p-valor | Conclusión |
|---|---|---|---|---|
| AutoML vs ResNet-18 | 22 | 14 | **0.2430** | No significativo |
| CNN + pesos vs ResNet-18 | 13 | 23 | **0.1325** | No significativo |
| AutoML vs CNN + pesos | 24 | 6 | **0.0014** | Significativo (también con Bonferroni α = 0.0167) |

### Métricas complementarias (CNN, u = 0.50, prueba)

| Modelo | Exactitud | Sensibilidad | Especificidad | F1 | Brier (↓ mejor) |
|---|---|---|---|---|---|
| CNN base | 0.835 | 0.997 | 0.564 | 0.883 | 0.1326 |
| CNN + pesos | 0.883 | 0.987 | 0.709 | — | 0.0964 |

> Lectura honesta de la CNN base: con u = 0.50 tiene una sensibilidad del 99.7 % y una
> especificidad del 56.4 %. Es decir, **casi todo lo clasifica como neumonía**; su
> exactitud (0.835) es solo 21 puntos mejor que la del clasificador trivial (0.625).
> Por eso la métrica principal del proyecto es la **exactitud balanceada**, no la exactitud.

### Variabilidad entre semillas (CNN, 3 semillas: 42, 7, 123)

| Configuración | AUC | Exact. bal. (u = 0.50) | Exact. bal. (umbral óptimo en val) |
|---|---|---|---|
| Sin pesos | 0.966 ± 0.005 | 0.781 ± 0.016 | 0.846 ± 0.016 |
| Con pesos | 0.967 ± 0.005 | 0.829 ± 0.018 | 0.853 ± 0.021 |

La desviación típica entre semillas (**≈ ± 0.02**) es del mismo orden que las diferencias
que se pretenden interpretar entre modelos.

### Trazabilidad de las cifras

Cada número de este README proviene de una salida del cuaderno, no de una estimación:

| Cifra | Celda del cuaderno |
|---|---|
| Tamaños de split y desbalance | 7, 11 |
| Métricas de la CNN base (exactitud, sens., espec., F1) | 19 |
| AUC y umbrales de validación | 24 |
| Comparación CNN sin/con pesos, umbral óptimo | 25 |
| Brier score y calibración | 26 |
| Variabilidad entre semillas | 27 |
| Optuna, hiperparámetros ganadores y métricas finales | 30 |
| ResNet-18 (AUC, bal. a u = 0.50 y u = 0.33) | 34 |
| Tabla comparativa + IC 95 % bootstrap | 36 |
| Test de McNemar | 37 |

---

## Hallazgos clave

1. **ResNet-18 tiene el mejor AUC (0.976)** — es decir, el mejor *ranking* de pacientes —,
   **pero no es significativamente mejor** que AutoML en exactitud balanceada
   (0.869 vs 0.892; IC 95 % solapados; McNemar p = 0.2430). Con este tamaño de muestra no
   se puede declarar un ganador.
2. **AutoML supera a la CNN simple en este split** (0.892 vs 0.781; McNemar p = 0.0014,
   significativo incluso tras Bonferroni), **pero con n = 624 la potencia estadística es
   limitada** y ± 0.02 entre semillas. El resultado es sugestivo, no concluyente.
3. **Los umbrales elegidos en validación empeoraron en prueba**: AutoML con u = 0.12 cae de
   0.892 a **0.852**; ResNet-18 con u = 0.33 cae de 0.869 a **0.848** (la CNN con pesos y
   u = 0.74, en cambio, *mejoró* de 0.848 a 0.876). Hay **evidencia de *distribution shift***
   entre los splits de validación y prueba: un umbral no es una propiedad del modelo, sino
   del reparto de datos.
4. **Las 25 pruebas de Optuna pueden haber sobreajustado la validación** (0.978 en
   validación frente a 0.852 en prueba con el mismo umbral). El valor de validación es
   optimista por construcción: se eligió el máximo de 25 intentos sobre ese mismo conjunto.
5. **No hay validez clínica.** Ejercicio retrospectivo, imágenes de 28 × 28, un único
   conjunto de datos y ningún centro externo. Nada de lo aquí presentado es apto para uso
   asistencial.

> Matiz que conviene leer junto al punto 3: la dirección del efecto **no es unánime**
> (AutoML y ResNet empeoran al transferir el umbral; la CNN con pesos mejora). Por eso el
> cuaderno lo describe como señal de inestabilidad de umbrales, no como una ley general.
> El AUC no depende de umbral y se mantiene ≈ 0.97 en los cuatro modelos.

---

## Cómo ejecutar en Colab

**Opción A — botón de Colab (recomendada):** pulsa el badge del inicio y ejecuta
*Entorno de ejecución → Ejecutar todas*. El cuaderno instala lo que necesita y descarga los
datos solo. Una GPU T4 gratuita basta; el cuaderno completo tarda aproximadamente 15–25 min.

**Opción B — local:**

```bash
pip install -r requirements.txt
jupyter notebook IAdoc.ipynb
```

El cuaderno usa `torch.device("cuda" if torch.cuda.is_available() else "cpu")`, así que
funciona en CPU (más lento) sin tocar código. Los pesos preentrenados de ResNet-18 se
descargan de `download.pytorch.org` la primera vez. En una T4, el cuaderno completo suele
tardar entre 20 y 40 minutos; la búsqueda de Optuna (25 pruebas × 8 épocas) es la parte
más lenta.

**Dónde se guardan los artefactos:** el cuaderno escribe `modelo_cnn_pneumoniamnist.pt` y
`modelo_automl.pt` en el directorio de trabajo (en Colab, `/content/`). No están versionados
en el repositorio; se regeneran al ejecutar. Los `.npz` de MedMNIST se descargan en
`./datos_medmnist/` (también ignorado por Git).

---

## Limitaciones

- **Split retrospectivo y único.** Una sola partición oficial de un solo dataset, con
  imágenes de origen pediátrico. No hay validación externa, temporal ni prospectiva.
- **Cambio de distribución validación → prueba.** Los umbrales óptimos de validación no se
  transfieren al conjunto de prueba (ver hallazgo 3). Parte de la brecha puede deberse a
  que validación y prueba no son intercambiables.
- **Resolución 28 × 28 píxeles en escala de grises.** Se pierde prácticamente toda la
  información radiológica fina: un diagnóstico real requiere mucha más resolución y, sobre
  todo, contexto clínico.
- **Una sola fuente de datos**, con etiquetas heredadas de un único estudio previo.
  No se evalúa sesgo por sexo, edad, equipo de rayos o centro.
- **Potencia estadística limitada** (n = 624 en prueba) y desbalance 234/390. Los intervalos
  de confianza bootstrap no corrigen el sesgo del split, solo su variabilidad.
- **Selección optimista en validación.** 25 intentos de Optuna sobre 524 imágenes de
  validación: la métrica de validación (0.978) es un máximo sobre muestras pequeñas.
- **Sin calibración clínica ni análisis de subgrupos.** El Brier score se reporta como
  diagnóstico de calibración, pero no se aplica recalibración.

---

## Cómo lo desplegaría en AWS

Esta sección es **de diseño**: no hay nada desplegado ni facturado. El objetivo es mostrar
cómo el cuaderno se traduciría a un flujo gestionado de MLOps en AWS, manteniendo las
mismas garantías metodológicas (nada de elegir umbrales con el conjunto de prueba).

| Pieza del cuaderno | Servicio AWS | Para qué |
|---|---|---|
| Datos `.npz` (MedMNIST) | **Amazon S3** con *versioning* | Almacén único de datos; el versionado permite reproducir cualquier corrida y auditar de dónde salió cada split. |
| Cuaderno / exploración | **Amazon SageMaker Studio** | Entorno gestionado con GPU para el desarrollo interactivo del `.ipynb`. |
| `pipeline_completo()` | **SageMaker Pipelines** | Encadenar de forma reproducible: preparación → entrenamiento → evaluación → registro (con linaje entre pasos). |
| Optuna (`estudio.optimize`) | **SageMaker Automatic Model Tuning** | Búsqueda de hiperparámetros gestionada, con *early stopping* equivalente al *pruning* de Optuna. |
| `torch.save(...)` → `.pt` | **SageMaker Model Registry** | Versionar el artefacto junto a sus métricas y *approval status*; nada pasa a producción sin aprobación. |
| `predecir()` | **SageMaker Endpoint** en tiempo real | Inferencia con un contenedor de PyTorch sobre las mismas transformaciones. Para uso educativo, un endpoint *serverless* o *Batch Transform* es más barato. |
| *Distribution shift* val → test | **SageMaker Model Monitor** | Comparar la distribución de entrada y de predicciones entre el baseline y el tráfico real, y alertar de la deriva (justo el efecto del hallazgo 3). |
| Desbalance de clases | **SageMaker Clarify** | Medir sesgo (*class imbalance*, impacto dispar entre clases) y explicar predicciones con SHAP. El remuestreo y los pesos de clase siguen siendo decisiones del pipeline, no de Clarify. |

```
   S3 (versioned: .npz, .pt, métricas)
        │
        ▼
   SageMaker Studio  ──►  SageMaker Pipelines
        │                       │
        │                       ├─► Training (CNN / ResNet-18)  ──► ECR (imagen)
        │                       ├─► Automatic Model Tuning (Optuna)
        │                       ├─► Evaluation + Clarify (sesgo, SHAP)
        │                       └─► Model Registry (candidato aprobado)
        │                                     │
        │                                     ▼
        │                            Endpoint (real-time) ◄── predecir()
        │                                     │
        └──────────────► Model Monitor ◄──────┘
                    (drift + métricas → CloudWatch)
```

Notas de criterio:

- **El umbral se elige en validación y se congela**, se despliegue donde se despliegue: el
  conjunto de prueba nunca participa en la selección del punto de corte.
- Un endpoint permanente cuesta dinero sin aportar nada a un proyecto educativo. Para una
  demo pública, *Batch Transform* o inferencia *serverless* son las opciones sensatas.
- Lo que **no** resolvería AWS: la falta de validez externa, la resolución 28 × 28 o la
  ausencia de revisión clínica. La infraestructura gestiona el flujo, no la evidencia.

---

## Estructura del repositorio

```
IAMED/
├── IAdoc.ipynb          # Cuaderno principal (ejecutado en Colab, salidas incluidas)
├── README.md            # Este documento
├── requirements.txt     # Dependencias mínimas reproducibles
├── LICENSE              # MIT para el código (los datos tienen su propia licencia)
└── .gitignore           # Artefactos, checkpoints y datos descargados
```

**Contenido del cuaderno:** exploración de datos y visualización → *baseline* CNN →
métricas y matriz de confusión → guardado/carga del modelo → pérdida ponderada por clase →
curvas ROC y calibración → variabilidad entre semillas → AutoML con Optuna → ResNet-18 con
ajuste fino → comparativa con bootstrap y McNemar → discusión y conclusiones.

---

## Créditos y cita

Datos creados y mantenidos por terceros. Si reutilizas este trabajo, cita las fuentes
originales del conjunto de datos:

- **Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2023).**
  *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image
  classification.* Scientific Data, 10(1), 41.
  <https://doi.org/10.1038/s41597-022-01721-8>
- **Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018).** *Identifying medical diagnoses
  and treatable diseases by image-based deep learning.* Cell, 172(5), 1122–1131.e9.
  <https://doi.org/10.1016/j.cell.2018.02.010>
  Conjunto de imágenes: Mendeley Data, <https://doi.org/10.17632/rscbjbr9sj.2>

Agradecimiento al equipo de MedMNIST por empaquetar y mantener estos datos en un formato
utilizable con fines docentes.

- **Autor del cuaderno:** Abdieel Camargo ([@abdiellzz](https://github.com/abdiellzz)).
- **Código de este repositorio:** licencia MIT (ver [`LICENSE`](LICENSE)).
- **Datos:** licencia propia de MedMNIST (CC BY 4.0 para el conjunto agregado; cada
  subconjunto conserva la licencia de su publicación original). **No** están cubiertos por
  la MIT de este repositorio.
