# Model Card — Clasificador educativo de neumonía en radiografías pediátricas

Documento basado en la plantilla de *Model Cards for Model Reporting* (Mitchell et al., 2019).

| Campo | Valor |
|---|---|
| **Nombre** | `pneumoniamnist-cnn-edu` (variantes: CNN base, CNN + pesos, AutoML/Optuna, ResNet-18) |
| **Versión del documento** | 1.0 — 2026-11-09 |
| **Tipo de modelo** | Clasificación binaria de imágenes (CNN / transfer learning) |
| **Responsable** | Abdieel Camargo ([@abdiellzz](https://github.com/abdiellzz)) |
| **Licencia del código** | MIT |
| **Estado** | **Solo educativo. No desplegado en ningún entorno real.** |

## ⚠️ Uso previsto

**Uso previsto:** docencia y demostración de metodología de *machine learning* aplicada a
imágenes médicas: comparación de enfoques, intervalos de confianza, tests de significación
y discusión honesta de limitaciones.

**Usos NO previstos (prohibidos por diseño del proyecto):**

- Diagnóstico, cribado, triaje o descarte de neumonía en cualquier paciente.
- Uso como dispositivo médico o herramienta de apoyo a la decisión clínica.
- Cualquier uso con datos de pacientes reales, presentes o futuros.
- Presentar sus métricas como evidencia de eficacia clínica.

**No es un dispositivo médico.** No ha pasado revisión regulatoria (FDA, EMA, ni ninguna
autoridad sanitaria) ni validación clínica prospectiva de ningún tipo.

## Datos

| Campo | Valor |
|---|---|
| **Fuente** | PneumoniaMNIST, de MedMNIST v2 (descarga automática con `medmnist`) |
| **Origen de las imágenes** | Radiografías de tórax pediátricas de Kermany et al. (2018) |
| **Resolución** | 28 × 28 píxeles, escala de grises |
| **Etiquetas** | 0 = `normal`, 1 = `pneumonia` |
| **Tamaños** | train 4 708 · val 524 · test 624 |
| **Desbalance (test)** | 234 normal / 390 neumonía (modelo trivial: exactitud 0.625) |
| **Licencia de los datos** | MedMNIST: CC BY 4.0; cada subconjunto conserva la licencia de su fuente original. Ver [`DATA_LICENSE.md`](DATA_LICENSE.md). |
| **Sesgos conocidos** | Un único origen y periodo; no se analiza sesgo por edad, sexo, equipo o centro |

Las imágenes están reducidas a 28 × 28 píxeles, lo que **elimina prácticamente toda la
información radiológica fina**. Esta decisión es didáctica (entrenamiento rápido en Colab),
no clínica.

## Rendimiento (conjunto de prueba, n = 624)

Umbral fijo u = 0.50. IC 95 % por bootstrap (2 000 remuestras). Los valores coinciden con
[`results/metrics.json`](results/metrics.json).

| Modelo | AUC | Exact. bal. | IC 95 % |
|---|---|---|---|
| CNN base | 0.959 | 0.781 | [0.750 – 0.813] |
| CNN + pesos | 0.968 | 0.848 | [0.818 – 0.878] |
| AutoML (Optuna) | 0.969 | 0.892 | [0.866 – 0.918] |
| ResNet-18 | 0.976 | 0.869 | [0.841 – 0.897] |

**Ninguna diferencia entre AutoML y ResNet-18 es estadísticamente significativa**
(McNemar p = 0.2430). La única diferencia significativa es AutoML vs CNN con pesos
(p = 0.0014, también tras Bonferroni α = 0.0167).

Métricas por subgrupo: **no disponibles**. No se ha evaluado el rendimiento por edad, sexo,
centro de origen ni gravedad de la enfermedad, porque el conjunto de datos no las expone.

### Limitaciones de la evaluación

- **Potencia estadística limitada** con n = 624 y variabilidad entre semillas ≈ ± 0.02.
- **Cambio de distribución validación → prueba**: los umbrales elegidos en validación
  (0.12 AutoML; 0.33 ResNet) rindieron peor en prueba. Es el hallazgo principal del trabajo.
- **Selección optimista en validación**: 25 intentos de Optuna sobre 524 imágenes (0.978 en
  validación frente a 0.852 en prueba con el mismo umbral).
- El conjunto de prueba **no** se usó para elegir hiperparámetros ni umbrales.

## Consideraciones éticas y de uso

- **Riesgo de daño por mala interpretación.** Los resultados (~0.97 AUC) parecen buenos en
  abstracto, pero un lector no experto podría concluir que el sistema "detecta neumonía".
  No es así: en 28 × 28 píxeles y sin validación externa, no hay ninguna garantía.
- **Sensibilidad aparente engañosa.** La CNN base a u = 0.50 tiene sensibilidad 0.997 pero
  especificidad 0.564: clasifica como neumonía casi todo. Una métrica aislada de
  sensibilidad sería irresponsable en un contexto clínico.
- **Datos de menores.** El conjunto original contiene radiografías de pacientes pediátricos.
  Aquí solo se usan las versiones públicas, anonimizadas y reducidas de MedMNIST.
- **Recomendación:** cualquier estudio real necesitaría validación externa, prospectiva,
  multi-centro y revisión por comité ético antes de acercarse a un paciente.

## Cómo reproducir

```bash
pip install -r requirements.txt
jupyter notebook IAdoc.ipynb   # Entorno de ejecución -> Ejecutar todas
```

O directamente en Colab con el badge del [README](README.md). Semilla global fija (42),
`cudnn.deterministic=True` y split oficial de MedMNIST.

## Cita

Si reutilizas el código, cita el repositorio. Si reutilizas los **datos**, cita siempre a
sus autores originales:

- Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2023).
  *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image
  classification.* Scientific Data, 10(1), 41. https://doi.org/10.1038/s41597-022-01721-8
- Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). *Identifying medical diagnoses and
  treatable diseases by image-based deep learning.* Cell, 172(5), 1122–1131.e9.
  https://doi.org/10.1016/j.cell.2018.02.010
