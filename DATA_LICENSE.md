# Licencia de los datos y de los pesos preentrenados

El código de este repositorio se distribuye bajo **MIT** (ver [`LICENSE`](LICENSE)).
Ese fichero **no** cubre los datos ni los pesos de terceros que se describen aquí.

## Datos: MedMNIST / PneumoniaMNIST

Los datos **no forman parte de este repositorio**: el cuaderno descarga el conjunto
PneumoniaMNIST de MedMNIST v2 en tiempo de ejecución.

- **MedMNIST v2** se distribuye bajo **Creative Commons Attribution 4.0 International
  (CC BY 4.0)**, salvo DermaMNIST (CC BY-NC 4.0).
- Cada subconjunto **conserva además la licencia y las condiciones de la publicación de la
  que procede**. En el caso de PneumoniaMNIST, las imágenes provienen del conjunto de
  radiografías pediátricas de **Kermany et al. (2018)**.

Por tanto, quien reutilice los datos debe respetar las licencias de MedMNIST y de la fuente
original, y citar:

- Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2023).
  *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image
  classification.* Scientific Data, 10(1), 41.
  <https://doi.org/10.1038/s41597-022-01721-8>
- Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). *Identifying medical diagnoses and
  treatable diseases by image-based deep learning.* Cell, 172(5), 1122–1131.e9.
  <https://doi.org/10.1016/j.cell.2018.02.010>

## Pesos preentrenados de ResNet-18

Se descargan de `download.pytorch.org` en tiempo de ejecución y se rigen por la licencia de
**torchvision / PyTorch (BSD-3-Clause)**.

## Aviso

Este repositorio es un ejercicio educativo retrospectivo. **No es un dispositivo médico y no
tiene validez clínica.**
