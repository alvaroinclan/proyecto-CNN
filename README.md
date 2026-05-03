# 🔬 Clasificación de Cáncer de Pulmón y Colon con Deep Learning y LBP

**Trabajo Final — Machine Learning II**

Replicación parcial del paper: *Transfer Learning Based Approach for Lung and Colon Cancer Detection Using Local Binary Pattern Features and Explainable AI Techniques* (Alsubai, 2024).

**Autores:** Álvaro Inclán y Emilio Domínguez · **Fecha:** Mayo 2026

---

## Descripción

Este proyecto evalúa el impacto de las características **LBP** (*Local Binary Patterns*) en la clasificación de imágenes histopatológicas de cáncer de pulmón y colon, utilizando el dataset **LC25000** (25 000 imágenes, 5 clases).

Se implementaron y entrenaron **tres arquitecturas** de las siete propuestas en el paper original:

| Modelo | Tipo |
|---|---|
| **CNN** | Red convolucional básica entrenada desde cero |
| **ResNet50** | Transfer learning (pesos ImageNet) |
| **InceptionResNetV2** | Transfer learning (pesos ImageNet) |

Debido a limitaciones computacionales (Google Colab), las imágenes se redujeron de **768×768** a **64×64** píxeles.

---

## Pipeline

```
Imágenes LC25000 (768×768)
        │
        ▼
  Redimensionado (64×64)          ← redimensionar_imagenes.py
        │
        ▼
  Extracción LBP Features        ← extraer_lbp_features.py
        │
        ▼
  Normalización [0, 1]
        │
        ▼
  Entrenamiento de modelos        ← entrenar_cnn.py / entrenar_resnet.py / entrenar_inceptionresnetv2.py
        │
        ▼
  Evaluación (Accuracy, Precision, Recall, F1-Score)
```

---

## Estructura del proyecto

```
Trabajo Final ML II/
├── redimensionar_imagenes.py          # Resize 768×768 → 64×64
├── extraer_lbp_features.py            # Extracción LBP por canal RGB
├── entrenar_cnn.py                    # Entrenamiento CNN básica
├── entrenar_resnet.py                 # Entrenamiento ResNet50
├── entrenar_inceptionresnetv2.py      # Entrenamiento InceptionResNetV2
├── Informe_resumen.md                 # Informe detallado del trabajo
├── Paper objetivo.pdf                 # Paper replicado (Alsubai, 2024)
├── Paper explicativo dataset.pdf      # Documentación del dataset LC25000
├── lung_colon_image_set/              # Dataset original (768×768)
├── lung_colon_image_set_64x64/        # Dataset redimensionado (64×64)
├── lung_colon_image_set_64x64_lbp/    # Dataset con features LBP
├── resultados_cnn/                    # Checkpoints, métricas e historial CNN
├── resultados_resnet/                 # Checkpoints, métricas e historial ResNet50
└── resultados_inceptionresnetv2/      # Checkpoints, métricas e historial InceptionResNetV2
```

---

## Resultados

### Nuestros resultados (LBP, 64×64)

| Modelo | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **CNN** | **85.11%** | **87.84%** | **85.11%** | **85.64%** |
| ResNet50 | 20.01% | 4.00% | 20.00% | 6.67% |
| InceptionResNetV2 | 19.89% | 4.01% | 19.88% | 6.68% |

### Comparación con el paper (LBP, 768×768)

| Modelo | Paper | Nosotros | Diferencia |
|---|---|---|---|
| CNN | 89.50% | 85.11% | −4.39 pp |
| ResNet50 | 94.25% | 20.01% | −74.24 pp |
| InceptionResNetV2 | 99.98% | 19.89% | −80.09 pp |

### Hallazgo principal

- La **CNN básica** mantuvo un rendimiento aceptable (−4.39 pp vs. el paper), demostrando que las texturas LBP a baja resolución aún contienen información discriminativa suficiente para un modelo ligero.
- Los modelos de **transfer learning colapsaron completamente** (~20% accuracy ≈ azar), prediciendo una sola clase para todas las muestras. La causa principal es la **incompatibilidad** entre los pesos de ImageNet (imágenes naturales, alta resolución) y las imágenes LBP a 64×64 (texturas binarias, baja resolución).

---

## Conclusiones clave

1. **El transfer learning no siempre es beneficioso**: cuando el dominio de entrada difiere radicalmente de ImageNet, los pesos preentrenados pueden ser contraproducentes.
2. **La resolución importa**: reducir de 768×768 a 64×64 supone perder el 99.3% de los píxeles, impactando especialmente a modelos profundos.
3. **Modelos simples pueden superar a modelos complejos** cuando la entrada es de baja dimensionalidad.
4. **Las features LBP retienen información útil** incluso a baja resolución para la distinción de texturas histopatológicas, al menos para arquitecturas sencillas.

---

## Requisitos

- Python 3.8+
- TensorFlow / Keras
- scikit-image (para LBP)
- NumPy, Pillow, scikit-learn
- Google Colab (GPU recomendada)

---

## Uso

```bash
# 1. Redimensionar imágenes
python redimensionar_imagenes.py

# 2. Extraer features LBP
python extraer_lbp_features.py

# 3. Entrenar modelos (individualmente)
python entrenar_cnn.py
python entrenar_resnet.py
python entrenar_inceptionresnetv2.py
```

> **Nota:** Los scripts de entrenamiento están diseñados para Google Colab con guardado de checkpoints en Google Drive, permitiendo reanudar el entrenamiento tras interrupciones.

---

## Referencias

1. Alsubai, S. (2024). *Transfer learning based approach for lung and colon cancer detection using local binary pattern features and explainable AI techniques.* PeerJ Computer Science, 10, e1996.
2. Borkowski, A. A. et al. (2019). *Lung and Colon Cancer Histopathological Image Dataset (LC25000).*
3. Bray, F. et al. (2018). *Global cancer statistics 2018.* CA: A Cancer Journal for Clinicians, 68(6), 394–424.
