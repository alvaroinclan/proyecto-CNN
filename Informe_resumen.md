# Trabajo Final - Machine Learning II

## Replicación Parcial del Paper: *Transfer Learning Based Approach for Lung and Colon Cancer Detection Using Local Binary Pattern Features and Explainable AI Techniques*

**Autores del trabajo:** Álvaro Inclán y Emilio Domínguez 
**Asignatura:** Machine Learning II  
**Fecha:** Mayo 2026

---

## 1. Introducción

### 1.1. Contexto del problema

El cáncer es una de las principales causas de mortalidad a nivel mundial. En particular, el cáncer de pulmón contribuye al 18.4% de las muertes por cáncer a nivel global, mientras que el cáncer de colon constituye el 9.2% (Bray et al., 2018). La detección temprana de estas enfermedades es crucial para mejorar las tasas de supervivencia de los pacientes.

La clasificación automática de imágenes histopatológicas mediante técnicas de aprendizaje profundo (*deep learning*) representa una herramienta prometedora para asistir a los patólogos en el diagnóstico. Este enfoque permite analizar grandes volúmenes de imágenes de forma rápida y eficiente.

### 1.2. Dataset: LC25000

El dataset utilizado en este trabajo es el **LC25000** (*Lung and Colon Cancer Histopathological Image Dataset*), creado por Borkowski et al. (2019). Sus características principales son:

- **25,000 imágenes** a color en formato JPEG.
- **5 clases** con 5,000 imágenes cada una:
  1. **colon_aca**: Adenocarcinoma de colon
  2. **colon_n**: Tejido benigno de colon
  3. **lung_aca**: Adenocarcinoma de pulmón
  4. **lung_n**: Tejido benigno de pulmón
  5. **lung_scc**: Carcinoma de células escamosas de pulmón
- **Resolución original**: 768 × 768 píxeles.
- Las imágenes originales (1,250) fueron augmentadas mediante rotaciones (hasta 25°) y *flips* horizontales/verticales para obtener las 25,000 finales.

El dataset ya viene particionado en dos conjuntos:
- **Train and Validation Set**: ~4,500 imágenes por clase (22,501 imágenes en total).
- **Test Set**: ~500 imágenes por clase (2,499 imágenes en total).

### 1.3. Paper objetivo

El paper que buscamos replicar parcialmente es:

> **Alsubai, S. (2024).** *Transfer learning based approach for lung and colon cancer detection using local binary pattern features and explainable artificial intelligence (AI) techniques.* PeerJ Computer Science, 10, e1996.

Este paper propone el uso de modelos de *transfer learning* combinados con **características LBP** (*Local Binary Patterns*) para mejorar la precisión en la clasificación de imágenes histopatológicas de cáncer de pulmón y colon. El modelo InceptionResNetV2 con características LBP alcanza una precisión del **99.98%** en el paper original.

---

## 2. Objetivos del trabajo

### 2.1. Objetivo general

Replicar parcialmente los resultados del paper de Alsubai (2024), evaluando el impacto de las características LBP en la clasificación de imágenes histopatológicas del dataset LC25000.

### 2.2. Objetivos específicos

1. **Implementar y evaluar tres modelos** de los siete propuestos en el paper:
   - **CNN** (Red Neuronal Convolucional básica)
   - **ResNet** (Red Residual - ResNet50)
   - **InceptionResNetV2** (modelo híbrido Inception + ResNet)

2. **Aplicar preprocesamiento LBP** (*Local Binary Patterns*) a las imágenes, tal como se describe en el paper.

3. **Comparar los resultados** obtenidos con los del paper original, analizando las diferencias derivadas de nuestras limitaciones computacionales.

### 2.3. Adaptaciones por limitaciones computacionales

Debido a que ejecutaremos el código en **Google Colab** (entorno con recursos limitados en tiempo y GPU), realizamos las siguientes adaptaciones respecto al paper original:

| Aspecto | Paper original | Nuestro trabajo |
|---------|---------------|-----------------|
| **Resolución de imágenes** | 768 × 768 px | **64 × 64 px** |
| **Modelos evaluados** | 7 (CNN, VGG16, ResNet, EfficientNetB4, MobileNet, Xception, InceptionResNetV2) | **3 (CNN, ResNet, InceptionResNetV2)** |
| **Entorno de ejecución** | Dell PowerEdge T430 GPU | **Google Colab (GPU compartida)** |
| **Preprocesamiento** | Full features + LBP features | **LBP features** |
| **Partición de datos** | 70:30 (train:test) | **Usamos la partición proporcionada (~90:10) con split interno de validación (80:20 dentro del train)** |

La reducción de resolución de 768×768 a 64×64 implica una pérdida significativa de información visual, lo que probablemente resultará en métricas inferiores a las del paper. Sin embargo, esto nos permite entrenar los modelos dentro de las restricciones de Google Colab.

---

## 3. Metodología

### 3.1. Pipeline de trabajo

El pipeline completo del trabajo sigue estos pasos:

```
Imágenes LC25000 (768×768)
        │
        ▼
  Redimensionado (64×64)
        │
        ▼
  Extracción LBP Features
        │
        ▼
  Normalización [0, 1]
        │
        ▼
  Entrenamiento de modelos
   (CNN, ResNet, InceptionResNetV2)
        │
        ▼
  Evaluación y comparación
   (Accuracy, Precision, Recall, F1-Score)
```

### 3.2. Preprocesamiento: Local Binary Patterns (LBP)

**LBP** (*Local Binary Pattern*) es un descriptor de textura ampliamente utilizado en visión por computador. Su funcionamiento es el siguiente:

1. Para cada píxel de la imagen, se examina su vecindario (un conjunto de píxeles circundantes).
2. Cada píxel vecino se compara con el píxel central:
   - Si el valor del vecino es **≥** al del centro entonces se le asigna un **1**.
   - Si es **<** entonces se le asigna un **0**.
3. Se genera un patrón binario que se convierte a un número decimal, representando la información de textura local.
4. Se construye un histograma de estos valores LBP, proporcionando una representación compacta de la textura.

En nuestro caso, aplicamos LBP sobre cada canal de la imagen RGB por separado, generando una imagen LBP de 3 canales que captura la información de textura de la imagen original. Esta transformación tiene dos ventajas:
- **Robustez ante cambios de iluminación**: LBP es invariante a transformaciones monótonas de la escala de grises.
- **Eficiencia computacional**: la representación LBP puede mejorar la capacidad discriminativa de los modelos al destacar patrones de textura relevantes.

### 3.3. Modelos implementados

#### 3.3.1. CNN (Red Neuronal Convolucional básica)

Implementamos una CNN desde cero con la siguiente arquitectura:
- 3 bloques convolucionales con filtros progresivamente más profundos (32, 64, 128).
- Cada bloque incluye: Conv2D (3×3, padding='same') - BatchNormalization - ReLU - MaxPooling2D (2×2).
- Capas densas: Dense(256) - BatchNorm - Dropout(0.5) - Dense(128) - BatchNorm - Dropout(0.3).
- Capa de salida Dense(5, softmax) para las 5 clases.
- Optimizador: Adam (lr=0.001). Loss: categorical crossentropy.

#### 3.3.2. ResNet (ResNet50)

ResNet (*Residual Network*) introduce el concepto de **conexiones residuales** (*skip connections*), que permiten entrenar redes muy profundas al mitigar el problema del desvanecimiento del gradiente. Utilizamos **ResNet50** preentrenada en ImageNet con `include_top=False` y `pooling='max'`, con todas las capas entrenables (fine-tuning completo). La cabeza de clasificación consiste en BatchNormalization - Dense(256, regularización L2+L1) - Dropout(0.45) - Dense(5, softmax). Se utilizó el optimizador **Adamax** (lr=0.001).

#### 3.3.3. InceptionResNetV2

InceptionResNetV2 es un modelo híbrido que combina:
- Los **módulos Inception** para una extracción eficiente de características a múltiples escalas.
- Las **conexiones residuales de ResNet** para facilitar el entrenamiento profundo.

Utilizamos el modelo preentrenado en ImageNet con `include_top=False` y `pooling='max'`, con todas las capas entrenables. Dado que InceptionResNetV2 requiere un input mínimo de **75×75 píxeles**, se añadió una capa de redimensionado (`tf.image.resize`) que escala las imágenes de 64×64 a 75×75 mediante interpolación bilineal antes de alimentar el modelo base. La cabeza de clasificación es idéntica a la de ResNet50 (BatchNormalization - Dense(256, L2+L1) - Dropout(0.45) - Dense(5, softmax)), con optimizador **Adamax** (lr=0.001).

### 3.4. Métricas de evaluación

Siguiendo el paper original, utilizamos las siguientes métricas:

- **Accuracy** (Exactitud): Proporción de predicciones correctas sobre el total.
  
  $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Precision** (Precisión): Proporción de positivos predichos que son realmente positivos.
  
  $$Precision = \frac{TP}{TP + FP}$$

- **Recall** (Sensibilidad): Capacidad del modelo para identificar correctamente los positivos.
  
  $$Recall = \frac{TP}{TP + FN}$$

- **F1-Score**: Media armónica de Precision y Recall.
  
  $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

### 3.5. Estrategia de entrenamiento en Google Colab

Para manejar las **interrupciones de Google Colab** (límites de tiempo y recursos), implementamos:

1. **ModelCheckpoint**: Guarda el modelo automáticamente al final de cada época (o cuando mejora la métrica de validación).
2. **EarlyStopping**: Detiene el entrenamiento si la métrica de validación no mejora durante un número determinado de épocas (*patience*).
3. **Guardado en Google Drive**: Todos los checkpoints se guardan en Google Drive para no perder el progreso entre sesiones.
4. **Reanudación del entrenamiento**: El código permite cargar un modelo guardado previamente y continuar el entrenamiento desde donde se detuvo.
5. **Guardado del historial**: Se guarda el historial de entrenamiento (loss y métricas por época) en formato JSON para poder reconstruir las curvas de aprendizaje.

---

## 4. Estructura del código

El proyecto está organizado en scripts Python independientes, cada uno encargado de una fase del pipeline:

1. **`redimensionar_imagenes.py`**: Redimensiona las 25,000 imágenes del dataset LC25000 de 768×768 a 64×64 píxeles, utilizando interpolación LANCZOS para máxima calidad. Genera el directorio `lung_colon_image_set_64x64/`.

2. **`extraer_lbp_features.py`**: Aplica la transformación LBP (P=8, R=1, method='uniform') sobre cada canal RGB de las imágenes redimensionadas. Genera el directorio `lung_colon_image_set_64x64_lbp/`.

3. **`entrenar_cnn.py`**: Construye, entrena y evalúa la CNN básica sobre las imágenes LBP. Incluye carga de datos, split train/val (80/20), entrenamiento con callbacks (ModelCheckpoint, EarlyStopping) y evaluación sobre el test set.

4. **`entrenar_resnet.py`**: Construye, entrena y evalúa el modelo ResNet50 con transfer learning sobre las imágenes LBP.

5. **`entrenar_inceptionresnetv2.py`**: Construye, entrena y evalúa el modelo InceptionResNetV2 con transfer learning sobre las imágenes LBP. Incluye la adaptación del input de 64×64 a 75×75.

Cada script de entrenamiento genera un directorio de resultados (`resultados_cnn/`, `resultados_resnet/`, `resultados_inceptionresnetv2/`) con:
- El mejor modelo guardado (checkpoint por `val_accuracy`).
- El modelo final tras el entrenamiento.
- El historial de entrenamiento en formato JSON.
- Las métricas de evaluación en formato JSON.

---

## 5. Resultados

### 5.1. Resultados del paper original (LBP features, 768×768)

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| CNN | 89.50% | 91.64% | 91.19% | 91.39% |
| ResNet | 94.25% | 95.37% | 95.68% | 95.73% |
| InceptionResNetV2 | **99.98%** | **99.99%** | **99.99%** | **99.99%** |

### 5.2. Nuestros resultados (LBP features, 64×64)

| Modelo | Accuracy | Precision | Recall | F1-Score | Épocas | Tiempo |
|--------|----------|-----------|--------|----------|--------|--------|
| **CNN** | **85.11%** | **87.84%** | **85.11%** | **85.64%** | 26 | ~43 min |
| ResNet50 | 20.01% | 4.00% | 20.00% | 6.67% | 29 | ~38 min |
| InceptionResNetV2 | 19.89% | 4.01% | 19.88% | 6.68% | 20 | ~61 min |

### 5.3. Comparación con el paper (diferencia en puntos porcentuales)

| Modelo | Δ Accuracy | Δ Precision | Δ Recall | Δ F1-Score |
|--------|------------|-------------|----------|------------|
| CNN | −4.39 pp | −3.80 pp | −6.08 pp | −5.75 pp |
| ResNet50 | −74.24 pp | −91.37 pp | −75.68 pp | −89.06 pp |
| InceptionResNetV2 | −80.09 pp | −95.98 pp | −80.11 pp | −93.31 pp |

### 5.4. Detalle por clase — CNN (único modelo funcional)

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| colon_aca | 95.07% | 81.00% | 87.47% | 500 |
| colon_n | 96.58% | 96.00% | 96.29% | 500 |
| lung_aca | 63.31% | 89.40% | 74.13% | 500 |
| lung_n | 99.23% | 77.40% | 86.97% | 500 |
| lung_scc | 85.00% | 81.76% | 83.35% | 499 |

### 5.5. Matrices de confusión

**CNN** (85.11% accuracy):

|  | colon_aca | colon_n | lung_aca | lung_n | lung_scc |
|--|-----------|---------|----------|--------|----------|
| **colon_aca** | **405** | 16 | 59 | 0 | 20 |
| **colon_n** | 15 | **480** | 3 | 0 | 2 |
| **lung_aca** | 0 | 0 | **447** | 3 | 50 |
| **lung_n** | 0 | 1 | 112 | **387** | 0 |
| **lung_scc** | 6 | 0 | 85 | 0 | **408** |

**ResNet50** (20.01% accuracy) — Colapso completo: clasifica todas las muestras como *colon_aca*.

**InceptionResNetV2** (19.89% accuracy) — Colapso completo: clasifica casi todas las muestras como *colon_aca*.

### 5.6. Curvas de entrenamiento

**CNN**: El entrenamiento mostró una convergencia clara en el accuracy de train (de 74% a 99.3% en 26 épocas), aunque el accuracy de validación fue muy inestable, oscilando entre el 20% y el 85%. La val_loss mostró una variabilidad extrema (desde 0.65 hasta 45.9), indicativo de un severo sobreajuste. El early stopping detuvo el entrenamiento tras 26 épocas. A pesar de la inestabilidad en validación, el mejor checkpoint logró generalizar razonablemente al test set (85.11%).

**ResNet50**: Tanto el accuracy de train como el de validación se mantuvieron estancados en ~19.5-20% durante las 29 épocas completas. El loss de train convergió a ~1.61 (valor cercano a −ln(1/5) ≈ 1.609, lo cual confirma predicciones uniformemente aleatorias o de una sola clase). El modelo nunca logró aprender patrones discriminativos.

**InceptionResNetV2**: Comportamiento análogo a ResNet50, con el accuracy estancado en ~19-20% durante 20 épocas. El loss de train convergió también a ~1.61. El early stopping detuvo el entrenamiento ante la ausencia de mejora.

---

## 6. Conclusiones

### 6.1. Resumen de hallazgos

De los tres modelos evaluados, **únicamente la CNN entrenada desde cero logró aprender a clasificar** las imágenes histopatológicas del dataset LC25000 con features LBP a resolución 64×64, alcanzando un accuracy del **85.11%**. Los modelos de transfer learning (ResNet50 e InceptionResNetV2) **sufrieron un colapso total**, prediciendo una sola clase para todas las muestras, con un accuracy equivalente al azar (~20%).

### 6.2. Análisis del rendimiento de la CNN

La CNN básica obtuvo resultados razonables pese a la drástica reducción de resolución (de 768×768 a 64×64, una reducción de 144x en el número de píxeles). La diferencia respecto al paper fue de solo **−4.39 puntos porcentuales** en accuracy (85.11% vs. 89.50%).

Analizando el rendimiento por clase:
- Las clases de **colon** (colon_aca y colon_n) obtuvieron los mejores resultados, con F1-scores de 87.47% y 96.29% respectivamente, lo que sugiere que las texturas LBP a baja resolución conservan suficiente información para distinguir tejido benigno de adenocarcinoma de colon.
- La clase **lung_aca** (adenocarcinoma de pulmón) presentó la precision más baja (63.31%), lo que indica que muchas muestras de otras clases fueron clasificadas erróneamente como lung_aca. La matriz de confusión confirma que 112 muestras de lung_n y 85 de lung_scc fueron clasificadas incorrectamente como lung_aca.
- La confusión entre las tres clases de pulmón (lung_aca, lung_n, lung_scc) sugiere que, a baja resolución, la información de textura LBP no es suficiente para distinguir de forma fiable los diferentes tipos de tejido pulmonar, que comparten patrones histológicos más similares entre sí que respecto al tejido de colon.

### 6.3. Análisis del fracaso de ResNet50 e InceptionResNetV2

El colapso de los modelos de transfer learning es el resultado más significativo y contraintuitivo de este trabajo. Mientras el paper original reporta un 94.25% para ResNet y un 99.98% para InceptionResNetV2 (ambos con LBP a 768×768), nuestros resultados muestran un fracaso absoluto. Las causas probables son:

1. **Incompatibilidad entre los pesos preentrenados y la entrada LBP a baja resolución**: Los pesos de ImageNet fueron aprendidos sobre imágenes naturales (objetos, animales, paisajes) a resolución alta (224×224 o superior). Las imágenes LBP a 64×64 representan un dominio radicalmente diferente: texturas binarias codificadas con muy pocos niveles de intensidad (0-9 valores posibles por píxel con LBP uniform, P=8). Los filtros convolucionales preentrenados, diseñados para detectar bordes, texturas y formas naturales, no son capaces de extraer características útiles de este tipo de representación tan diferente.

2. **Resolución insuficiente para las arquitecturas profundas**: ResNet50 tiene 50 capas con múltiples operaciones de reducción espacial (stride-2 convolutions, pooling). Con una entrada de 64×64, el mapa de características se reduce rápidamente a dimensiones muy pequeñas (1×1 o 2×2) en las capas intermedias, eliminando toda la información espacial antes de que las capas más profundas puedan procesarla. InceptionResNetV2 tiene un problema aún mayor: su input mínimo es de 75×75, por lo que fue necesario un upsampling de 64→75, que introduce interpolación artificial sin añadir información real.

3. **Problema de la escala de gradientes (gradient flow)**: Al hacer fine-tuning completo (todas las capas entrenables) de modelos con millones de parámetros (~23.6M para ResNet50, ~54.3M para InceptionResNetV2) sobre un dataset relativamente pequeño de imágenes LBP a baja resolución, se produce un desajuste masivo. Los gradientes que fluyen desde la capa de salida deben recorrer decenas o cientos de capas para actualizar los pesos preentrenados. Con una señal de entrada tan pobre (LBP 64×64), el modelo no consigue encontrar un mínimo útil y colapsa hacia una solución trivial: predecir siempre la clase más frecuente (o la primera en el orden).

4. **Colapso a la solución trivial**: Las matrices de confusión de ambos modelos muestran que todas (o casi todas) las muestras se clasifican como *colon_aca*. Este patrón clásico de colapso ocurre cuando la función de loss se queda estancada en un mínimo local donde el modelo predice una distribución constante. El loss de ambos modelos convergió a ~1.61, que es exactamente −ln(1/5) ≈ 1.609, es decir, la entropía cruzada de predecir uniformemente (o una sola clase) en un problema de 5 clases.

### 6.4. ¿Por qué la CNN sí funciona y los modelos preentrenados no?

La clave está en la diferencia entre **aprender desde cero** y **adaptar conocimiento previo**:

- La **CNN básica** parte de pesos aleatorios y tiene una arquitectura mucho más simple (3 bloques convolucionales, ~6.8M parámetros totales incluyendo las capas densas). Al no tener conocimiento previo que "desaprender", puede adaptarse directamente a las peculiaridades de las imágenes LBP a baja resolución. Sus filtros convolucionales aprenden desde cero a detectar los patrones de textura binaria específicos del dominio histopatológico.

- Los modelos **preentrenados** llevan incorporado un sesgo fuerte hacia las características de ImageNet. Con imágenes LBP a 64×64, la señal de entrada es tan diferente a lo que estos modelos "esperan" que el proceso de fine-tuning no logra superar el sesgo de los pesos preentrenados. En el paper original, con resolución 768×768, las imágenes LBP aún contienen suficiente estructura espacial y riqueza de textura para que el fine-tuning funcione; a 64×64, esa información se pierde.

### 6.5. Lecciones aprendidas

1. **El transfer learning no es universalmente beneficioso**: Cuando el dominio de origen (ImageNet: imágenes naturales) difiere drásticamente del dominio objetivo (imágenes LBP histopatológicas a baja resolución), los pesos preentrenados pueden ser contraproducentes.

2. **La resolución es crítica para las features LBP**: LBP captura patrones de textura local. Al reducir la resolución de 768×768 a 64×64, la cantidad y calidad de los micropatrones de textura se reduce enormemente. Sin embargo, la CNN básica demuestra que aún queda suficiente información para una clasificación razonable (85.11%).

3. **La complejidad del modelo debe ser proporcional a la complejidad de la entrada**: Para entradas de baja resolución con información limitada, un modelo simple y entrenado desde cero puede superar ampliamente a modelos complejos preentrenados.

4. **Posibles mejoras**: Para mejorar los resultados de los modelos de transfer learning en este escenario, se podrían explorar:
   - Congelar las capas del modelo base en lugar de hacer fine-tuning completo.
   - Usar una resolución intermedia (por ejemplo, 128×128 o 224×224) que retenga más información.
   - Aplicar una tasa de aprendizaje diferencial (más baja para las capas preentrenadas).
   - Entrenar primero solo la cabeza de clasificación y luego descongelar progresivamente las capas del modelo base.
   - Omitir la transformación LBP y usar las imágenes redimensionadas directamente, ya que los modelos preentrenados están diseñados para trabajar con imágenes naturales, no con representaciones de textura.

### 6.6. Conclusión final

Este trabajo demuestra que la replicación de resultados publicados es un ejercicio valioso que no siempre produce los resultados esperados. La reducción drástica de resolución (de 768×768 a 64×64) impacta de forma desigual a los distintos modelos: mientras la CNN básica mantiene un rendimiento aceptable (85.11% vs. 89.50% del paper), los modelos de transfer learning colapsan completamente. Esto subraya la importancia de considerar la compatibilidad entre el preprocesamiento aplicado, la resolución de las imágenes y la arquitectura del modelo al diseñar pipelines de clasificación de imágenes médicas.

---

## 7. Referencias

1. Borkowski, A. A., Bui, M. M., Thomas, L. B., Wilson, C. P., DeLand, L. A., & Mastorides, S. M. (2019). Lung and Colon Cancer Histopathological Image Dataset (LC25000).

2. Alsubai, S. (2024). Transfer learning based approach for lung and colon cancer detection using local binary pattern features and explainable artificial intelligence (AI) techniques. *PeerJ Computer Science*, 10, e1996.

3. Bray, F., Ferlay, J., Soerjomataram, I., Siegel, R. L., Torre, L. A., & Jemal, A. (2018). Global cancer statistics 2018. *CA: A Cancer Journal for Clinicians*, 68(6), 394–424.
