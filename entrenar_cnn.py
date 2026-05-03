"""
==============================================================================
Entrenamiento y Validación del Modelo CNN - Dataset LC25000
==============================================================================

Descripción:
    Este script entrena y evalúa una CNN básica (Red Neuronal Convolucional)
    sobre las imágenes LBP del dataset LC25000 (64x64 píxeles).

    Arquitectura CNN (basada en Alsubai, 2024):
      - 3 bloques convolucionales (32, 64, 128 filtros)
      - MaxPooling2D y BatchNormalization en cada bloque
      - Dropout para regularización
      - Capas densas finales (256 → 5 clases, softmax)

    Pipeline de datos:
      1. Carga imágenes LBP desde lung_colon_image_set_64x64_lbp/
      2. Normalización [0, 1]
      3. Split Train/Val (80/20) dentro del conjunto Train and Validation Set
      4. Test Set separado para evaluación final

    Callbacks implementados:
      - ModelCheckpoint: guarda el mejor modelo por val_accuracy
      - EarlyStopping: detiene si val_loss no mejora en N épocas
      - Guardado de historial en JSON para reconstruir curvas

    Métricas de evaluación:
      - Accuracy, Precision, Recall, F1-Score
      - Matriz de confusión
      - Classification report por clase

==============================================================================
"""

# =============================================================================
# 1. Importar librerías
# =============================================================================
import os
import sys
import json
import time
import numpy as np

from PIL import Image

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

# Scikit-learn para métricas y split
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# =============================================================================
# 2. Configuración
# =============================================================================

# Resolución de las imágenes de entrada
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Clases del dataset
CLASES = [
    "colon_aca",   # 0
    "colon_n",     # 1
    "lung_aca",    # 2
    "lung_n",      # 3
    "lung_scc",    # 4
]
NUM_CLASES = len(CLASES)

# Conjuntos del dataset
CONJUNTOS = ["Train and Validation Set", "Test Set"]

# Extensiones válidas
EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Directorio del dataset LBP
NOMBRE_DIR_LBP = "lung_colon_image_set_64x64_lbp"

# Directorio para guardar resultados del modelo
NOMBRE_DIR_RESULTADOS = "resultados_cnn"

# Hiperparámetros de entrenamiento
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2  # 80% train, 20% val (dentro del Train and Validation Set)
EARLY_STOPPING_PATIENCE = 7
RANDOM_SEED = 42

# Reproducibilidad
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# =============================================================================
# 3. Funciones de carga de datos
# =============================================================================

def obtener_rutas(ruta_base: str = None) -> tuple:
    """
    Obtiene las rutas del dataset LBP y del directorio de resultados.

    Parameters
    ----------
    ruta_base : str, optional
        Ruta base del proyecto.

    Returns
    -------
    tuple
        (ruta_dataset, ruta_resultados)
    """
    if ruta_base is None:
        ruta_base = os.path.dirname(os.path.abspath(__file__))

    ruta_dataset = os.path.join(ruta_base, NOMBRE_DIR_LBP)
    ruta_resultados = os.path.join(ruta_base, NOMBRE_DIR_RESULTADOS)

    return ruta_dataset, ruta_resultados


def cargar_imagenes_conjunto(
    ruta_dataset: str,
    nombre_conjunto: str,
) -> tuple:
    """
    Carga todas las imágenes de un conjunto y sus etiquetas.

    Parameters
    ----------
    ruta_dataset : str
        Ruta al directorio del dataset LBP.
    nombre_conjunto : str
        Nombre del conjunto a cargar.

    Returns
    -------
    tuple
        (imagenes, etiquetas) como arrays numpy.
        imagenes: shape (N, 64, 64, 3), dtype float32, rango [0, 1]
        etiquetas: shape (N,), dtype int, valores 0-4
    """
    imagenes = []
    etiquetas = []

    print(f"\n[INFO] Cargando conjunto: {nombre_conjunto}")

    for idx_clase, clase in enumerate(CLASES):
        directorio = os.path.join(ruta_dataset, nombre_conjunto, clase)

        if not os.path.isdir(directorio):
            print(f"  [WARN] Directorio no encontrado: {directorio}")
            continue

        archivos = [
            f for f in os.listdir(directorio)
            if os.path.splitext(f)[1].lower() in EXTENSIONES_VALIDAS
        ]

        for archivo in archivos:
            ruta_img = os.path.join(directorio, archivo)
            try:
                img = Image.open(ruta_img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_array = np.array(img, dtype=np.float32) / 255.0
                imagenes.append(img_array)
                etiquetas.append(idx_clase)
            except Exception as e:
                print(f"  [ERROR] No se pudo cargar: {ruta_img} - {e}")

        print(f"  [OK] {clase}: {len([e for e in etiquetas if e == idx_clase])} imágenes")

    imagenes = np.array(imagenes)
    etiquetas = np.array(etiquetas)

    print(f"  Total: {len(imagenes)} imágenes, shape: {imagenes.shape}")
    return imagenes, etiquetas


def preparar_datos(ruta_dataset: str) -> dict:
    """
    Carga y prepara todos los datos para entrenamiento, validación y test.

    El Train and Validation Set se divide en train/val (80/20).
    El Test Set se usa íntegro para evaluación final.

    Parameters
    ----------
    ruta_dataset : str
        Ruta al directorio del dataset LBP.

    Returns
    -------
    dict
        Diccionario con X_train, y_train, X_val, y_val, X_test, y_test.
    """
    print("=" * 70)
    print("  CARGA Y PREPARACIÓN DE DATOS")
    print("=" * 70)

    # Cargar Train and Validation Set
    X_trainval, y_trainval = cargar_imagenes_conjunto(
        ruta_dataset, "Train and Validation Set"
    )

    # Cargar Test Set
    X_test, y_test = cargar_imagenes_conjunto(
        ruta_dataset, "Test Set"
    )

    # Dividir Train and Validation Set en train/val (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )

    # Convertir etiquetas a one-hot encoding
    y_train_cat = to_categorical(y_train, NUM_CLASES)
    y_val_cat = to_categorical(y_val, NUM_CLASES)
    y_test_cat = to_categorical(y_test, NUM_CLASES)

    print(f"\n{'-' * 70}")
    print(f"  Resumen de datos:")
    print(f"    Train:      {X_train.shape[0]:>6d} imágenes")
    print(f"    Validación: {X_val.shape[0]:>6d} imágenes")
    print(f"    Test:       {X_test.shape[0]:>6d} imágenes")
    print(f"    Shape:      {X_train.shape[1:]}")
    print(f"    Rango:      [{X_train.min():.1f}, {X_train.max():.1f}]")
    print(f"{'-' * 70}")

    return {
        "X_train": X_train, "y_train": y_train, "y_train_cat": y_train_cat,
        "X_val": X_val, "y_val": y_val, "y_val_cat": y_val_cat,
        "X_test": X_test, "y_test": y_test, "y_test_cat": y_test_cat,
    }


# =============================================================================
# 4. Construcción del modelo CNN
# =============================================================================

def construir_modelo_cnn(input_shape: tuple = IMG_SHAPE) -> Sequential:
    """
    Construye la arquitectura CNN básica.

    Arquitectura (siguiendo la descripción del Informe.md, Sección 3.3.1):
      - Bloque 1: Conv2D(32, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
      - Bloque 2: Conv2D(64, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
      - Bloque 3: Conv2D(128, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
      - Flatten
      - Dense(256) → BatchNorm → ReLU → Dropout(0.5)
      - Dense(128) → BatchNorm → ReLU → Dropout(0.3)
      - Dense(5, softmax)

    Parameters
    ----------
    input_shape : tuple
        Shape de la imagen de entrada (height, width, channels).

    Returns
    -------
    Sequential
        Modelo CNN compilado.
    """
    model = Sequential([
        # --- Bloque Convolucional 1 ---
        Conv2D(32, (3, 3), activation="relu", padding="same",
               input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # --- Bloque Convolucional 2 ---
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # --- Bloque Convolucional 3 ---
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # --- Capas Densas ---
        Flatten(),

        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        # --- Capa de Salida ---
        Dense(NUM_CLASES, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =============================================================================
# 5. Entrenamiento
# =============================================================================

def entrenar_modelo(
    model: Sequential,
    datos: dict,
    ruta_resultados: str,
) -> dict:
    """
    Entrena el modelo CNN con callbacks de checkpoint y early stopping.

    Si existe un checkpoint previo, lo carga y reanuda el entrenamiento.

    Parameters
    ----------
    model : Sequential
        Modelo CNN compilado.
    datos : dict
        Diccionario con los datos de train, val y test.
    ruta_resultados : str
        Ruta al directorio donde guardar checkpoints y resultados.

    Returns
    -------
    dict
        Historial de entrenamiento.
    """
    os.makedirs(ruta_resultados, exist_ok=True)

    ruta_checkpoint = os.path.join(ruta_resultados, "cnn_mejor_modelo.keras")
    ruta_historial = os.path.join(ruta_resultados, "cnn_historial.json")
    ruta_modelo_final = os.path.join(ruta_resultados, "cnn_modelo_final.keras")

    # Intentar cargar checkpoint previo
    epoca_inicial = 0
    historial_previo = None

    if os.path.exists(ruta_checkpoint):
        print(f"\n[INFO] Checkpoint encontrado: {ruta_checkpoint}")
        print(f"       Cargando pesos del mejor modelo previo...")
        model.load_weights(ruta_checkpoint)

        if os.path.exists(ruta_historial):
            with open(ruta_historial, "r") as f:
                historial_previo = json.load(f)
            epoca_inicial = len(historial_previo.get("accuracy", []))
            print(f"       Reanudando desde época {epoca_inicial + 1}")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=ruta_checkpoint,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(f"\n{'=' * 70}")
    print(f"  ENTRENAMIENTO DEL MODELO CNN")
    print(f"{'=' * 70}")
    print(f"  Épocas máximas:     {EPOCHS}")
    print(f"  Batch size:         {BATCH_SIZE}")
    print(f"  Learning rate:      {LEARNING_RATE}")
    print(f"  Early stopping:     patience={EARLY_STOPPING_PATIENCE}")
    print(f"  Checkpoint:         {ruta_checkpoint}")
    print(f"{'=' * 70}\n")

    tiempo_inicio = time.time()

    history = model.fit(
        datos["X_train"], datos["y_train_cat"],
        validation_data=(datos["X_val"], datos["y_val_cat"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        initial_epoch=epoca_inicial,
        verbose=1,
    )

    tiempo_total = time.time() - tiempo_inicio
    print(f"\n[INFO] Entrenamiento completado en {tiempo_total:.1f}s "
          f"({tiempo_total / 60:.1f} min)")

    # Combinar historial previo con el nuevo (si existe)
    historial_completo = {}
    claves = ["accuracy", "val_accuracy", "loss", "val_loss"]

    for clave in claves:
        valores_previos = []
        if historial_previo and clave in historial_previo:
            valores_previos = historial_previo[clave]
        valores_nuevos = history.history.get(clave, [])
        historial_completo[clave] = valores_previos + valores_nuevos

    historial_completo["tiempo_entrenamiento_seg"] = tiempo_total

    # Guardar historial
    with open(ruta_historial, "w") as f:
        json.dump(historial_completo, f, indent=2)
    print(f"[INFO] Historial guardado en: {ruta_historial}")

    # Guardar modelo final
    model.save(ruta_modelo_final)
    print(f"[INFO] Modelo final guardado en: {ruta_modelo_final}")

    return historial_completo


# =============================================================================
# 6. Evaluación
# =============================================================================

def evaluar_modelo(
    model: Sequential,
    datos: dict,
    ruta_resultados: str,
) -> dict:
    """
    Evalúa el modelo sobre el conjunto de test y genera métricas.

    Métricas calculadas (siguiendo el paper):
      - Accuracy, Precision, Recall, F1-Score (macro average)
      - Classification report por clase
      - Matriz de confusión

    Parameters
    ----------
    model : Sequential
        Modelo CNN entrenado.
    datos : dict
        Diccionario con los datos.
    ruta_resultados : str
        Directorio para guardar los resultados.

    Returns
    -------
    dict
        Diccionario con las métricas de evaluación.
    """
    print(f"\n{'=' * 70}")
    print(f"  EVALUACIÓN DEL MODELO CNN EN TEST SET")
    print(f"{'=' * 70}")

    # Predicciones
    y_pred_proba = model.predict(datos["X_test"], batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = datos["y_test"]

    # Métricas globales (macro average, como en el paper)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n  Métricas globales (macro average):")
    print(f"  {'─' * 40}")
    print(f"  Accuracy:   {acc * 100:.2f}%")
    print(f"  Precision:  {prec * 100:.2f}%")
    print(f"  Recall:     {rec * 100:.2f}%")
    print(f"  F1-Score:   {f1 * 100:.2f}%")
    print(f"  {'─' * 40}")

    # Comparación con el paper
    print(f"\n  Comparación con el paper (CNN + LBP, 768×768):")
    print(f"  {'Métrica':<12s} {'Paper':>10s} {'Nuestro':>10s} {'Δ':>10s}")
    print(f"  {'─' * 44}")
    paper = {"Accuracy": 89.50, "Precision": 91.64, "Recall": 91.19, "F1-Score": 91.39}
    nuestro = {"Accuracy": acc*100, "Precision": prec*100, "Recall": rec*100, "F1-Score": f1*100}
    for metrica in paper:
        delta = nuestro[metrica] - paper[metrica]
        signo = "+" if delta >= 0 else ""
        print(f"  {metrica:<12s} {paper[metrica]:>9.2f}% {nuestro[metrica]:>9.2f}% {signo}{delta:>8.2f}%")

    # Classification report
    print(f"\n  Classification Report por clase:")
    print(f"  {'─' * 60}")
    report_str = classification_report(
        y_true, y_pred,
        target_names=CLASES,
        digits=4,
    )
    for line in report_str.split("\n"):
        print(f"  {line}")

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Matriz de Confusión:")
    print(f"  {'─' * 60}")
    header = "  {:>12s}".format("") + "".join(f" {c[:8]:>8s}" for c in CLASES)
    print(header)
    for i, clase in enumerate(CLASES):
        row = f"  {clase[:12]:>12s}" + "".join(f" {cm[i, j]:>8d}" for j in range(NUM_CLASES))
        print(row)

    # Guardar métricas
    metricas = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "accuracy_pct": float(acc * 100),
        "precision_pct": float(prec * 100),
        "recall_pct": float(rec * 100),
        "f1_score_pct": float(f1 * 100),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASES, digits=4, output_dict=True
        ),
    }

    ruta_metricas = os.path.join(ruta_resultados, "cnn_metricas.json")
    with open(ruta_metricas, "w") as f:
        json.dump(metricas, f, indent=2)
    print(f"\n[INFO] Métricas guardadas en: {ruta_metricas}")

    return metricas


# =============================================================================
# 7. Función principal
# =============================================================================

def main(ruta_base: str = None):
    """
    Función principal: carga datos, construye CNN, entrena y evalúa.

    Parameters
    ----------
    ruta_base : str, optional
        Ruta base del proyecto.
    """
    print("=" * 70)
    print("  ENTRENAMIENTO Y EVALUACIÓN - MODELO CNN")
    print("  Dataset LC25000 con LBP Features (64×64)")
    print("=" * 70)

    # Rutas
    ruta_dataset, ruta_resultados = obtener_rutas(ruta_base)
    print(f"[INFO] Dataset LBP:  {ruta_dataset}")
    print(f"[INFO] Resultados:   {ruta_resultados}")

    # Verificar que el dataset LBP existe
    if not os.path.isdir(ruta_dataset):
        print(f"[ERROR] Dataset LBP no encontrado: {ruta_dataset}")
        print(f"        Ejecute primero 'extraer_lbp_features.py'.")
        sys.exit(1)

    # 1. Cargar y preparar datos
    datos = preparar_datos(ruta_dataset)

    # 2. Construir modelo CNN
    print(f"\n{'=' * 70}")
    print(f"  ARQUITECTURA DEL MODELO CNN")
    print(f"{'=' * 70}")
    model = construir_modelo_cnn()
    model.summary()

    # 3. Entrenar
    historial = entrenar_modelo(model, datos, ruta_resultados)

    # 4. Evaluar sobre test
    metricas = evaluar_modelo(model, datos, ruta_resultados)

    # Resumen final
    print(f"\n{'=' * 70}")
    print(f"  RESUMEN FINAL - MODELO CNN")
    print(f"{'=' * 70}")
    print(f"  Accuracy:   {metricas['accuracy_pct']:.2f}%")
    print(f"  Precision:  {metricas['precision_pct']:.2f}%")
    print(f"  Recall:     {metricas['recall_pct']:.2f}%")
    print(f"  F1-Score:   {metricas['f1_score_pct']:.2f}%")
    print(f"{'=' * 70}")
    print(f"\n[INFO] Archivos generados en: {ruta_resultados}/")
    print(f"  - cnn_mejor_modelo.keras  (mejor checkpoint)")
    print(f"  - cnn_modelo_final.keras  (modelo final)")
    print(f"  - cnn_historial.json      (curvas de aprendizaje)")
    print(f"  - cnn_metricas.json       (métricas de evaluación)")


