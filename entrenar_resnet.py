"""
==============================================================================
Entrenamiento y Validación del Modelo ResNet50 - Dataset LC25000
==============================================================================

Descripción:
    Este script entrena y evalúa un modelo ResNet50 (Transfer Learning)
    sobre las imágenes LBP del dataset LC25000 (64x64 píxeles).

    Arquitectura (basada en Alsubai, 2024):
      - ResNet50 preentrenada en ImageNet (sin capa top)
      - GlobalAveragePooling + BatchNorm
      - Dense(256) con regularización L1/L2 → Dropout(0.45)
      - Dense(5, softmax)

    Pipeline de datos:
      1. Carga imágenes LBP desde lung_colon_image_set_64x64_lbp/
      2. Normalización [0, 1]
      3. Split Train/Val (80/20) dentro del conjunto Train and Validation Set
      4. Test Set separado para evaluación final

    Callbacks implementados:
      - ModelCheckpoint: guarda el mejor modelo por val_accuracy
      - EarlyStopping: detiene si val_loss no mejora en N épocas
      - Guardado de historial en JSON para reconstruir curvas

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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

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

IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

CLASES = [
    "colon_aca",   # 0
    "colon_n",     # 1
    "lung_aca",    # 2
    "lung_n",      # 3
    "lung_scc",    # 4
]
NUM_CLASES = len(CLASES)

CONJUNTOS = ["Train and Validation Set", "Test Set"]
EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

NOMBRE_DIR_LBP = "lung_colon_image_set_64x64_lbp"
NOMBRE_DIR_RESULTADOS = "resultados_resnet"

# Hiperparámetros
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 7
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# =============================================================================
# 3. Funciones de carga de datos
# =============================================================================

def obtener_rutas(ruta_base: str = None) -> tuple:
    """
    Obtiene las rutas del dataset LBP y del directorio de resultados.
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
    """
    print("=" * 70)
    print("  CARGA Y PREPARACIÓN DE DATOS")
    print("=" * 70)

    X_trainval, y_trainval = cargar_imagenes_conjunto(
        ruta_dataset, "Train and Validation Set"
    )
    X_test, y_test = cargar_imagenes_conjunto(
        ruta_dataset, "Test Set"
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )

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
# 4. Construcción del modelo ResNet50
# =============================================================================

def construir_modelo_resnet(input_shape: tuple = IMG_SHAPE) -> Model:
    """
    Construye el modelo ResNet50 con transfer learning.

    Arquitectura (siguiendo el patrón del paper de Alsubai, 2024):
      - ResNet50 preentrenada en ImageNet (include_top=False, pooling='max')
      - BatchNormalization
      - Dense(256) con regularización L2 + L1 → ReLU
      - Dropout(0.45)
      - Dense(5, softmax)

    Parameters
    ----------
    input_shape : tuple
        Shape de la imagen de entrada (height, width, channels).

    Returns
    -------
    Model
        Modelo ResNet50 compilado.
    """
    # Base model: ResNet50 preentrenada en ImageNet
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="max",
    )

    # Congelar las capas del modelo base inicialmente
    # (se pueden descongelar para fine-tuning posterior)
    base_model.trainable = True

    # Construir el modelo completo
    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(
        256,
        kernel_regularizer=regularizers.l2(0.016),
        activity_regularizer=regularizers.l1(0.006),
        bias_regularizer=regularizers.l1(0.006),
        activation="relu",
    )(x)
    x = Dropout(rate=0.45, seed=RANDOM_SEED)(x)
    output = Dense(NUM_CLASES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adamax(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =============================================================================
# 5. Entrenamiento
# =============================================================================

def entrenar_modelo(
    model: Model,
    datos: dict,
    ruta_resultados: str,
) -> dict:
    """
    Entrena el modelo ResNet50 con callbacks de checkpoint y early stopping.
    Si existe un checkpoint previo, lo carga y reanuda el entrenamiento.
    """
    os.makedirs(ruta_resultados, exist_ok=True)

    ruta_checkpoint = os.path.join(ruta_resultados, "resnet_mejor_modelo.keras")
    ruta_historial = os.path.join(ruta_resultados, "resnet_historial.json")
    ruta_modelo_final = os.path.join(ruta_resultados, "resnet_modelo_final.keras")

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
    print(f"  ENTRENAMIENTO DEL MODELO RESNET50")
    print(f"{'=' * 70}")
    print(f"  Épocas máximas:     {EPOCHS}")
    print(f"  Batch size:         {BATCH_SIZE}")
    print(f"  Learning rate:      {LEARNING_RATE}")
    print(f"  Optimizer:          Adamax")
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

    # Combinar historial
    historial_completo = {}
    claves = ["accuracy", "val_accuracy", "loss", "val_loss"]

    for clave in claves:
        valores_previos = []
        if historial_previo and clave in historial_previo:
            valores_previos = historial_previo[clave]
        valores_nuevos = history.history.get(clave, [])
        historial_completo[clave] = valores_previos + valores_nuevos

    historial_completo["tiempo_entrenamiento_seg"] = tiempo_total

    with open(ruta_historial, "w") as f:
        json.dump(historial_completo, f, indent=2)
    print(f"[INFO] Historial guardado en: {ruta_historial}")

    model.save(ruta_modelo_final)
    print(f"[INFO] Modelo final guardado en: {ruta_modelo_final}")

    return historial_completo


# =============================================================================
# 6. Evaluación
# =============================================================================

def evaluar_modelo(
    model: Model,
    datos: dict,
    ruta_resultados: str,
) -> dict:
    """
    Evalúa el modelo sobre el conjunto de test y genera métricas.
    """
    print(f"\n{'=' * 70}")
    print(f"  EVALUACIÓN DEL MODELO RESNET50 EN TEST SET")
    print(f"{'=' * 70}")

    y_pred_proba = model.predict(datos["X_test"], batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = datos["y_test"]

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

    # Comparación con el paper (ResNet + LBP, 768×768)
    print(f"\n  Comparación con el paper (ResNet + LBP, 768×768):")
    print(f"  {'Métrica':<12s} {'Paper':>10s} {'Nuestro':>10s} {'Δ':>10s}")
    print(f"  {'─' * 44}")
    paper = {"Accuracy": 94.25, "Precision": 95.37, "Recall": 95.68, "F1-Score": 95.73}
    nuestro = {"Accuracy": acc*100, "Precision": prec*100, "Recall": rec*100, "F1-Score": f1*100}
    for metrica in paper:
        delta = nuestro[metrica] - paper[metrica]
        signo = "+" if delta >= 0 else ""
        print(f"  {metrica:<12s} {paper[metrica]:>9.2f}% {nuestro[metrica]:>9.2f}% {signo}{delta:>8.2f}%")

    # Classification report
    print(f"\n  Classification Report por clase:")
    print(f"  {'─' * 60}")
    report_str = classification_report(
        y_true, y_pred, target_names=CLASES, digits=4,
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

    ruta_metricas = os.path.join(ruta_resultados, "resnet_metricas.json")
    with open(ruta_metricas, "w") as f:
        json.dump(metricas, f, indent=2)
    print(f"\n[INFO] Métricas guardadas en: {ruta_metricas}")

    return metricas


# =============================================================================
# 7. Función principal
# =============================================================================

def main(ruta_base: str = None):
    """
    Función principal: carga datos, construye ResNet50, entrena y evalúa.
    """
    print("=" * 70)
    print("  ENTRENAMIENTO Y EVALUACIÓN - MODELO RESNET50")
    print("  Dataset LC25000 con LBP Features (64×64)")
    print("=" * 70)

    ruta_dataset, ruta_resultados = obtener_rutas(ruta_base)
    print(f"[INFO] Dataset LBP:  {ruta_dataset}")
    print(f"[INFO] Resultados:   {ruta_resultados}")

    if not os.path.isdir(ruta_dataset):
        print(f"[ERROR] Dataset LBP no encontrado: {ruta_dataset}")
        print(f"        Ejecute primero 'extraer_lbp_features.py'.")
        sys.exit(1)

    # 1. Cargar y preparar datos
    datos = preparar_datos(ruta_dataset)

    # 2. Construir modelo ResNet50
    print(f"\n{'=' * 70}")
    print(f"  ARQUITECTURA DEL MODELO RESNET50")
    print(f"{'=' * 70}")
    model = construir_modelo_resnet()

    # Mostrar resumen compacto (solo capas añadidas)
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable = total_params - trainable_params
    print(f"  Total parámetros:       {total_params:>12,d}")
    print(f"  Parámetros entrenables: {trainable_params:>12,d}")
    print(f"  Parámetros fijos:       {non_trainable:>12,d}")

    # 3. Entrenar
    historial = entrenar_modelo(model, datos, ruta_resultados)

    # 4. Evaluar sobre test
    metricas = evaluar_modelo(model, datos, ruta_resultados)

    # Resumen final
    print(f"\n{'=' * 70}")
    print(f"  RESUMEN FINAL - MODELO RESNET50")
    print(f"{'=' * 70}")
    print(f"  Accuracy:   {metricas['accuracy_pct']:.2f}%")
    print(f"  Precision:  {metricas['precision_pct']:.2f}%")
    print(f"  Recall:     {metricas['recall_pct']:.2f}%")
    print(f"  F1-Score:   {metricas['f1_score_pct']:.2f}%")
    print(f"{'=' * 70}")
    print(f"\n[INFO] Archivos generados en: {ruta_resultados}/")
    print(f"  - resnet_mejor_modelo.keras  (mejor checkpoint)")
    print(f"  - resnet_modelo_final.keras  (modelo final)")
    print(f"  - resnet_historial.json      (curvas de aprendizaje)")
    print(f"  - resnet_metricas.json       (métricas de evaluación)")


