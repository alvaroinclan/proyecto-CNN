"""
==============================================================================
Redimensionado de Imagenes - Dataset LC25000
==============================================================================


Descripción:
    Este script realiza el redimensionado de las imágenes del dataset LC25000
    (Lung and Colon Cancer Histopathological Image Dataset) de su resolución
    original de 768x768 píxeles a 64x64 píxeles.

    Esta reducción de resolución es necesaria para poder entrenar los modelos
    de deep learning (CNN, ResNet50, InceptionResNetV2) dentro de las
    limitaciones computacionales de Google Colab.

    El dataset contiene 25,000 imágenes organizadas en 5 clases:
      - colon_aca:  Adenocarcinoma de colon
      - colon_n:    Tejido benigno de colon
      - lung_aca:   Adenocarcinoma de pulmón
      - lung_n:     Tejido benigno de pulmón
      - lung_scc:   Carcinoma de células escamosas de pulmón

    El dataset ya viene particionado en:
      - Train and Validation Set: ~4,500 imágenes por clase (22,501 en total)
      - Test Set: ~500 imágenes por clase (2,499 en total)


==============================================================================
"""

# =============================================================================
# 1. Importar las librerías necesarias
# =============================================================================
import os
import sys
import time
import glob
from pathlib import Path

# Librería para procesamiento de imágenes
from PIL import Image

# Numpy para operaciones numéricas
import numpy as np


# =============================================================================
# 2. Configuración de rutas y parámetros
# =============================================================================

# -------------------------------------------------------------------------
# Resolución objetivo para las imágenes redimensionadas
# El paper original usa 768x768, pero nosotros reducimos a 64x64
# debido a las limitaciones de Google Colab (ver Informe.md, Sección 2.3)
# -------------------------------------------------------------------------
RESOLUCION_OBJETIVO = (64, 64)

# -------------------------------------------------------------------------
# Clases del dataset LC25000
# 5 clases con ~5,000 imágenes cada una (25,000 en total)
# -------------------------------------------------------------------------
CLASES = [
    "colon_aca",   # Adenocarcinoma de colon
    "colon_n",     # Tejido benigno de colon
    "lung_aca",    # Adenocarcinoma de pulmón
    "lung_n",      # Tejido benigno de pulmón
    "lung_scc",    # Carcinoma de células escamosas de pulmón
]

# -------------------------------------------------------------------------
# Conjuntos del dataset (ya vienen particionados)
# -------------------------------------------------------------------------
CONJUNTOS = [
    "Train and Validation Set",
    "Test Set",
]

# -------------------------------------------------------------------------
# Extensiones de imagen válidas
# -------------------------------------------------------------------------
EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# -------------------------------------------------------------------------
# Método de interpolación para el redimensionado
# LANCZOS proporciona la mejor calidad para reducciones de tamaño
# -------------------------------------------------------------------------
METODO_INTERPOLACION = Image.LANCZOS

# -------------------------------------------------------------------------
# Sufijo para el directorio de salida
# -------------------------------------------------------------------------
SUFIJO_SALIDA = "_64x64"


def obtener_ruta_dataset(ruta_base: str = None) -> str:
    """
    Obtiene la ruta al directorio del dataset original.

    Si se ejecuta en Google Colab, intenta montar Google Drive y usar
    la ruta estándar. Si se ejecuta localmente, usa la ruta relativa
    al directorio del script.

    Parameters
    ----------
    ruta_base : str, optional
        Ruta base al directorio que contiene el dataset.
        Si es None, se detecta automáticamente.

    Returns
    -------
    str
        Ruta al directorio del dataset.
    """
    if ruta_base is not None:
        return ruta_base


    # Ruta relativa al script
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(directorio_script, "lung_colon_image_set")
    print(f"[INFO] Ejecutando localmente. Ruta del dataset: {ruta}")
    return ruta


def crear_estructura_salida(ruta_entrada: str, ruta_salida: str) -> None:
    """
    Crea la estructura de directorios de salida replicando la estructura
    del dataset original.

    Parameters
    ----------
    ruta_entrada : str
        Ruta al directorio del dataset original.
    ruta_salida : str
        Ruta al directorio donde se guardarán las imágenes redimensionadas.
    """
    print(f"\n[INFO] Creando estructura de directorios de salida...")
    print(f"       Entrada: {ruta_entrada}")
    print(f"       Salida:  {ruta_salida}")

    for conjunto in CONJUNTOS:
        for clase in CLASES:
            directorio = os.path.join(ruta_salida, conjunto, clase)
            os.makedirs(directorio, exist_ok=True)
            print(f"       [OK] Creado: {os.path.join(conjunto, clase)}")

    print(f"[INFO] Estructura de directorios creada correctamente.\n")


def redimensionar_imagen(
    ruta_imagen_entrada: str,
    ruta_imagen_salida: str,
    resolucion: tuple = RESOLUCION_OBJETIVO,
    metodo: int = METODO_INTERPOLACION,
) -> bool:
    """
    Redimensiona una imagen individual a la resolución especificada.

    Parameters
    ----------
    ruta_imagen_entrada : str
        Ruta completa a la imagen original.
    ruta_imagen_salida : str
        Ruta completa donde se guardará la imagen redimensionada.
    resolucion : tuple
        Tupla (ancho, alto) con la resolución objetivo. Default: (64, 64).
    metodo : int
        Método de interpolación de PIL. Default: Image.LANCZOS.

    Returns
    -------
    bool
        True si la imagen se redimensionó correctamente, False en caso contrario.
    """
    try:
        # Abrir la imagen original
        with Image.open(ruta_imagen_entrada) as img:
            # Asegurarse de que la imagen está en modo RGB (3 canales)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Redimensionar la imagen
            img_redimensionada = img.resize(resolucion, metodo)

            # Guardar la imagen redimensionada
            img_redimensionada.save(ruta_imagen_salida, quality=95)

        return True

    except Exception as e:
        print(f"[ERROR] No se pudo redimensionar: {ruta_imagen_entrada}")
        print(f"        Razón: {e}")
        return False


def procesar_conjunto(
    ruta_entrada: str,
    ruta_salida: str,
    nombre_conjunto: str,
) -> dict:
    """
    Procesa todas las imágenes de un conjunto (Train/Test) redimensionándolas.

    Parameters
    ----------
    ruta_entrada : str
        Ruta al directorio del dataset original.
    ruta_salida : str
        Ruta al directorio de salida.
    nombre_conjunto : str
        Nombre del conjunto a procesar ("Train and Validation Set" o "Test Set").

    Returns
    -------
    dict
        Diccionario con estadísticas del procesamiento:
        - total: número total de imágenes encontradas
        - exitosas: número de imágenes redimensionadas correctamente
        - fallidas: número de imágenes que no se pudieron redimensionar
        - por_clase: dict con conteo por cada clase
    """
    estadisticas = {
        "total": 0,
        "exitosas": 0,
        "fallidas": 0,
        "por_clase": {},
    }

    print(f"{'=' * 70}")
    print(f"  Procesando conjunto: {nombre_conjunto}")
    print(f"{'=' * 70}")

    tiempo_inicio_conjunto = time.time()

    for clase in CLASES:
        directorio_entrada = os.path.join(ruta_entrada, nombre_conjunto, clase)
        directorio_salida = os.path.join(ruta_salida, nombre_conjunto, clase)

        # Verificar que el directorio de entrada existe
        if not os.path.isdir(directorio_entrada):
            print(f"[WARN] Directorio no encontrado: {directorio_entrada}")
            estadisticas["por_clase"][clase] = {"total": 0, "exitosas": 0}
            continue

        # Obtener lista de imágenes
        archivos = [
            f for f in os.listdir(directorio_entrada)
            if os.path.splitext(f)[1].lower() in EXTENSIONES_VALIDAS
        ]
        num_imagenes = len(archivos)
        estadisticas["total"] += num_imagenes

        print(f"\n  Clase: {clase:<12s} | Imagenes encontradas: {num_imagenes}")

        exitosas_clase = 0
        tiempo_inicio_clase = time.time()

        for idx, archivo in enumerate(archivos):
            ruta_img_entrada = os.path.join(directorio_entrada, archivo)
            ruta_img_salida = os.path.join(directorio_salida, archivo)

            if redimensionar_imagen(ruta_img_entrada, ruta_img_salida):
                exitosas_clase += 1

            # Mostrar progreso cada 500 imágenes
            if (idx + 1) % 500 == 0 or (idx + 1) == num_imagenes:
                porcentaje = (idx + 1) / num_imagenes * 100
                print(
                    f"    Progreso: {idx + 1:>5d}/{num_imagenes} "
                    f"({porcentaje:5.1f}%)"
                )

        tiempo_clase = time.time() - tiempo_inicio_clase
        fallidas_clase = num_imagenes - exitosas_clase

        estadisticas["exitosas"] += exitosas_clase
        estadisticas["fallidas"] += fallidas_clase
        estadisticas["por_clase"][clase] = {
            "total": num_imagenes,
            "exitosas": exitosas_clase,
            "tiempo": tiempo_clase,
        }

        print(
            f"    [OK] Completado: {exitosas_clase}/{num_imagenes} "
            f"en {tiempo_clase:.1f}s"
        )
        if fallidas_clase > 0:
            print(f"    [FAIL] Imagenes fallidas: {fallidas_clase}")

    tiempo_total = time.time() - tiempo_inicio_conjunto
    print(f"\n{'-' * 70}")
    print(f"  Resumen del conjunto '{nombre_conjunto}':")
    print(f"    Total procesadas: {estadisticas['exitosas']}/{estadisticas['total']}")
    print(f"    Tiempo total: {tiempo_total:.1f}s")
    if estadisticas["fallidas"] > 0:
        print(f"    Imagenes fallidas: {estadisticas['fallidas']}")
    print(f"{'-' * 70}\n")

    return estadisticas


def verificar_dataset(ruta: str) -> bool:
    """
    Verifica que el dataset existe y tiene la estructura esperada.

    Parameters
    ----------
    ruta : str
        Ruta al directorio del dataset.

    Returns
    -------
    bool
        True si el dataset tiene la estructura esperada, False en caso contrario.
    """
    print(f"[INFO] Verificando estructura del dataset en: {ruta}")

    if not os.path.isdir(ruta):
        print(f"[ERROR] El directorio del dataset no existe: {ruta}")
        return False

    todo_ok = True
    for conjunto in CONJUNTOS:
        for clase in CLASES:
            directorio = os.path.join(ruta, conjunto, clase)
            if not os.path.isdir(directorio):
                print(f"[ERROR] Directorio faltante: {directorio}")
                todo_ok = False
            else:
                # Contar imágenes
                n_imgs = len([
                    f for f in os.listdir(directorio)
                    if os.path.splitext(f)[1].lower() in EXTENSIONES_VALIDAS
                ])
                print(f"  [OK] {conjunto}/{clase}: {n_imgs} imágenes")

    return todo_ok


def mostrar_resumen_final(estadisticas_train: dict, estadisticas_test: dict) -> None:
    """
    Muestra un resumen final del procesamiento completo.

    Parameters
    ----------
    estadisticas_train : dict
        Estadísticas del conjunto de entrenamiento/validación.
    estadisticas_test : dict
        Estadísticas del conjunto de test.
    """
    print(f"\n{'=' * 70}")
    print(f"  RESUMEN FINAL DEL REDIMENSIONADO")
    print(f"{'=' * 70}")
    print(f"  Resolución original: 768 x 768 px")
    print(f"  Resolución objetivo: {RESOLUCION_OBJETIVO[0]} x {RESOLUCION_OBJETIVO[1]} px")
    print(f"  Método de interpolación: LANCZOS")
    print(f"{'-' * 70}")
    print(f"  {'Conjunto':<30s} {'Exitosas':>10s} {'Total':>10s} {'%':>8s}")
    print(f"{'-' * 70}")

    total_exitosas = 0
    total_imagenes = 0

    for nombre, est in [
        ("Train and Validation Set", estadisticas_train),
        ("Test Set", estadisticas_test),
    ]:
        total_exitosas += est["exitosas"]
        total_imagenes += est["total"]
        pct = est["exitosas"] / est["total"] * 100 if est["total"] > 0 else 0
        print(f"  {nombre:<30s} {est['exitosas']:>10d} {est['total']:>10d} {pct:>7.1f}%")

    pct_total = total_exitosas / total_imagenes * 100 if total_imagenes > 0 else 0
    print(f"{'-' * 70}")
    print(f"  {'TOTAL':<30s} {total_exitosas:>10d} {total_imagenes:>10d} {pct_total:>7.1f}%")
    print(f"{'=' * 70}")

    # Detalle por clase
    print(f"\n  Detalle por clase:")
    print(f"  {'Clase':<12s} {'Train/Val':>10s} {'Test':>10s} {'Total':>10s}")
    print(f"  {'-' * 44}")
    for clase in CLASES:
        n_train = estadisticas_train["por_clase"].get(clase, {}).get("exitosas", 0)
        n_test = estadisticas_test["por_clase"].get(clase, {}).get("exitosas", 0)
        print(f"  {clase:<12s} {n_train:>10d} {n_test:>10d} {n_train + n_test:>10d}")

    print(f"\n[INFO] Proceso de redimensionado completado exitosamente.")
    


# =============================================================================
# 3. Función principal
# =============================================================================
def main(ruta_base: str = None):
    """
    Función principal que ejecuta el pipeline de redimensionado de imágenes.

    Parameters
    ----------
    ruta_base : str, optional
        Ruta base al directorio que contiene el dataset.
        Si es None, se detecta automáticamente.
    """
    print("=" * 70)
    print("  REDIMENSIONADO DE IMÁGENES - DATASET LC25000")
    print("  Resolucion: 768x768 -> 64x64")
    print("=" * 70)

    # Obtener ruta del dataset
    ruta_dataset = obtener_ruta_dataset(ruta_base)

    # Verificar que el dataset existe y tiene la estructura correcta
    if not verificar_dataset(ruta_dataset):
        print("[ERROR] La verificación del dataset ha fallado.")
        print("        Asegurese de que el dataset LC25000 esta descargado")
        print("        y organizado correctamente en la ruta especificada.")
        sys.exit(1)

    # Definir ruta de salida (mismo nivel, con sufijo _64x64)
    ruta_salida = ruta_dataset + SUFIJO_SALIDA
    print(f"\n[INFO] Directorio de salida: {ruta_salida}")

    # Crear estructura de directorios de salida
    crear_estructura_salida(ruta_dataset, ruta_salida)

    # Procesar conjunto de Train and Validation
    estadisticas_train = procesar_conjunto(
        ruta_dataset, ruta_salida, "Train and Validation Set"
    )

    # Procesar conjunto de Test
    estadisticas_test = procesar_conjunto(
        ruta_dataset, ruta_salida, "Test Set"
    )

    # Mostrar resumen final
    mostrar_resumen_final(estadisticas_train, estadisticas_test)



