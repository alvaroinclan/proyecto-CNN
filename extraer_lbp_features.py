"""
==============================================================================
Extracción de LBP Features - Dataset LC25000
==============================================================================

Descripción:
    Este script aplica la transformación LBP (Local Binary Patterns) a las
    imágenes del dataset LC25000 previamente redimensionadas a 64x64 píxeles.

    LBP es un descriptor de textura que, para cada píxel, compara su valor
    con los de sus vecinos circundantes, generando un patrón binario que
    codifica la información de textura local. Este descriptor es:
      - Robusto ante cambios de iluminación (invariante a transformaciones
        monótonas de la escala de grises).
      - Computacionalmente eficiente.
      - Capaz de capturar micropatrones de textura relevantes para la
        clasificación histopatológica.

    Siguiendo la metodología del paper de Alsubai (2024), se aplica LBP
    sobre cada canal de la imagen RGB por separado, generando una imagen
    LBP de 3 canales que preserva la información de textura por canal.

    Parámetros LBP utilizados:
      - P (n_points): 8 puntos en el vecindario circular.
      - R (radius):   1 píxel de radio.
      - method:       'uniform' (patrones uniformes, reduce el número de
                      bins del histograma y es más robusto al ruido).

    Entrada:  lung_colon_image_set_64x64/
    Salida:   lung_colon_image_set_64x64_lbp/

==============================================================================
"""

# =============================================================================
# 1. Importar las librerías necesarias
# =============================================================================
import os
import sys
import time

# Numpy para operaciones numéricas
import numpy as np

# Librería para procesamiento de imágenes
from PIL import Image

# Skimage para el cálculo de LBP
from skimage.feature import local_binary_pattern


# =============================================================================
# 2. Configuración de parámetros
# =============================================================================

# -------------------------------------------------------------------------
# Parámetros de LBP
# P: número de puntos en el vecindario circular
# R: radio del vecindario circular (en píxeles)
# method: tipo de LBP ('uniform' reduce el número de patrones distintos)
#
# Con P=8, R=1 y method='uniform', cada píxel puede tomar valores en
# el rango [0, P+1] = [0, 9] (P*(P-1)+2 = 58+2 = 10 bins uniformes).
# -------------------------------------------------------------------------
LBP_N_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"

# -------------------------------------------------------------------------
# Clases del dataset LC25000
# -------------------------------------------------------------------------
CLASES = [
    "colon_aca",   # Adenocarcinoma de colon
    "colon_n",     # Tejido benigno de colon
    "lung_aca",    # Adenocarcinoma de pulmón
    "lung_n",      # Tejido benigno de pulmón
    "lung_scc",    # Carcinoma de células escamosas de pulmón
]

# -------------------------------------------------------------------------
# Conjuntos del dataset
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
# Nombres de directorio de entrada y salida
# -------------------------------------------------------------------------
NOMBRE_DIR_ENTRADA = "lung_colon_image_set_64x64"
NOMBRE_DIR_SALIDA = "lung_colon_image_set_64x64_lbp"


# =============================================================================
# 3. Funciones de procesamiento LBP
# =============================================================================

def obtener_rutas(ruta_base: str = None) -> tuple:
    """
    Obtiene las rutas de entrada (imágenes 64x64) y salida (imágenes LBP).

    Parameters
    ----------
    ruta_base : str, optional
        Ruta base del proyecto. Si es None, se usa el directorio del script.

    Returns
    -------
    tuple
        (ruta_entrada, ruta_salida)
    """
    if ruta_base is None:
        ruta_base = os.path.dirname(os.path.abspath(__file__))

    ruta_entrada = os.path.join(ruta_base, NOMBRE_DIR_ENTRADA)
    ruta_salida = os.path.join(ruta_base, NOMBRE_DIR_SALIDA)

    return ruta_entrada, ruta_salida


def verificar_dataset_entrada(ruta: str) -> bool:
    """
    Verifica que el dataset de entrada (64x64) existe y tiene la
    estructura esperada.

    Parameters
    ----------
    ruta : str
        Ruta al directorio del dataset de entrada.

    Returns
    -------
    bool
        True si la estructura es correcta, False en caso contrario.
    """
    print(f"[INFO] Verificando dataset de entrada en: {ruta}")

    if not os.path.isdir(ruta):
        print(f"[ERROR] El directorio no existe: {ruta}")
        print(f"        Asegúrese de haber ejecutado primero el script")
        print(f"        'redimensionar_imagenes.py' para generar las")
        print(f"        imágenes 64x64.")
        return False

    todo_ok = True
    total_imagenes = 0

    for conjunto in CONJUNTOS:
        for clase in CLASES:
            directorio = os.path.join(ruta, conjunto, clase)
            if not os.path.isdir(directorio):
                print(f"  [ERROR] Directorio faltante: {conjunto}/{clase}")
                todo_ok = False
            else:
                n_imgs = len([
                    f for f in os.listdir(directorio)
                    if os.path.splitext(f)[1].lower() in EXTENSIONES_VALIDAS
                ])
                total_imagenes += n_imgs
                print(f"  [OK] {conjunto}/{clase}: {n_imgs} imágenes")

    if todo_ok:
        print(f"  [OK] Total de imágenes encontradas: {total_imagenes}")
    return todo_ok


def crear_estructura_salida(ruta_salida: str) -> None:
    """
    Crea la estructura de directorios de salida para las imágenes LBP.

    Parameters
    ----------
    ruta_salida : str
        Ruta al directorio de salida.
    """
    print(f"\n[INFO] Creando estructura de directorios de salida...")
    print(f"       Salida: {ruta_salida}")

    for conjunto in CONJUNTOS:
        for clase in CLASES:
            directorio = os.path.join(ruta_salida, conjunto, clase)
            os.makedirs(directorio, exist_ok=True)
            print(f"       [OK] Creado: {os.path.join(conjunto, clase)}")

    print(f"[INFO] Estructura de directorios creada correctamente.\n")


def aplicar_lbp_imagen(
    ruta_imagen_entrada: str,
    ruta_imagen_salida: str,
    n_points: int = LBP_N_POINTS,
    radius: int = LBP_RADIUS,
    method: str = LBP_METHOD,
) -> bool:
    """
    Aplica LBP a una imagen RGB, procesando cada canal por separado.

    Para cada canal (R, G, B):
      1. Se extrae el canal como imagen en escala de grises.
      2. Se aplica la transformación LBP con los parámetros dados.
      3. Se normaliza el resultado al rango [0, 255] para guardar como
         imagen de 8 bits.

    Los tres canales LBP se recombinan en una imagen RGB que representa
    la textura de la imagen original.

    Parameters
    ----------
    ruta_imagen_entrada : str
        Ruta completa a la imagen de entrada (64x64, RGB).
    ruta_imagen_salida : str
        Ruta completa donde se guardará la imagen LBP resultante.
    n_points : int
        Número de puntos en el vecindario circular de LBP. Default: 8.
    radius : int
        Radio del vecindario circular en píxeles. Default: 1.
    method : str
        Método de LBP ('uniform', 'default', 'ror', 'var'). Default: 'uniform'.

    Returns
    -------
    bool
        True si se procesó correctamente, False en caso contrario.
    """
    try:
        # Cargar la imagen como array numpy
        img = Image.open(ruta_imagen_entrada)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img)

        # Preparar array de salida para la imagen LBP (misma resolución, 3 canales)
        alto, ancho = img_array.shape[:2]
        lbp_imagen = np.zeros((alto, ancho, 3), dtype=np.float64)

        # Aplicar LBP a cada canal RGB por separado
        for canal in range(3):
            lbp_canal = local_binary_pattern(
                img_array[:, :, canal],
                P=n_points,
                R=radius,
                method=method,
            )
            lbp_imagen[:, :, canal] = lbp_canal

        # Normalizar al rango [0, 255] para guardar como imagen de 8 bits.
        # Con method='uniform', los valores posibles están en [0, n_points+1].
        # Normalizamos dividiendo por el valor máximo teórico (n_points + 1).
        valor_max = n_points + 1  # Para 'uniform' con P=8, valor_max = 9
        lbp_normalizada = (lbp_imagen / valor_max * 255.0).astype(np.uint8)

        # Guardar la imagen LBP resultante
        img_lbp = Image.fromarray(lbp_normalizada, mode="RGB")
        img_lbp.save(ruta_imagen_salida, quality=95)

        return True

    except Exception as e:
        print(f"[ERROR] No se pudo procesar: {ruta_imagen_entrada}")
        print(f"        Razón: {e}")
        return False


def procesar_conjunto(
    ruta_entrada: str,
    ruta_salida: str,
    nombre_conjunto: str,
) -> dict:
    """
    Procesa todas las imágenes de un conjunto aplicando LBP.

    Parameters
    ----------
    ruta_entrada : str
        Ruta al directorio del dataset de entrada (64x64).
    ruta_salida : str
        Ruta al directorio de salida (LBP).
    nombre_conjunto : str
        Nombre del conjunto ("Train and Validation Set" o "Test Set").

    Returns
    -------
    dict
        Diccionario con estadísticas del procesamiento:
        - total: número total de imágenes encontradas
        - exitosas: número de imágenes procesadas correctamente
        - fallidas: número de imágenes que fallaron
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

            if aplicar_lbp_imagen(ruta_img_entrada, ruta_img_salida):
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


def mostrar_resumen_final(estadisticas_train: dict, estadisticas_test: dict) -> None:
    """
    Muestra un resumen final del procesamiento LBP completo.

    Parameters
    ----------
    estadisticas_train : dict
        Estadísticas del conjunto de entrenamiento/validación.
    estadisticas_test : dict
        Estadísticas del conjunto de test.
    """
    print(f"\n{'=' * 70}")
    print(f"  RESUMEN FINAL - EXTRACCIÓN DE LBP FEATURES")
    print(f"{'=' * 70}")
    print(f"  Parámetros LBP:")
    print(f"    P (n_points): {LBP_N_POINTS}")
    print(f"    R (radius):   {LBP_RADIUS}")
    print(f"    Método:       {LBP_METHOD}")
    print(f"  Entrada:        {NOMBRE_DIR_ENTRADA}/")
    print(f"  Salida:         {NOMBRE_DIR_SALIDA}/")
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

    print(f"\n[INFO] Extracción de LBP features completada exitosamente.")


# =============================================================================
# 4. Función principal
# =============================================================================
def main(ruta_base: str = None):
    """
    Función principal que ejecuta el pipeline de extracción de LBP features.

    Parameters
    ----------
    ruta_base : str, optional
        Ruta base del proyecto. Si es None, se usa el directorio del script.
    """
    print("=" * 70)
    print("  EXTRACCIÓN DE LBP FEATURES - DATASET LC25000")
    print(f"  Parámetros: P={LBP_N_POINTS}, R={LBP_RADIUS}, method='{LBP_METHOD}'")
    print("=" * 70)

    # Obtener rutas de entrada y salida
    ruta_entrada, ruta_salida = obtener_rutas(ruta_base)
    print(f"[INFO] Directorio de entrada: {ruta_entrada}")
    print(f"[INFO] Directorio de salida:  {ruta_salida}")

    # Verificar que el dataset de entrada existe
    if not verificar_dataset_entrada(ruta_entrada):
        print("[ERROR] La verificación del dataset de entrada ha fallado.")
        sys.exit(1)

    # Crear estructura de directorios de salida
    crear_estructura_salida(ruta_salida)

    # Procesar conjunto de Train and Validation
    estadisticas_train = procesar_conjunto(
        ruta_entrada, ruta_salida, "Train and Validation Set"
    )

    # Procesar conjunto de Test
    estadisticas_test = procesar_conjunto(
        ruta_entrada, ruta_salida, "Test Set"
    )

    # Mostrar resumen final
    mostrar_resumen_final(estadisticas_train, estadisticas_test)


