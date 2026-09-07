import time
import numpy as np
import librosa
from joblib import load
import sounddevice as sd


# ==========================================
# CONFIGURACIÓN
# ==========================================

SR = 22050
DURACION = 5


# ==========================================
# CARGA DE MODELO
# ==========================================

print("Cargando modelo...")

t0 = time.perf_counter()

modelo = load('mini_rocket_modelo_exportado.pkl')
le = load('label_encoder.pkl')

t1 = time.perf_counter()

print(f"Modelo cargado en: {t1 - t0:.4f} s")


# ==========================================
# FUNCIÓN MFCC
# ==========================================

def procesar_audio_a_mfcc(y, sr):

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40,
        hop_length=512
    )

    mfcc_3d = np.expand_dims(mfcc, axis=0)

    return mfcc_3d


# ==========================================
# WARM-UP
# ==========================================

print("\nRealizando warm-up...")

# Genero 5 segundos de silencio
audio_dummy = np.zeros(
    int(SR * DURACION),
    dtype=np.float32
)

# MFCC dummy
datos_dummy = procesar_audio_a_mfcc(
    audio_dummy,
    SR
)

t0 = time.perf_counter()

_ = modelo.predict(datos_dummy)

t1 = time.perf_counter()

print(f"Warm-up terminado en: {t1 - t0:.4f} s")


# ==========================================
# LOOP PRINCIPAL
# ==========================================

while True:

    input("\nPresioná ENTER para grabar...")

    # --------------------------------------
    # GRABACIÓN
    # --------------------------------------

    print("Grabando...")

    t_inicio_grabacion = time.perf_counter()

    audio_grabado = sd.rec(
        int(DURACION * SR),
        samplerate=SR,
        channels=1,
        dtype='float32'
    )

    sd.wait()

    t_fin_grabacion = time.perf_counter()

    print("Grabación finalizada")

    audio_grabado = audio_grabado.flatten()


    # --------------------------------------
    # MFCC
    # --------------------------------------

    t_inicio_mfcc = time.perf_counter()

    datos_listos = procesar_audio_a_mfcc(
        audio_grabado,
        SR
    )

    t_fin_mfcc = time.perf_counter()


    # --------------------------------------
    # PREDICCIÓN
    # --------------------------------------

    print("Clasificando...")

    t_inicio_prediccion = time.perf_counter()

    prediccion = modelo.predict(datos_listos)

    t_fin_prediccion = time.perf_counter()


    # --------------------------------------
    # ETIQUETA
    # --------------------------------------

    t_inicio_etiqueta = time.perf_counter()

    clase_texto = le.inverse_transform(prediccion)

    t_fin_etiqueta = time.perf_counter()


    # ======================================
    # RESULTADOS
    # ======================================

    tiempo_grabacion = (
        t_fin_grabacion
        - t_inicio_grabacion
    )

    tiempo_mfcc = (
        t_fin_mfcc
        - t_inicio_mfcc
    )

    tiempo_prediccion = (
        t_fin_prediccion
        - t_inicio_prediccion
    )

    tiempo_etiqueta = (
        t_fin_etiqueta
        - t_inicio_etiqueta
    )

    tiempo_procesamiento = (
        tiempo_mfcc
        + tiempo_prediccion
        + tiempo_etiqueta
    )


    print(
        f"\n🔊 Sonido detectado: "
        f"{clase_texto[0]}"
    )

    print("\n========== TIEMPOS ==========")

    print(
        f"Grabación:             "
        f"{tiempo_grabacion:.4f} s"
    )

    print(
        f"Extracción MFCC:       "
        f"{tiempo_mfcc:.4f} s"
    )

    print(
        f"Predicción MiniRocket: "
        f"{tiempo_prediccion:.4f} s"
    )

    print(
        f"Inverse transform:     "
        f"{tiempo_etiqueta:.4f} s"
    )

    print("--------------------------------")

    print(
        f"Procesamiento total:   "
        f"{tiempo_procesamiento:.4f} s"
    )

    print("=============================")