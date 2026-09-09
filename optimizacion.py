import time
import numpy as np
import librosa
import os
from joblib import load
import sounddevice as sd
import soundfile as sf



# ==========================================
# CONFIGURACIÓN
# ==========================================

SR = 22050
DURACION = 5

#Directorio donde dejo los audios grabados

DIRECTORIO_AUDIOS = "/home/rcaseres/audios_grabados"

os.makedirs(
    DIRECTORIO_AUDIOS,
    exist_ok=True
)


# ==========================================
# CARGA DE MODELO
# ==========================================

print("Carga de modelo")

t0 = time.perf_counter()

modelo = load('modelo_exportado.pkl')
le = load('label_encoder.pkl')

t1 = time.perf_counter()

print(f"Modelo cargado en: {t1 - t0:.4f} s")


# ==========================================
# FUNCIÓN MFCC
# ==========================================

def procesar_audio_a_mfcc(y, sr): 

    #Extraigo MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40,
        hop_length=512
    )

    mfcc_3d = np.expand_dims(mfcc, axis=0)

    return mfcc_3d


# ==========================================
# PRE-CARGO MODELO
# ==========================================

print("\nPrecargo modelo")

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

modelo.predict(datos_dummy)

t1 = time.perf_counter()

print(f"Precarga de modelo terminado en: {t1 - t0:.4f} s")


# ==========================================
# LOOP PRINCIPAL
# ==========================================
contador_audio = 1
while True:

    opcion = input(
    "\nPresioná ENTER para grabar o escribí q para terminar: "
    )
    if opcion.lower() == "q":
        break

    print("Grabando en 3s")
    time.sleep(3)
    print("Grabando")
    t_inicio_grabacion = time.perf_counter()

    audio_grabado = sd.rec(
        int(DURACION * SR),
        samplerate=SR,
        channels=1,
        dtype='float32'
    )
    #5 segundos × 22050 muestras/segundos = 110250 muestras
    sd.wait()

    t_fin_grabacion = time.perf_counter()

    print("Grabación finalizada")

    audio_grabado = audio_grabado.flatten() #recibo 2 dimensiones, con flatten queda de 1 dimension

    # GUARDADO DE AUDIO
    # --------------------------------------

    nombre_audio = f"audio_{contador_audio:04d}.wav"

    ruta_audio = os.path.join(
        DIRECTORIO_AUDIOS,
        nombre_audio
    )

    sf.write(
        ruta_audio,
        audio_grabado,
        SR
    )

    print(
        f"Audio guardado en: "
        f"{ruta_audio}"
    )

    contador_audio += 1

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
        f"\nSonido detectado: "
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