# Este script capta audio por 5 segundos, mide tiempos y luego clasifica

import time
import numpy as np
import librosa
from joblib import load
import sounddevice as sd


# ==========================================
# CARGA DE MODELO Y ETIQUETAS
# ==========================================

t_inicio_carga = time.perf_counter()

modelo = load('mini_rocket_modelo_exportado.pkl')
le = load('label_encoder.pkl')

t_fin_carga = time.perf_counter()

tiempo_carga = t_fin_carga - t_inicio_carga


# ==========================================
# EXTRACCIÓN DE MFCC
# ==========================================

def procesar_audio_a_mfcc(y, sr):

    # Extraigo MFCC (40 coeficientes)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40,
        hop_length=512
    )

    # Agrego dimensión para que quede:
    # (1, n_mfcc, frames)
    mfcc_3d = np.expand_dims(mfcc, axis=0)

    return mfcc_3d


# ==========================================
# CONFIGURACIÓN DE AUDIO
# ==========================================

SR = 22050
DURACION = 5


# ==========================================
# INICIO DEL PROCESO
# ==========================================

t_inicio_total = time.perf_counter()

print("Grabando...")


# ==========================================
# CAPTURA DE AUDIO
# ==========================================

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


# Pasamos de 2D a 1D
audio_grabado = audio_grabado.flatten()


# ==========================================
# EXTRACCIÓN DE MFCC
# ==========================================

t_inicio_mfcc = time.perf_counter()

datos_listos = procesar_audio_a_mfcc(
    audio_grabado,
    SR
)

t_fin_mfcc = time.perf_counter()


# ==========================================
# CLASIFICACIÓN
# ==========================================

print("Clasificando...")

t_inicio_prediccion = time.perf_counter()

prediccion = modelo.predict(datos_listos)

t_fin_prediccion = time.perf_counter()


# ==========================================
# DECODIFICACIÓN DE ETIQUETA
# ==========================================

t_inicio_etiqueta = time.perf_counter()

clase_texto = le.inverse_transform(prediccion)

t_fin_etiqueta = time.perf_counter()


t_fin_total = time.perf_counter()


# ==========================================
# RESULTADO
# ==========================================

print(f"\n🔊 Sonido detectado: {clase_texto[0]}")


# ==========================================
# TIEMPOS
# ==========================================

tiempo_grabacion = t_fin_grabacion - t_inicio_grabacion
tiempo_mfcc = t_fin_mfcc - t_inicio_mfcc
tiempo_prediccion = t_fin_prediccion - t_inicio_prediccion
tiempo_etiqueta = t_fin_etiqueta - t_inicio_etiqueta
tiempo_total = t_fin_total - t_inicio_total

tiempo_procesamiento = (
    tiempo_mfcc
    + tiempo_prediccion
    + tiempo_etiqueta
)


print("\n========== TIEMPOS ==========")

print(f"Carga del modelo:       {tiempo_carga:.4f} s")
print(f"Grabación:              {tiempo_grabacion:.4f} s")
print(f"Extracción MFCC:        {tiempo_mfcc:.4f} s")
print(f"Predicción MiniRocket:  {tiempo_prediccion:.4f} s")
print(f"Inverse transform:      {tiempo_etiqueta:.4f} s")

print("--------------------------------")

print(f"Procesamiento total:    {tiempo_procesamiento:.4f} s")
print(f"Tiempo total:           {tiempo_total:.4f} s")

print("=============================")