# Este script capta audio por 5 segundos y luego clasifica

import numpy as np
import librosa
from joblib import load
import sounddevice as sd

# Carga de modelo y etiquetas
modelo = load('mini_rocket_modelo_exportado.pkl')
le = load('label_encoder.pkl')

# Extraigo audio
def procesar_audio_a_mfcc(ruta_audio):
    # Extraigo MFCC (40 coeficientes)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=512)
    
    mfcc_3d = np.expand_dims(mfcc, axis=0)
    
    return mfcc_3d


# ==========================================
# Captura de audio en vivo
# ==========================================
SR = 22050         # Frecuencia de muestreo 
DURACION = 5       # Segundos de grabacion

print(f"Grabando")

# Captura de audio
# channels=1 para que sea Mono, dtype='float32' es el formato matematico que le gusta a Librosa
audio_grabado = sd.rec(int(DURACION * SR), samplerate=SR, channels=1, dtype='float32')

# Se pausa el script hasta que pasen los 5 segundos de grabación
sd.wait() 
print("Grabación finalizada")

# sounddevice nos devuelve una matriz 2D (muestras, canales). 
# La aplanamos a 1D con flatten() para que librosa la pueda leer bien.
audio_grabado = audio_grabado.flatten()

# ==========================================
# EXTRACCIÓN Y CLASIFICACIÓN
# ==========================================
datos_listos = procesar_audio_a_mfcc(audio_grabado, SR)

print("Clasificando")
prediccion = modelo.predict(datos_listos)
clase_texto = le.inverse_transform(prediccion)

print(f"🔊 Sonido detectado: {clase_texto[0]}")