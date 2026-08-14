import numpy as np
import librosa
import joblib

# Carga de modelo y etiquetas
modelo = joblib.load('modelo_exportado.pkl')
le = joblib.load('label_encoder.pkl')

# Extraigo audio
def procesar_audio_a_mfcc(ruta_audio):
    # Cargo audio
    y, sr = librosa.load(ruta_audio, sr=22050)
    
    # Extraigo MFCC (40 coeficientes)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=512)
    
    mfcc_3d = np.expand_dims(mfcc, axis=0)
    
    return mfcc_3d


audio_prueba = "prueba_audio_pinamar.wav" # Acá luego conectarás el micrófono I2S
datos_listos = procesar_audio_a_mfcc(audio_prueba)

print("Clasificando")
prediccion = modelo.predict(datos_listos)
clase_texto = le.inverse_transform(prediccion)

print(f"Sonido detectado: {clase_texto[0]}")