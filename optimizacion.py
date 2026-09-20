
import os
NUMBA_CACHE_DIR = "/tmp/numba_cache"

os.environ["NUMBA_CACHE_DIR"] = NUMBA_CACHE_DIR

os.makedirs(
    NUMBA_CACHE_DIR,
    exist_ok=True
)

print(f"NUMBA_CACHE_DIR: {NUMBA_CACHE_DIR}")

from joblib import load
import sounddevice as sd
import soundfile as sf
import subprocess
import time
import numpy as np
import librosa
import types

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
# ADAPTACION ROCKET
# ==========================================

def adaptar_detach_rocket(modelo):
    """
    Adapta automáticamente modelos DetachRocket antiguos o actuales.
    No copia el transformer ni aumenta significativamente la RAM.
    """

    atributos_antiguos = (
        "_is_fitted",
        "_full_transformer",
        "_scaler",
        "_feature_mask",
        "_classifier",
    )

    atributos_actuales = (
        "is_fitted_",
        "classifier_",
        "pruned_transformer_",
        "pruned_scaler_",
    )

    # ==========================================
    # MODELO CON API ANTIGUA
    # ==========================================

    if all(
        hasattr(modelo, atributo)
        for atributo in atributos_antiguos
    ):
        if not modelo._is_fitted:
            raise RuntimeError(
                "El modelo DetachRocket antiguo no está entrenado."
            )

        def preparar_X_legacy(self, X):
            # Aplicación del transformer ROCKET original
            transformed_X = self._full_transformer.transform(X)

            if hasattr(transformed_X, "to_numpy"):
                transformed_X = transformed_X.to_numpy()
            else:
                transformed_X = np.asarray(transformed_X)

            if transformed_X.ndim != 2:
                raise ValueError(
                    "El transformer produjo una matriz con forma "
                    f"{transformed_X.shape}; se esperaba una matriz 2D."
                )

            # Verificación del número de características
            feature_mask = np.asarray(
                self._feature_mask,
                dtype=bool
            )

            if transformed_X.shape[1] != feature_mask.size:
                raise ValueError(
                    "El número de características generado por ROCKET "
                    "no coincide con la máscara guardada. "
                    f"ROCKET generó {transformed_X.shape[1]} y la máscara "
                    f"contiene {feature_mask.size}."
                )

            # Escalado exactamente igual al entrenamiento original
            transformed_X = self._scaler.transform(transformed_X)

            # Conservación de las características seleccionadas
            transformed_X = transformed_X[:, feature_mask]

            # Verificación del clasificador
            cantidad_esperada = getattr(
                self._classifier,
                "n_features_in_",
                transformed_X.shape[1]
            )

            if transformed_X.shape[1] != cantidad_esperada:
                raise ValueError(
                    "El clasificador espera "
                    f"{cantidad_esperada} características, pero después "
                    f"de aplicar la máscara quedaron "
                    f"{transformed_X.shape[1]}."
                )

            return transformed_X

        def predict_legacy(self, X):
            X_preparado = preparar_X_legacy(self, X)
            return self._classifier.predict(X_preparado)

        # Reemplaza predict solamente para este objeto cargado
        modelo.predict = types.MethodType(
            predict_legacy,
            modelo
        )

        print(
            "Compatibilidad completa aplicada: "
            "DetachRocket API antigua"
        )

        return modelo

    # ==========================================
    # MODELO CON API ACTUAL
    # ==========================================

    if all(
        hasattr(modelo, atributo)
        for atributo in atributos_actuales
    ):
        if not modelo.is_fitted_:
            raise RuntimeError(
                "El modelo DetachRocket actual no está entrenado."
            )

        print("Modelo DetachRocket con API actual")

        return modelo

    # ==========================================
    # ESTRUCTURA DESCONOCIDA O INCOMPLETA
    # ==========================================

    atributos_presentes = sorted(
        atributo
        for atributo in vars(modelo)
        if (
            "classifier" in atributo
            or "transformer" in atributo
            or "scaler" in atributo
            or "mask" in atributo
            or "fitted" in atributo
        )
    )

    raise RuntimeError(
        "La estructura del modelo DetachRocket no coincide con "
        "la API antigua ni con la actual.\n"
        f"Atributos relevantes encontrados: {atributos_presentes}"
    )

# ==========================================
# CARGA DE MODELO
# ==========================================

print("Carga de modelo")

t0 = time.perf_counter()

modelo = load('modelo_detach_rocket.pkl')
le = load('label_encoder.pkl')

modelo = adaptar_detach_rocket(modelo)

t1 = time.perf_counter()

print(f"Modelo cargado en: {t1 - t0:.4f} s")


# ==========================================
# FUNCION MFCC
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
    # PREDICCION
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