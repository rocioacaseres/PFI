import os
import numpy as np
import pandas as pd
import librosa
import kagglehub 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
from sktime.transformations.panel.rocket import Rocket

# ==========================================
# 1. CARGA Y FILTRADO DEL DATASET (ESC-50)
# ==========================================
print("Descarga del dataset desde Kaggle...")
dataset_root_path = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")

# Imprimo los nombres de los archivos descargados
print(f"Contenido de la carpeta descargada por Kagglehub: {os.listdir(dataset_root_path)}")

# Guardo la carpeta de metadata
metadata_path = os.path.join(dataset_root_path, 'esc50.csv')

if os.path.exists(metadata_path):
    df = pd.read_csv(metadata_path)
    print("Archivo de metadatos cargado correctamente.")
else:
    raise FileNotFoundError(f"No se encontró el archivo esc50.csv en la ruta: {metadata_path}")

# Mis clases
mis_clases = ['alarm', 'door_bell', 'cat', 'crying_baby', 'dog', 'shouting']
df_filtrado = df[df['category'].isin(mis_clases)].copy()

# Muestras con las etiquetas que busco
print(f"Total de muestras encontradas para tus 6 clases en el CSV: {len(df_filtrado)}")

# ==========================================
# 2. EXTRACCIÓN DE CARACTERÍSTICAS (MFCC)
# ==========================================
def extract_mfcc(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return mfccs
    except Exception as e:
        print(f"Error cargando {file_name}: {e}")
        return None

X_list = []
y_list = []

# Determinamos de forma inteligente el nombre real de la carpeta de audios
# En algunas estructuras de Kaggle es 'audio', en otras viene doble 'audio/audio' o 'audio_dir'
possible_audio_dir = os.path.join(dataset_root_path, 'audio')

print("Extraigo huellas digitales (MFCC)")

for index, row in df_filtrado.iterrows():
    file_path = os.path.join(possible_audio_dir, row['filename'])
    
    # Doble verificación por si en la compresión de Kaggle se movieron a una carpeta anidada
    if not os.path.exists(file_path):
        file_path = os.path.join(possible_audio_dir, 'audio', row['filename'])

    features = extract_mfcc(file_path)
    
    if features is not None:
        X_list.append(features)
        y_list.append(row['category'])

X_3D = np.array(X_list) 
y = np.array(y_list)

print(f"Forma final de los MFCCs lista para ROCKET: {X_3D.shape} (Muestras, Coeficientes, Tiempo)")

# ==========================================
# 3. DIVISIÓN DE DATOS (TRAIN / TEST)
# ==========================================
# Separo mi dataset para que haya una porción que no sea vista por el modelo mientras está siendo entrenado
#test_size: el 20% del dataset es usado solo para evaluar al modelo. El otro 80% se usa para que el modelo estudie y entrene.
#stratify: busca que haya una proporción equivalente de cada una de las clases en mi dataset de testing.
#random_state: división de datos se hace aleatoriamente para que no queden como venían en la carpeta.

X_train, X_test, y_train, y_test = train_test_split(X_3D, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 4. APLICACIÓN DE ROCKET
# ==========================================
print("\nINICIA ROCKET")
#Creacion de objeto ROCKET
#Idealmente, kernel = 10000, pero lo reduzco a la mitad para que sea mas eficiente.
rocket = Rocket(num_kernels=5000, random_state=42)
rocket.fit(X_train) #VER rocket.fit

#Se hace convolucion entre los audios originales y los 5000 kernels.
#Cada kernel escanea el audio completo y extrae dos caracteristicas: valor maximo y ppv (proporcion de valores positivos)
X_train_transform = rocket.transform(X_train)
X_test_transform = rocket.transform(X_test)
#obtengo un vector plano de 10000 numeros


# ==========================================
# 5. ENTRENAMIENTO Y VALIDACIÓN VELOZ
# ==========================================
# RidgeClassifierCV aplica una regresión lineal con validación cruzada integrada, ideal para vectores de ROCKET.
#Los alphas controlan qué tan fuerte es la penalización matemática
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_transform, y_train)

# Evaluolos resultados. Comparo training data con testing data
train_acc = classifier.score(X_train_transform, y_train)
test_acc = classifier.score(X_test_transform, y_test)

print("\n==================================================")
print(f"Precisión de Entrenamiento (Train Accuracy): {train_acc * 100:.2f}%")
print(f"Precisión de Examen (Test Accuracy): {test_acc * 100:.2f}%")
print("==================================================")