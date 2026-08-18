import threading
import queue
import time
import numpy as np

cola_procesamiento = queue.Queue(maxsize=10) #queue implementa cola fifo
cola_grabacion = queue.Queue(maxsize=50)

# Evento para apagar todos los hilos de forma segura si hay un error
sistema_corriendo = threading.Event()
sistema_corriendo.set()

def hilo_captura_audio():
    """ HILO 1. Escuchar al microfono"""
    print("Captura de audio")
        
    while sistema_corriendo.is_set():
        # Simulo que se captura un bloque de audio de medio segundo
        time.sleep(0.5) 
        bloque_audio = np.random.rand(11025) # datos crudos simulados
        
        # Se envia a la cola de procesamiento si no esta llena
        if not cola_procesamiento.full():
            cola_procesamiento.put(bloque_audio)
            
        # Grabacion en memoria
        if not cola_grabacion.full():
            cola_grabacion.put(bloque_audio)
            
    print("[Hilo 1] FIN CAPTURA")

def hilo_procesamiento_ia():
    """ HILO 2: Procesamiento y clasificacion """
    print("[Hilo 2] Procesamiento de la data con ROCKET")
    
    # Modelo cargado con joblib
    
    while sistema_corriendo.is_set():
        try:
            # Espera hasta que haya un bloque de audio disponible (timeout de 1 seg)
            bloque_audio = cola_procesamiento.get(timeout=1)
            
            # ---PIPELINE
            # ---------------------------
            
            time.sleep(0.05) 
            print("[Hilo 2] Audio procesado.")
            
            cola_procesamiento.task_done()
            
        except queue.Empty:
            continue 
            
    print("[Hilo 2] Apagando motor IA.")

def hilo_datalogger():
    """ HILO 3: CGuarda el historial en la MicroSD """
    print("[Hilo 3] Datalogger ")
    
    
    while sistema_corriendo.is_set():
        try:
            bloque_audio = cola_grabacion.get(timeout=1)
            
            # Simulamos la escritura lenta en la memoria SD
            time.sleep(0.1) 
            # print("[Hilo 3] Bloque guardado en memoria.")
            
            cola_grabacion.task_done()
            
        except queue.Empty:
            continue
            
    print("[Hilo 3] Apagando Datalogger.")

if __name__ == "__main__":
    
    # Instanciamos los hilos
    t1 = threading.Thread(target=hilo_captura_audio, name="Thread-Captura")
    t2 = threading.Thread(target=hilo_procesamiento_ia, name="Thread-IA")
    t3 = threading.Thread(target=hilo_datalogger, name="Thread-Logger")
    
    # Iniciamos los hilos en paralelo
    t1.start()
    t2.start()
    t3.start()
    
    try:
        # El programa principal se queda acá infinitamente mientras los hilos trabajan
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n Apagando el sistema de forma segura...")
        sistema_corriendo.clear()
        
        # Esperamos a que los hilos terminen ordenadamente
        t1.join()
        t2.join()
        t3.join()
        print("Sistema apagado correctamente.")