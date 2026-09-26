"""Diagnóstico Linux del pipeline PFI reducido. No modifica modelos.

python3 etapa2_memoria_por_etapas.py
python3 etapa2_memoria_por_etapas.py --wav audios_grabados/audio_0001.wav

RSS actual: /proc/self/smaps_rollup (fallback: smaps, luego status).
Pico acumulado: resource.getrusage(RUSAGE_SELF).ru_maxrss, KiB en Linux.
Las diferencias entre checkpoints NO son memoria exclusiva de una biblioteca.
Los tiempos están instrumentados: no sustituyen al benchmark sin instrumentar.
"""
import os
import sys
import time
import resource
import argparse
from contextlib import contextmanager
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLBACKEND", "Agg")


def memoria():
    """Lectura puntual, sin hilo de muestreo ni psutil."""
    for path, rss_key, swap_key in (
        ("/proc/self/smaps_rollup", "Rss:", "Swap:"),
        ("/proc/self/smaps", "Rss:", "Swap:"),
        ("/proc/self/status", "VmRSS:", "VmSwap:"),
    ):
        try:
            rss = swap = 0
            found = False
            with open(path, encoding="ascii") as f:
                for line in f:
                    if line.startswith(rss_key):
                        rss += int(line.split()[1]); found = True
                    elif line.startswith(swap_key):
                        swap += int(line.split()[1])
            if found:
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                return rss / 1024, peak / 1024, swap / 1024, path
        except OSError:
            continue
    raise RuntimeError("No se pudo leer RAM desde /proc/self; ejecutar en Linux.")


class Monitor:
    def __init__(self):
        self.rows = []
        self.previous = None
        self.modules = set(sys.modules)

    def registrar(self, nombre, segundos=0):
        rss, peak, swap, source = memoria()
        delta = 0 if self.previous is None else rss - self.previous
        self.rows.append((nombre, segundos, rss, delta, peak, swap))
        self.previous = rss
        print(f"{nombre:<34} {segundos:>9.3f} {rss:>10.2f} {delta:>10.2f} "
              f"{peak:>10.2f} {swap:>9.2f}", flush=True)
        added = set(sys.modules) - self.modules
        watched = ("numpy", "scipy", "sklearn", "sktime", "numba", "llvmlite",
                   "librosa", "pandas", "matplotlib", "detach_rocket")
        roots = [name for name in watched if name in added]
        if roots:
            print("  Módulos principales recién importados: " + ", ".join(roots), flush=True)
        self.modules = set(sys.modules)
        if len(self.rows) == 1:
            print(f"  Fuente RSS: {source}; pico: getrusage (métricas independientes).", flush=True)

    @contextmanager
    def etapa(self, nombre):
        print(f"\nEjecutando: {nombre}", flush=True)
        inicio = time.perf_counter()
        try:
            yield
        except Exception:
            self.registrar(nombre + " [ERROR]", time.perf_counter() - inicio)
            raise
        else:
            self.registrar(nombre, time.perf_counter() - inicio)

    def resumen(self):
        print("\n===== TABLA FINAL =====")
        print("Etapa;Tiempo_s;RSS_MiB;Delta_RSS_MiB;Pico_MiB;Swap_MiB")
        for name, seconds, rss, delta, peak, swap in self.rows:
            print(f"{name};{seconds:.4f};{rss:.3f};{delta:.3f};{peak:.3f};{swap:.3f}")
        print("\nDelta RSS: diferencia neta entre puntos, no asignación exclusiva.")
        print("Pico: máximo histórico; no baja al liberar objetos.")
        print("Si una etapa no supera el pico previo, no conocemos su máximo local.")
        print("Lecturas, impresiones y objetos del monitor agregan una pequeña sobrecarga.")


def main():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Este diagnóstico utiliza las métricas de Linux de la Raspberry.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", type=Path, default=Path("modelo_detach_reducido_v1.pkl"))
    parser.add_argument("--wav", type=Path, help="Usar WAV de 5 s en lugar de grabar")
    parser.add_argument("--audios", type=Path, default=Path("/home/rcaseres/audios_grabados"))
    args = parser.parse_args()
    if not args.modelo.is_file():
        raise FileNotFoundError(f"No se encuentra el modelo: {args.modelo.resolve()}")
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
    print("===== MEMORIA POR ETAPAS: MODELO REDUCIDO =====")
    print("Modo:", "WAV guardado" if args.wav else "micrófono")
    print(f"{'Etapa':<34} {'Tiempo s':>9} {'RSS MiB':>10} {'Delta MiB':>10} "
          f"{'Pico MiB':>10} {'Swap MiB':>9}")
    mon = Monitor()
    mon.registrar("Base Python + monitor")
    with mon.etapa("Importar numpy"):
        import numpy as np
    with mon.etapa("Importar sounddevice"):
        import sounddevice as sd
    with mon.etapa("Importar soundfile"):
        import soundfile as sf
    with mon.etapa("Importar librosa"):
        import librosa
    with mon.etapa("Importar joblib"):
        from joblib import load
    with mon.etapa("Cargar modelo reducido"):
        paquete = load(args.modelo)
        if paquete.get("formato") != "PFI-reducido-v1":
            raise ValueError("El archivo no tiene el formato PFI-reducido-v1.")
    with mon.etapa("Verificar configuración"):
        import importlib.metadata as metadata
        for name, expected in paquete["versions"].items():
            if metadata.version(name) != expected:
                raise ValueError(f"{name}: se requiere versión de exportación {expected}")
        for name, expected in (("sr", 22050), ("muestras", 110250), ("n_mfcc", 40), ("hop_length", 512)):
            if paquete[name] != expected:
                raise ValueError(f"Configuración incompatible: {name}")
        tr, sc, clf = paquete["transformer"], paquete["scaler"], paquete["classifier"]
        posiciones, encoder = paquete["posiciones"], paquete["encoder"]
    def mfcc(y):
        return librosa.feature.mfcc(y=y, sr=22050, n_mfcc=40, hop_length=512)[None, ...]
    def transformar(X):
        values = tr.transform(X)
        return values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
    with mon.etapa("Crear silencio de precarga"):
        silencio = np.zeros(110250, dtype=np.float32)
    with mon.etapa("Primera extracción MFCC"):
        dummy = mfcc(silencio)
    with mon.etapa("Primera transformación ROCKET"):
        z = transformar(dummy)
    with mon.etapa("Primera selección y escalado"):
        selected = sc.transform(z[:, posiciones])
    with mon.etapa("Primera clasificación Ridge"):
        clf.predict(selected)
    with mon.etapa("Liberar datos de precarga"):
        del silencio, dummy, z, selected
    if args.wav:
        with mon.etapa("Leer audio WAV"):
            audio, sr = sf.read(args.wav, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if sr != 22050:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    else:
        with mon.etapa("Cuenta regresiva 3 segundos"):
            print("Prepará el sonido; se grabará al terminar la cuenta regresiva.", flush=True)
            time.sleep(3)
        with mon.etapa("Grabación 5 segundos"):
            audio = sd.rec(110250, samplerate=22050, channels=1, dtype="float32")
            sd.wait()
            audio = audio[:, 0]
        with mon.etapa("Guardar WAV nuevo"):
            import tempfile
            args.audios.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=args.audios, prefix="perfil_", suffix=".wav", delete=False) as f:
                destino = f.name
            sf.write(destino, audio, 22050)
        print("Audio guardado:", destino, flush=True)
    if len(audio) != 110250 or not np.isfinite(audio).all():
        raise ValueError("Se requiere audio finito de 5 segundos (110250 muestras a 22050 Hz).")
    with mon.etapa("MFCC audio real"):
        X = mfcc(audio)
    with mon.etapa("ROCKET audio real"):
        z = transformar(X)
    with mon.etapa("Selección y escalado real"):
        selected = sc.transform(z[:, posiciones])
    with mon.etapa("Ridge audio real"):
        pred = clf.predict(selected)
    with mon.etapa("Decodificar etiqueta"):
        label = encoder.inverse_transform(pred)[0]
    print("Sonido detectado:", label, flush=True)
    with mon.etapa("Liberar matrices temporales"):
        del audio, X, z, selected
    print("DetachRocket importado:", any(k == "detach_rocket" or k.startswith("detach_rocket.") for k in sys.modules))
    mon.resumen()
    print("DIAGNÓSTICO COMPLETADO")


if __name__ == "__main__":
    main()
