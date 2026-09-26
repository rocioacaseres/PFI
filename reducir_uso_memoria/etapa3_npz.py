"""PFI: exportar parámetros NPZ, validar en otro proceso y medir inferencia.

  python3 etapa3_npz.py exportar
  python3 etapa3_npz.py medir --wav /home/rcaseres/audios_grabados/perfil_ciiy7sn6.wav
  python3 etapa3_npz.py medir

Solo exportar importa joblib/sktime/sklearn. La inferencia numérica usa
NumPy/Numba; el procesamiento de audio conserva librosa/soundfile.
No sobrescribe modelos. La validación no mide ahorro de memoria.
"""
import os
import time
START = time.perf_counter()
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLBACKEND", "Agg")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
import argparse
from pathlib import Path
import sys

RTOL, ATOL = 1e-5, 1e-6
KERNEL_KEYS = ("weights", "lengths", "biases", "dilations", "paddings", "counts", "channels")


def require(ok, text):
    if not ok:
        raise ValueError(text)


def array(x):
    import numpy as np
    return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)


def compiled_transform():
    # Same loop order and PPV/max ordering as the inspected sktime implementation.
    import numpy as np
    from numba import njit

    @njit(cache=True, fastmath=True)
    def uni(X, weights, length, bias, dilation, padding):
        n = len(X)
        output_length = n + 2 * padding - (length - 1) * dilation
        positives = 0
        maximum = -np.inf
        end = n + padding - (length - 1) * dilation
        for i in range(-padding, end):
            value = bias
            index = i
            for j in range(length):
                if index > -1 and index < n:
                    value = value + weights[j] * X[index]
                index = index + dilation
            if value > maximum:
                maximum = value
            if value > 0:
                positives += 1
        return np.float32(positives / output_length), np.float32(maximum)

    @njit(cache=True, fastmath=True)
    def multi(X, weights, length, bias, dilation, padding, count, channels):
        n = X.shape[1]
        output_length = n + 2 * padding - (length - 1) * dilation
        positives = 0
        maximum = -np.inf
        end = n + padding - (length - 1) * dilation
        for i in range(-padding, end):
            value = bias
            index = i
            for j in range(length):
                if index > -1 and index < n:
                    for k in range(count):
                        value = value + weights[k, j] * X[channels[k], index]
                index = index + dilation
            if value > maximum:
                maximum = value
            if value > 0:
                positives += 1
        return np.float32(positives / output_length), np.float32(maximum)

    @njit(cache=True)
    def apply(X, weights, lengths, biases, dilations, paddings, counts, channels):
        result = np.zeros((X.shape[0], 2 * len(lengths)), dtype=np.float32)
        for i in range(X.shape[0]):
            wp = cp = 0
            for k in range(len(lengths)):
                wend = wp + counts[k] * lengths[k]
                cend = cp + counts[k]
                if counts[k] == 1:
                    ppv, maximum = uni(X[i, channels[cp]], weights[wp:wend],
                        lengths[k], biases[k], dilations[k], paddings[k])
                else:
                    w = weights[wp:wend].reshape((counts[k], lengths[k]))
                    ppv, maximum = multi(X[i], w, lengths[k], biases[k],
                        dilations[k], paddings[k], counts[k], channels[cp:cend])
                result[i, 2*k] = ppv
                result[i, 2*k+1] = maximum
                wp, cp = wend, cend
        return result
    return apply


class NumericModel:
    def __init__(self, path):
        import numpy as np
        with np.load(path, allow_pickle=False) as f:
            self.p = {k: f[k] for k in f.files}
        p = self.p
        require(str(p["format"]) == "PFI-NPZ-v1", "Formato NPZ no compatible")
        require(tuple(p["audio_config"]) == (22050, 110250, 40, 512), "Configuración de audio no compatible")
        n = len(p["lengths"])
        for k in KERNEL_KEYS:
            require(p[k].ndim == 1, f"Arreglo inválido: {k}")
        for k in ("biases", "dilations", "paddings", "counts"):
            require(len(p[k]) == n, f"Longitud inválida: {k}")
        require(n > 0 and (p["lengths"] > 0).all() and (p["counts"] > 0).all(), "Kernels vacíos")
        require((p["dilations"] > 0).all() and (p["paddings"] >= 0).all(), "Dilatación/padding inválido")
        require(int((p["lengths"].astype(np.int64)*p["counts"]).sum()) == len(p["weights"]), "Pesos incompatibles")
        require(int(p["counts"].sum()) == len(p["channels"]), "Canales incompatibles")
        require(((p["channels"] >= 0) & (p["channels"] < 40)).all(), "Canales fuera de rango")
        pos = p["positions"]
        require(pos.ndim == 1 and pos.dtype.kind in "iu" and pos.size > 0, "Selección inválida")
        require(((pos >= 0) & (pos < 2*n)).all(), "Posiciones fuera de rango")
        require(p["coef"].ndim == 2 and p["coef"].shape[1] == len(pos), "Coeficientes incompatibles")
        require(p["intercept"].shape == (p["coef"].shape[0],), "Intercepto incompatible")
        require(p["classes"].ndim == 1 and p["labels"].shape == p["classes"].shape, "Clases incompatibles")
        require(p["coef"].shape[0] == len(p["classes"]) or
                (len(p["classes"]) == 2 and p["coef"].shape[0] == 1), "Clasificador no compatible")
        require(p["mean"].shape == pos.shape and p["scale"].shape == pos.shape, "Escalador incompatible")
        require((p["scale"] > 0).all(), "Escala no positiva")
        for k in ("weights", "biases", "mean", "scale", "coef", "intercept"):
            require(np.isfinite(p[k]).all(), f"Valores no finitos en {k}")
        self.apply = compiled_transform()

    def raw(self, X):
        import numpy as np
        require(X.ndim == 3 and X.shape[1] == 40 and np.isfinite(X).all(), "MFCC incompatibles")
        p = self.p
        require((X.shape[2]+2*p["paddings"]-(p["lengths"]-1)*p["dilations"] > 0).all(), "Serie demasiado corta")
        if bool(p["normalise"]):
            X = (X-X.mean(axis=-1, keepdims=True))/(X.std(axis=-1, keepdims=True)+1e-8)
        return self.apply(X.astype(np.float32), *(p[k] for k in KERNEL_KEYS))[:, p["positions"]]

    def scale(self, raw):
        # Preserve sklearn's float32 in-place rounding at each operation.
        z = raw.copy()
        if bool(self.p["with_mean"]):
            z -= self.p["mean"]
        if bool(self.p["with_std"]):
            z /= self.p["scale"]
        return z

    def classify(self, z):
        import numpy as np
        p = self.p
        scores = z @ p["coef"].T + p["intercept"]
        ids = (scores[:, 0] > 0).astype(int) if scores.shape[1] == 1 else scores.argmax(axis=1)
        return scores, p["classes"][ids], p["labels"][ids]

    def predict(self, X):
        return self.classify(self.scale(self.raw(X)))


def mfcc(y):
    import librosa
    return librosa.feature.mfcc(y=y, sr=22050, n_mfcc=40, hop_length=512)[None, ...]


def read_wav(path):
    import numpy as np
    import soundfile as sf
    import librosa
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
    require(len(y) == 110250 and np.isfinite(y).all(), f"{path}: se requiere audio finito de 5 s")
    return y


def dependencies():
    roots = ("sktime", "sklearn", "pandas", "joblib", "detach_rocket", "scipy", "numba", "librosa")
    loaded = [r for r in roots if r in sys.modules]
    print("Dependencias presentes:", ", ".join(loaded), flush=True)
    return loaded


def validate(path, references):
    import numpy as np
    model = NumericModel(path)
    print("Primera llamada: puede compilar kernels Numba; esperar...", flush=True)
    with np.load(references, allow_pickle=False) as refs:
        for i, X in enumerate(refs["X"]):
            raw = model.raw(X[None, ...]); z = model.scale(raw)
            scores, pred, labels = model.classify(z)
            for name, actual in (("raw", raw), ("z", z), ("scores", scores)):
                expected = refs[name][i:i+1]
                np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False,
                                           err_msg=f"{refs['names'][i]}: {name}")
            np.testing.assert_array_equal(pred, refs["pred"][i:i+1])
            np.testing.assert_array_equal(labels, refs["labels"][i:i+1])
            error = np.max(np.abs(scores-refs["scores"][i:i+1]))
            print(f"OK NPZ: {refs['names'][i]} | error puntuaciones={error:.8g}", flush=True)
    loaded = dependencies()
    require(not set(loaded) & {"sktime", "sklearn", "pandas", "joblib", "detach_rocket"},
            "La validación numérica importó dependencias excluidas")
    print("VALIDACIÓN NPZ EN PROCESO INDEPENDIENTE: OK", flush=True)


def export(args):
    import numpy as np
    from joblib import load
    import tempfile
    import subprocess
    import shutil
    import importlib.metadata as metadata
    require(not args.npz.exists(), "El destino ya existe; usar otro --npz")
    paths = sorted(args.audios.glob("*.wav"))[:10]
    require(bool(paths), "No hay archivos WAV para validar")
    print("Cargando el PKL reducido validado...", flush=True)
    p = load(args.pkl)
    require(p.get("formato") == "PFI-reducido-v1", "Se requiere el PKL reducido de etapa 1")
    for name, version in p["versions"].items():
        require(metadata.version(name) == version, f"Versión distinta de {name}: se requiere {version}")
    tr, sc, clf = p["transformer"], p["scaler"], p["classifier"]
    require(len(tr.kernels) == 7 and tr.n_columns == 40, "Transformer no compatible")
    require((p["sr"],p["muestras"],p["n_mfcc"],p["hop_length"]) == (22050,110250,40,512), "Audio no compatible")
    data = {k: np.asarray(v).copy() for k, v in zip(KERNEL_KEYS, tr.kernels)}
    n = len(p["posiciones"])
    classes = np.asarray(clf.classes_)
    require(classes.dtype.kind in "iu", "Se requieren clases codificadas como enteros")
    data.update(format=np.array("PFI-NPZ-v1"), normalise=np.array(tr.normalise),
        positions=np.asarray(p["posiciones"]).copy(),
        with_mean=np.array(sc.with_mean), with_std=np.array(sc.with_std),
        mean=sc.mean_.copy() if sc.with_mean else np.zeros(n),
        scale=sc.scale_.copy() if sc.with_std else np.ones(n),
        coef=clf.coef_.copy(), intercept=np.atleast_1d(clf.intercept_).copy(),
        classes=classes.copy(), labels=p["encoder"].inverse_transform(classes).astype(str),
        audio_config=np.array([22050,110250,40,512]),
        packages=np.array(list(p["versions"])), versions=np.array(list(p["versions"].values())))
    require(all(v.dtype.kind != "O" for v in data.values()), "No se permiten objetos en el NPZ")
    refs = {k: [] for k in ("X", "raw", "z", "scores", "pred", "labels", "names")}
    def reference(name, y):
        X = mfcc(y)
        raw = array(tr.transform(X))[:, p["posiciones"]]
        z = sc.transform(raw.copy())
        pred = clf.predict(z)
        score = clf.decision_function(z).reshape(1, -1)
        for key, val in (("X",X[0]),("raw",raw[0]),("z",z[0]),("scores",score[0]),
                         ("pred",pred[0]),("labels",str(p["encoder"].inverse_transform(pred)[0])),("names",name)):
            refs[key].append(val)
        print("Referencia:", name, flush=True)
    for path in paths:
        reference(path.name, read_wav(path))
    reference("silencio", np.zeros(110250, dtype=np.float32))
    reference("ruido sintético", np.random.default_rng(42).normal(0,.1,110250).astype(np.float32))
    with tempfile.TemporaryDirectory(prefix="pfi_npz_") as td:
        candidate, references = Path(td)/"modelo.npz", Path(td)/"referencias.npz"
        np.savez(candidate, **data)
        np.savez(references, **{k: np.asarray(v) for k,v in refs.items()})
        subprocess.run([sys.executable,str(Path(__file__).resolve()),"_validar",str(candidate),str(references)],check=True)
        with candidate.open("rb") as source, args.npz.open("xb") as dest:
            shutil.copyfileobj(source,dest)
    print("EXPORTACIÓN NPZ VALIDADA:", args.npz.resolve())
    print("Tamaño:", args.npz.stat().st_size, "bytes")
    print("Los modelos PKL se conservaron. No medir RAM del exportador.")


def memory(label):
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/smaps_rollup") as f:
                values={s.split(':')[0]:int(s.split()[1])/1024 for s in f if s.startswith(("Rss:","Swap:"))}
            print(f"MEM {label}: RSS={values['Rss']:.2f} MiB; pico={peak:.2f} MiB; swap={values['Swap']:.2f} MiB",flush=True)
        except OSError:
            print(f"MEM {label}: pico={peak:.2f} MiB",flush=True)


def measure(args):
    import numpy as np
    import soundfile as sf
    import librosa
    if not args.wav:
        import sounddevice as sd
    memory("importaciones audio")
    t=time.perf_counter(); model=NumericModel(args.npz); load_time=time.perf_counter()-t
    memory("carga NPZ + importar Numba")
    t=time.perf_counter(); dummy=mfcc(np.zeros(110250,dtype=np.float32)); cold_mfcc=time.perf_counter()-t
    memory("primer MFCC")
    print("Precarga numérica (primera ejecución puede compilar)...",flush=True)
    t=time.perf_counter();model.predict(dummy);warm=time.perf_counter()-t
    del dummy
    memory("primera predicción")
    if args.wav:
        t=time.perf_counter();y=read_wav(args.wav);acquisition=time.perf_counter()-t
        saved=0
    else:
        import tempfile
        print("Grabando en 3 segundos...",flush=True);time.sleep(3)
        t=time.perf_counter();y=sd.rec(110250,samplerate=22050,channels=1,dtype="float32");sd.wait()
        acquisition=time.perf_counter()-t;y=y[:,0]
        t=time.perf_counter();args.audios.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=args.audios,prefix="npz_",suffix=".wav",delete=False) as f:
            audio_path=f.name
        sf.write(audio_path,y,22050);saved=time.perf_counter()-t
        print("Audio guardado:",audio_path)
    t=time.perf_counter();X=mfcc(y);mfcc_time=time.perf_counter()-t
    t=time.perf_counter();scores,pred,labels=model.predict(X);prediction=time.perf_counter()-t
    memory("audio real procesado")
    print("Sonido detectado:",labels[0])
    for label,value in (("Carga NPZ e inicialización Numba",load_time),("Primer MFCC",cold_mfcc),
        ("Precarga numérica",warm),("Lectura WAV" if args.wav else "Grabación",acquisition),
        ("Guardado",saved),("MFCC",mfcc_time),("Predicción incl. etiqueta",prediction),
        ("Procesamiento",mfcc_time+prediction),("Total desde inicio Python",time.perf_counter()-START)):
        print(f"{label}: {value:.4f} s")
    dependencies()
    print("MEDICIÓN NPZ COMPLETADA")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modo",choices=("exportar","medir","_validar"))
    parser.add_argument("files",nargs="*")
    parser.add_argument("--pkl",type=Path,default=Path("modelo_detach_reducido_v1.pkl"))
    parser.add_argument("--npz",type=Path,default=Path("modelo_detach_numerico_v1.npz"))
    parser.add_argument("--audios",type=Path,default=Path("/home/rcaseres/audios_grabados"))
    parser.add_argument("--wav",type=Path)
    args=parser.parse_args()
    if args.modo=="exportar": export(args)
    elif args.modo=="medir": measure(args)
    else:
        require(len(args.files)==2,"Faltan modelo y referencias")
        validate(*args.files)


if __name__=="__main__":
    main()
