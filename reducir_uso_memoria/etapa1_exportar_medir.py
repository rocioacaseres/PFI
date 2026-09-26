"""PFI: exportación validada y medición independiente de DetachRocket reducido.

En la Raspberry, desde la carpeta de los modelos:
  python3 etapa1_exportar_medir.py exportar
  python3 etapa1_exportar_medir.py medir
  python3 etapa1_exportar_medir.py medir --original

No reinstala paquetes ni sobrescribe modelos. Mantiene sktime y sklearn;
la inferencia reducida no carga el objeto DetachRocket original.
"""
import os
import time

INICIO = time.perf_counter()
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLBACKEND", "Agg")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import argparse
import copy
import importlib.metadata as metadata
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SR, MUESTRAS = 22050, 110250
RTOL, ATOL = 1e-5, 1e-6


def require(condicion, mensaje):
    if not condicion:
        raise ValueError(mensaje)


def array(x):
    import numpy as np
    return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)


def mfcc(audio):
    import librosa
    return librosa.feature.mfcc(
        y=audio, sr=SR, n_mfcc=40, hop_length=512
    )[None, ...]


def features(paquete, X):
    z = array(paquete["transformer"].transform(X))
    return paquete["scaler"].transform(z[:, paquete["posiciones"]])


def features_original(modelo, X):
    z = array(modelo._full_transformer.transform(X))
    return modelo._scaler.transform(z)[:, modelo._feature_mask]


def reducir(modelo, encoder):
    import numpy as np
    original = modelo._full_transformer
    mask = np.asarray(modelo._feature_mask)
    require(mask.ndim == 1 and mask.dtype == np.bool_, "Máscara no booleana 1D")
    require(len(original.kernels) == 7, "Estructura de kernels desconocida")
    w, lengths, bias, dilation, padding, counts, channels = original.kernels
    require(mask.size == 2 * len(lengths), "Máscara y kernels incompatibles")
    idx = np.flatnonzero(mask)
    chosen = np.unique(idx // 2)
    require(idx.size > 0, "Selección vacía")
    require(idx.size == modelo._classifier.n_features_in_, "Clasificador incompatible")
    wp = np.r_[0, np.cumsum(lengths.astype(np.int64) * counts.astype(np.int64))]
    cp = np.r_[0, np.cumsum(counts.astype(np.int64))]
    require(wp[-1] == w.size and cp[-1] == channels.size, "Offsets incompatibles")
    tr = copy.copy(original)
    tr.kernels = (
        np.concatenate([w[wp[k]:wp[k+1]] for k in chosen]),
        lengths[chosen].copy(), bias[chosen].copy(), dilation[chosen].copy(),
        padding[chosen].copy(), counts[chosen].copy(),
        np.concatenate([channels[cp[k]:cp[k+1]] for k in chosen]),
    )
    tr.num_kernels = len(chosen)
    sc = copy.copy(modelo._scaler)
    for key in ("mean_", "var_", "scale_"):
        value = getattr(sc, key, None)
        if value is not None:
            setattr(sc, key, value[idx].copy())
    sc.n_features_in_ = len(idx)
    seen = getattr(sc, "n_samples_seen_", None)
    if isinstance(seen, np.ndarray) and seen.ndim == 1:
        sc.n_samples_seen_ = seen[idx].copy()
    # Feature names no longer describe the reduced numeric matrix.
    if hasattr(sc, "feature_names_in_"):
        del sc.feature_names_in_
    versions = {}
    for name in ("numpy", "sktime", "scikit-learn", "numba", "joblib", "librosa"):
        versions[name] = metadata.version(name)
    return {
        "formato": "PFI-reducido-v1",
        "transformer": tr, "scaler": sc,
        "posiciones": 2 * np.searchsorted(chosen, idx // 2) + idx % 2,
        "classifier": modelo._classifier, "encoder": encoder,
        "sr": SR, "muestras": MUESTRAS, "n_mfcc": 40, "hop_length": 512,
        "versions": versions, "kernels_originales": len(lengths),
    }


def cargar_reducido(path):
    from joblib import load
    p = load(path)
    require(p.get("formato") == "PFI-reducido-v1", "Formato de modelo desconocido")
    require(p["sr"] == SR and p["muestras"] == MUESTRAS, "Configuración incompatible")
    for name, version in p["versions"].items():
        require(metadata.version(name) == version,
                f"Versión diferente de {name}: se exportó con {version}")
    return p


def validar_archivo(model_path, refs_path):
    import numpy as np
    p = cargar_reducido(model_path)
    with np.load(refs_path, allow_pickle=False) as refs:
        for i, X in enumerate(refs["X"]):
            z = features(p, X[None, ...])
            scores = p["classifier"].decision_function(z)
            pred = p["classifier"].predict(z)
            np.testing.assert_allclose(z, refs["z"][i:i+1], rtol=RTOL, atol=ATOL, equal_nan=False)
            np.testing.assert_allclose(scores, refs["scores"][i:i+1], rtol=RTOL, atol=ATOL, equal_nan=False)
            np.testing.assert_array_equal(pred, refs["pred"][i:i+1])
            text = p["encoder"].inverse_transform(pred).astype(str)
            np.testing.assert_array_equal(text, refs["texto"][i:i+1])
            error = np.max(np.abs(scores - refs["scores"][i:i+1]))
            print(f"OK recarga: {refs['nombres'][i]} | error={error:.8g}", flush=True)
    require(not any(k == "detach_rocket" or k.startswith("detach_rocket.") for k in sys.modules),
            "La recarga aún importa DetachRocket; revisar referencias")
    print("VALIDACIÓN EN PROCESO NUEVO: OK; DetachRocket no importado.", flush=True)


def exportar(args):
    import numpy as np
    import soundfile as sf
    import librosa
    from joblib import load, dump
    out = args.salida.resolve()
    require(not out.exists(), f"Ya existe {out}; elegí otro nombre con --salida")
    paths = sorted(args.audios.glob("*.wav"))[:10]
    require(bool(paths), f"No hay WAV en {args.audios}")
    print("Cargando original y encoder...", flush=True)
    model = load(args.modelo)
    encoder = load(args.encoder)
    p = reducir(model, encoder)
    print(f"Kernels: {p['kernels_originales']} -> {p['transformer'].num_kernels}")
    print(f"Características finales: {len(p['posiciones'])}")
    Xs, zs, scores, predictions, texts, names = [], [], [], [], [], []

    def referencia(nombre, audio):
        X = mfcc(audio)
        z = features_original(model, X)
        pred = model._classifier.predict(z)
        Xs.append(X[0]); zs.append(z[0])
        scores.append(model._classifier.decision_function(z)[0])
        predictions.append(pred[0])
        texts.append(str(encoder.inverse_transform(pred)[0]))
        names.append(nombre)
        print(f"Referencia: {nombre}", flush=True)

    for path in paths:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        require(len(audio) == MUESTRAS, f"{path.name}: se requieren 5 segundos")
        require(np.isfinite(audio).all(), f"{path.name}: audio no finito")
        referencia(path.name, audio)
    referencia("silencio", np.zeros(MUESTRAS, dtype=np.float32))
    referencia("ruido sintético", np.random.default_rng(42).normal(0, .1, MUESTRAS).astype(np.float32))
    # Only publishes after a separate Python process validates the serialized file.
    with tempfile.TemporaryDirectory(prefix="pfi_export_") as td:
        candidate = Path(td) / "reducido.pkl"
        refs = Path(td) / "referencias.npz"
        dump(p, candidate, compress=0)
        np.savez(refs, X=np.stack(Xs), z=np.stack(zs), scores=np.stack(scores),
                 pred=np.asarray(predictions), texto=np.asarray(texts), nombres=np.asarray(names))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "_validar",
                        str(candidate), str(refs)], check=True)
        with candidate.open("rb") as source, out.open("xb") as target:
            shutil.copyfileobj(source, target)
    print(f"\nEXPORTACIÓN VALIDADA: {out}")
    print(f"Tamaño lógico: {out.stat().st_size} bytes ({out.stat().st_size/1024:.2f} KiB)")
    print("Modelo original conservado. No uses la RAM de esta exportación como benchmark.")


def medir(args):
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
    import librosa
    from joblib import load
    imports = time.perf_counter() - INICIO
    t = time.perf_counter()
    if args.original:
        model = load(args.modelo)
        encoder = load(args.encoder)
        clf = model._classifier
        transform = lambda X: features_original(model, X)
    else:
        p = cargar_reducido(args.salida)
        encoder, clf = p["encoder"], p["classifier"]
        transform = lambda X: features(p, X)
    carga = time.perf_counter() - t
    print("Modo:", "original" if args.original else "reducido", flush=True)
    t = time.perf_counter()
    dummy = mfcc(np.zeros(MUESTRAS, dtype=np.float32))
    warm_mfcc = time.perf_counter() - t
    t = time.perf_counter()
    clf.predict(transform(dummy))
    warm_model = time.perf_counter() - t
    del dummy
    print("Grabando en 3 segundos...", flush=True)
    time.sleep(3)
    print("Grabando...", flush=True)
    t = time.perf_counter()
    audio = sd.rec(MUESTRAS, samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    grabacion = time.perf_counter() - t
    audio = audio[:, 0]
    args.audios.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    # Unique filename preserves prior recordings.
    with tempfile.NamedTemporaryFile(dir=args.audios, prefix="etapa1_", suffix=".wav", delete=False) as f:
        wav_path = f.name
    sf.write(wav_path, audio, SR)
    guardado = time.perf_counter() - t
    t = time.perf_counter()
    X = mfcc(audio)
    extraction = time.perf_counter() - t
    t = time.perf_counter()
    pred = clf.predict(transform(X))
    prediction = time.perf_counter() - t
    t = time.perf_counter()
    label = encoder.inverse_transform(pred)[0]
    etiqueta = time.perf_counter() - t
    print("Sonido detectado:", label)
    print("Audio guardado:", wav_path)
    for name, value in (
        ("Inicio e importaciones", imports), ("Carga", carga),
        ("Primera extracción MFCC", warm_mfcc), ("Precarga modelo", warm_model),
        ("Grabación", grabacion), ("Guardado WAV", guardado),
        ("MFCC", extraction), ("Predicción completa", prediction),
        ("Etiqueta", etiqueta), ("Procesamiento", extraction+prediction+etiqueta),
        ("Total desde inicio Python", time.perf_counter()-INICIO),
    ):
        print(f"{name}: {value:.4f} s")
    print("MEDICIÓN COMPLETADA")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modo", choices=("exportar", "medir", "_validar"))
    parser.add_argument("archivos", nargs="*")
    parser.add_argument("--modelo", type=Path, default=Path("modelo_detach_rocket.pkl"))
    parser.add_argument("--encoder", type=Path, default=Path("label_encoder.pkl"))
    parser.add_argument("--salida", type=Path, default=Path("modelo_detach_reducido_v1.pkl"))
    parser.add_argument("--audios", type=Path, default=Path("/home/rcaseres/audios_grabados"))
    parser.add_argument("--original", action="store_true")
    args = parser.parse_args()
    if args.modo == "_validar":
        require(len(args.archivos) == 2, "Se necesitan modelo y referencias")
        validar_archivo(*args.archivos)
    elif args.modo == "exportar":
        exportar(args)
    else:
        medir(args)


if __name__ == "__main__":
    main()
