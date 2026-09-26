"""PFI etapa 6: capturas de micrófono con MFCC ligero y ROCKET C.
Copiar este archivo junto a etapa5.py y librocket_native.so.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import resource
import statistics
import sys
import tempfile
import time

SR, SAMPLES = 22050, 110250
HERE = Path(__file__).resolve().parent


def memory():
    values = {'Rss': float('nan'), 'Swap': float('nan')}
    try:
        with open('/proc/self/smaps_rollup') as f:
            for line in f:
                key = line.split(':', 1)[0]
                if key in values: values[key] = int(line.split()[1])/1024
    except OSError:
        pass
    return values['Rss'], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, values['Swap']


def show_mem(label):
    rss, peak, swap = memory()
    print(f'MEM {label}: RSS={rss:.2f} MiB; pico={peak:.2f} MiB; swap={swap:.2f} MiB', flush=True)
    return rss, peak, swap


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capturas', type=int, default=10, help='Número máximo de capturas, ENTER antes de cada una')
    p.add_argument('--dispositivo', help='Índice o nombre de dispositivo de entrada; por defecto el del sistema')
    p.add_argument('--listar-dispositivos', action='store_true')
    p.add_argument('--npz', type=Path, default=Path('/home/rcaseres/modelo_detach_numerico_v1.npz'))
    p.add_argument('--config', type=Path, default=Path('/home/rcaseres/etapa4_mfcc/datos_etapa4/mfcc_config.npz'))
    p.add_argument('--references', type=Path, default=Path('/home/rcaseres/etapa4_mfcc/datos_etapa4/referencias.npz'))
    p.add_argument('--trabajo', type=Path, default=HERE/'datos_etapa5', help='Carpeta con validacion_ok.json de etapa 5')
    p.add_argument('--salida', type=Path, default=Path('/home/rcaseres/audios_grabados'))
    return p


def main():
    args = parser().parse_args()
    if args.capturas < 1: raise ValueError('--capturas debe ser mayor que cero')
    import sounddevice as sd
    if args.listar_dispositivos:
        print(sd.query_devices())
        return
    import numpy as np
    import soundfile as sf
    import etapa5 as stage5
    from mfcc_ligero import MFCCLigero
    from rocket_native import NativeApply
    device = args.dispositivo
    if device is not None:
        try: device = int(device)
        except ValueError: pass
    stamp = args.trabajo/'validacion_ok.json'
    if not stamp.exists() or json.loads(stamp.read_text()) != stage5.signature(args):
        raise RuntimeError('Falta validación vigente de etapa 5. Ejecutar python3 etapa5.py validar desde su carpeta.')
    sd.check_input_settings(device=device, channels=1, dtype='float32', samplerate=SR)
    info = sd.query_devices(device=device, kind='input')
    print('Micrófono:', info['name'], flush=True)
    print('Captura REAL: mono, 22050 Hz, 5 segundos. Modelo cargado una sola vez.', flush=True)
    show_mem('antes del modelo')
    t = time.perf_counter()
    stage5.base.compiled_transform = NativeApply
    model = stage5.base.NumericModel(args.npz)
    extractor = MFCCLigero(args.config)
    load_time = time.perf_counter()-t
    t = time.perf_counter()
    dummy = extractor(np.zeros(SAMPLES, dtype=np.float32))
    model.predict(dummy)
    del dummy
    warm_time = time.perf_counter()-t
    # No se conservan matrices ni puntuaciones de las capturas anteriores.
    forbidden = {'librosa', 'numba', 'llvmlite', 'sklearn', 'sktime', 'pandas', 'joblib'} & set(sys.modules)
    if forbidden: raise RuntimeError(f'Dependencias inesperadas: {sorted(forbidden)}')
    args.salida.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix='etapa6_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_', dir=args.salida))
    meta = {'microfono': info['name'], 'dispositivo': device, 'sr': SR, 'muestras': SAMPLES,
            'modelo': str(args.npz.resolve()), 'config': str(args.config.resolve()),
            'carga_s': load_time, 'precarga_mfcc_y_modelo_s': warm_time,
            'formato_wav': 'FLOAT', 'python': sys.version, 'capturas_solicitadas': args.capturas}
    (session/'sesion.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    csv_path = session/'mediciones.csv'
    print(f'Carga: {load_time:.4f} s; precarga MFCC + modelo: {warm_time:.4f} s', flush=True)
    print('Carpeta de sesión:', session, flush=True)
    baseline = show_mem('listo para capturar')[0]
    fields = ['captura','etiqueta','wav','grabacion_s','guardado_s','mfcc_s','prediccion_s',
              'procesamiento_s','rss_mib','pico_mib','swap_mib']
    summaries = []
    state = 'completado'
    try:
        with csv_path.open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); f.flush()
            for i in range(1, args.capturas+1):
                while True:
                    option = input(f'\nCaptura {i}/{args.capturas}: ENTER para grabar; q para terminar: ').strip().lower()
                    if option in ('', 'q'): break
                    print('Presioná ENTER o escribí q.')
                if option == 'q':
                    state = 'fin solicitado'; break
                for sec in (3, 2, 1):
                    print(f'Grabando en {sec}...', flush=True); time.sleep(1)
                print('GRABANDO: hacé el sonido ahora (5 segundos).', flush=True)
                t = time.perf_counter()
                recorded = sd.rec(SAMPLES, samplerate=SR, channels=1, dtype='float32', device=device)
                status = sd.wait(ignore_errors=False)
                capture_time = time.perf_counter()-t
                if status:
                    raise RuntimeError(f'Captura con error PortAudio: {status}. No se clasifica como captura válida.')
                if recorded.shape != (SAMPLES,1) or recorded.dtype != np.float32 or not np.isfinite(recorded).all():
                    raise ValueError('Audio capturado incompatible o no finito')
                y = recorded[:, 0]
                wav = session/f'audio_{i:04d}.wav'
                t = time.perf_counter(); sf.write(wav, y, SR, subtype='FLOAT'); save_time = time.perf_counter()-t
                t = time.perf_counter(); X = extractor(y); mfcc_time = time.perf_counter()-t
                t = time.perf_counter(); scores, pred, labels = model.predict(X); pred_time = time.perf_counter()-t
                label = str(labels[0])
                del scores, pred, labels, X, y, recorded
                rss, peak, swap = memory()
                processing = mfcc_time+pred_time
                row = dict(zip(fields, [i,label,str(wav),capture_time,save_time,mfcc_time,pred_time,processing,rss,peak,swap]))
                writer.writerow(row); f.flush()
                summaries.append((rss, processing))
                print(f'Sonido detectado: {label}', flush=True)
                print(f'Audio guardado: {wav}', flush=True)
                print(f'Grabación: {capture_time:.4f} s; guardado: {save_time:.4f} s; MFCC: {mfcc_time:.4f} s; predicción: {pred_time:.4f} s', flush=True)
                print(f'CAPTURA {i}: procesamiento={processing:.4f} s; RSS={rss:.2f} MiB; pico={peak:.2f} MiB; swap={swap:.2f} MiB', flush=True)
    except (KeyboardInterrupt, EOFError):
        state = 'interrumpido por usuario'
        print('\nCaptura detenida; se conservan los resultados ya escritos.', flush=True)
    except Exception:
        state = 'error'
        raise
    finally:
        sd.stop()
        print('\n===== RESUMEN MICRÓFONO =====', flush=True)
        print('Estado:', state, '| Capturas válidas:', len(summaries), flush=True)
        print('CSV:', csv_path, flush=True)
        if summaries:
            values = [v[0] for v in summaries]
            print(f'RSS antes de capturas: {baseline:.2f} MiB', flush=True)
            print(f'RSS primera/última: {values[0]:.2f} / {values[-1]:.2f} MiB', flush=True)
            print(f'RSS mínimo/máximo: {min(values):.2f} / {max(values):.2f} MiB', flush=True)
            print(f'Diferencia última - primera: {values[-1]-values[0]:.2f} MiB', flush=True)
            print(f'Procesamiento medio: {statistics.mean(v[1] for v in summaries):.4f} s', flush=True)