import inspect
import importlib.metadata as metadata

import numpy as np
from joblib import load


def mib(cantidad_bytes):
    return cantidad_bytes / (1024 ** 2)


def describir_atributos(nombre, objeto):
    print(f"\n===== {nombre} =====")
    print("Clase:", type(objeto))
    print("Módulo:", type(objeto).__module__)

    try:
        archivo = inspect.getsourcefile(type(objeto))
    except TypeError:
        archivo = None

    print("Archivo fuente:", archivo)

    for atributo, valor in vars(objeto).items():

        if isinstance(valor, np.ndarray):
            print(
                f"{atributo}: ndarray | "
                f"shape={valor.shape} | "
                f"dtype={valor.dtype} | "
                f"RAM={mib(valor.nbytes):.4f} MiB"
            )

        elif isinstance(valor, (tuple, list)):
            arrays = [
                elemento
                for elemento in valor
                if isinstance(elemento, np.ndarray)
            ]

            if arrays:
                memoria_total = sum(
                    elemento.nbytes
                    for elemento in arrays
                )

                formas = [
                    elemento.shape
                    for elemento in arrays
                ]

                print(
                    f"{atributo}: {type(valor).__name__} | "
                    f"elementos={len(valor)} | "
                    f"formas={formas} | "
                    f"RAM={mib(memoria_total):.4f} MiB"
                )

        elif isinstance(
            valor,
            (int, float, str, bool, type(None))
        ):
            print(
                f"{atributo}: "
                f"{valor!r}"
            )

#Chequeo versiones instaladas

print("===== VERSIONES =====")

for paquete in (
    "detach-rocket",
    "sktime",
    "numpy",
    "numba",
    "scikit-learn",
    "joblib",
):
    try:
        version = metadata.version(paquete)
    except metadata.PackageNotFoundError:
        version = "no encontrada"

    print(f"{paquete}: {version}")


print("\nCarga el modelo con Joblib")

modelo = load("modelo_detach_rocket.pkl")
#Se recosntruye en memoria los objetos y parametros.
transformer = modelo._full_transformer
scaler = modelo._scaler
clasificador = modelo._classifier

mascara = np.asarray(
    modelo._feature_mask,
    dtype=bool
) #Chequea cuantas caracteristicas conserva DetachRocket

indices_seleccionados = np.flatnonzero(mascara)


# ==========================================
# RESUMEN DE LA MÁSCARA
# ==========================================

print("\n===== MÁSCARA =====")
print("Características totales:", mascara.size)
print(
    "Características seleccionadas:",
    indices_seleccionados.size
)
print(
    "Características descartadas:",
    mascara.size - indices_seleccionados.size
)
print(
    "Porcentaje conservado:",
    f"{100 * mascara.mean():.2f}%"
)

print(
    "Primeros 20 índices seleccionados:",
    indices_seleccionados[:20]
)

print(
    "Índices pares seleccionados:",
    np.count_nonzero(
        indices_seleccionados % 2 == 0
    )
)

print(
    "Índices impares seleccionados:",
    np.count_nonzero(
        indices_seleccionados % 2 == 1
    )
)


# ==========================================
# COMPONENTES INTERNOS
# ==========================================

describir_atributos(
    "TRANSFORMER ROCKET",
    transformer
)

describir_atributos(
    "STANDARD SCALER",
    scaler
)

describir_atributos(
    "RIDGE CLASSIFIER",
    clasificador
)


# ==========================================
# FORMAS IMPORTANTES
# ==========================================

print("\n===== FORMAS IMPORTANTES =====")

for atributo in (
    "mean_",
    "scale_",
    "var_",
):
    if hasattr(scaler, atributo):
        valor = getattr(
            scaler,
            atributo
        )

        print(
            f"scaler.{atributo}:",
            valor.shape
        )

if hasattr(clasificador, "coef_"):
    print(
        "classifier.coef_:",
        clasificador.coef_.shape
    )

if hasattr(clasificador, "intercept_"):
    print(
        "classifier.intercept_:",
        clasificador.intercept_.shape
    )


# ==========================================
# INSPECCIÓN DEL CÓDIGO DE SKTIME
# ==========================================

print("\n===== CÓDIGO DE TRANSFORMACIÓN =====")

modulo = inspect.getmodule(
    type(transformer)
)

print(
    "Archivo del módulo:",
    inspect.getsourcefile(modulo)
)

palabras_interes = (
    "ppv",
    "max",
    "* 2",
    "2 *",
    "2*",
    "_apply",
    "kernel",
)

for nombre in dir(modulo):

    if "kernel" not in nombre.lower():
        continue

    objeto = getattr(
        modulo,
        nombre
    )

    try:
        fuente = inspect.getsource(objeto)
    except (TypeError, OSError):
        continue

    lineas_relevantes = [
        linea.strip()
        for linea in fuente.splitlines()
        if any(
            palabra in linea.lower()
            for palabra in palabras_interes
        )
    ]

    if lineas_relevantes:
        print(f"\nFunción: {nombre}")

        for linea in lineas_relevantes:
            print("   ", linea)


print("\n===== FIN DEL DIAGNÓSTICO =====")