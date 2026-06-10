"""
control_maestro.py
==================
Orquestador interactivo para el flujo completo de redes neuronales:

  1. Muestra un menú para elegir qué controlador ejecutar en el ESP32
     (sigmoid / relu / tanh).
  2. Sube el archivo al ESP32 con mpremote y lo ejecuta.
  3. Al terminar pregunta si quiere re-entrenar la red con los datos CSV
     que se acaban de generar.
  4. Si re-entrena, exporta los nuevos pesos al archivo del controlador
     que se usó (actualiza las secciones X_MEAN, X_STD, Y_MEAN, Y_STD,
     W1…B4 directamente en el .py del ESP32).
  5. Vuelve a preguntar si desea ejecutar de nuevo y repite el ciclo.

Uso:
    python control_maestro.py
"""

import os
import sys
import json
import pickle
import re
import subprocess
import textwrap
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")            # sin ventana — guarda PNG
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — ajusta si tu workspace está en otra carpeta
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PESOS_JSON   = os.path.join(SCRIPT_DIR, "pesos_red_levitador.json")

CONTROLADORES = {
    "sigmoid": os.path.join(SCRIPT_DIR, "levitacion_red_neuronal_sigmoid.py"),
    "relu":    os.path.join(SCRIPT_DIR, "levitacion_red_neuronal_relu.py"),
    "tanh":    os.path.join(SCRIPT_DIR, "levitacion_red_neuronal_tanh.py"),
}

# Nombre del CSV que cada controlador guarda en el ESP32
CSV_NOMBRES = {
    "sigmoid": "datos_rn_sigmoid.csv",
    "relu":    "datos_rn_relu.csv",
    "tanh":    "datos_rn_tanh.csv",
}

# ─────────────────────────────────────────────────────────────────────────────
# CLASES Y FUNCIONES DE LA RED NEURONAL
# (copiadas de entrenar_red_levitador.py para no depender de importaciones)
# ─────────────────────────────────────────────────────────────────────────────

class Layer:
    def forward_propagation(self, input_data): pass
    def backward_propagation(self, output_error, learning_rate): pass

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))

    def forward_propagation(self, input_data):
        self.input  = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error   = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias    -= learning_rate * output_error
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation       = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        return self.activation(input_data)

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

class Network:
    def __init__(self):
        self.layers     = []
        self.loss       = None
        self.loss_prime = None

    def add(self, layer):       self.layers.append(layer)
    def use(self, loss, prime): self.loss = loss; self.loss_prime = prime

    def predict(self, input_data):
        result = []
        for sample in input_data:
            output = sample
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        history = []
        n = len(x_train)
        for i in range(epochs):
            err = 0.0
            for j in range(n):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                err  += self.loss(y_train[j], output)
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            err /= n
            history.append(err)
            if (i + 1) % 50 == 0:
                print(f"  epoch {i+1}/{epochs}  MSE={err:.6f}")
        return history

def _sigmoid(x):     return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def _sigmoid_prime(x): s = _sigmoid(x); return s * (1 - s)
def _relu(x):        return np.maximum(0, x)
def _relu_prime(x):  return (x > 0).astype(float)
def _tanh(x):        return np.tanh(x)
def _tanh_prime(x):  return 1 - np.tanh(x) ** 2
def _linear(x):      return x
def _linear_prime(x): return np.ones_like(x)
def _mse(yt, yp):    return np.mean((yt - yp) ** 2)
def _mse_prime(yt, yp): return 2 * (yp - yt) / yt.size

ACTIVACION_FNS = {
    "sigmoid": (_sigmoid, _sigmoid_prime),
    "relu":    (_relu,    _relu_prime),
    "tanh":    (_tanh,    _tanh_prime),
}

# ─────────────────────────────────────────────────────────────────────────────
# LÓGICA DIFUSA CENTROIDE (fuente de datos de entrenamiento base)
# ─────────────────────────────────────────────────────────────────────────────

def _trapmf(x, a, b, c, d):
    if x <= a or x >= d:  return 0.0
    if a < x <= b:        return (x - a) / (b - a) if b != a else 1.0
    if b < x <= c:        return 1.0
    if c < x < d:         return (d - x) / (d - c) if d != c else 1.0
    return 0.0

_NV, _NB, _NM, _NS, _Z, _PS, _PM, _PB, _PV = -6.0, -3.0, -1.5, -0.5, 0.0, 0.8, 2.5, 6.0, 18.0
_FAM = [
    [_NV, _NV, _NB, _NM, _NS, _Z,  _Z ],
    [_NV, _NB, _NM, _NS, _Z,  _Z,  _PS],
    [_NB, _NM, _NS, _NS, _Z,  _PS, _PM],
    [_NM, _NS, _NS, _Z,  _Z,  _PS, _PM],
    [_NM, _NS, _Z,  _Z,  _Z,  _PS, _PM],
    [_NM, _NS, _Z,  _Z,  _PS, _PS, _PM],
    [_NM, _NS, _Z,  _PS, _PS, _PM, _PB],
    [_NS, _Z,  _Z,  _PS, _PM, _PB, _PV],
    [_Z,  _Z,  _PS, _PM, _PB, _PV, _PV],
]
_KI = 0.10

def _fuzzy_centroide(error, deriv_f, integral):
    e_niv = [
        _trapmf(error, -50,-50,-15,-8),  _trapmf(error,-12,-9,-7,-4),
        _trapmf(error,-6,-4.5,-3.5,-1.5),_trapmf(error,-2.5,-1.5,-0.5,0),
        _trapmf(error,-1,-0.3,0.3,1),    _trapmf(error,0,0.5,1.5,2.5),
        _trapmf(error,1.5,3,5,6),        _trapmf(error,4,6,10,12),
        _trapmf(error,8,15,50,50),
    ]
    de_niv = [
        _trapmf(deriv_f,-80,-80,-25,-10), _trapmf(deriv_f,-20,-12,-8,-3),
        _trapmf(deriv_f,-6,-4,-2,0),      _trapmf(deriv_f,-1.5,-0.5,0.5,1.5),
        _trapmf(deriv_f,0,2,4,6),         _trapmf(deriv_f,3,7,13,20),
        _trapmf(deriv_f,10,25,80,80),
    ]
    reglas = []
    for i in range(9):
        if e_niv[i] <= 0: continue
        for j in range(7):
            peso = min(e_niv[i], de_niv[j])
            if peso > 0:
                reglas.append((peso, _FAM[i][j]))
    if not reglas: return _KI * integral
    num = sum(p * v for p, v in reglas)
    den = sum(p     for p, _ in reglas)
    return (num / den if den > 0 else 0.0) + _KI * integral

# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE DATOS DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def _generar_datos_fuzzy():
    """Genera la grilla base de datos desde lógica difusa centroide."""
    errores   = np.linspace(-15.0, 15.0, 15)
    derivadas = np.linspace(-30.0, 30.0, 15)
    integrales= np.linspace(-20.0, 20.0, 10)
    X, Y = [], []
    for e in errores:
        for d in derivadas:
            for it in integrales:
                dp = _fuzzy_centroide(float(e), float(d), float(it))
                X.append([e, d, it])
                Y.append([dp])
    return np.array(X, dtype='float32'), np.array(Y, dtype='float32')

def _cargar_csv_esp32(csv_path: str):
    """
    Carga un CSV generado por el controlador del ESP32.
    Columnas: tiempo, distancia, setpoint, error, derivada, integral, delta_pwm, pwm
    Devuelve (X, Y) donde X = [error, derivada, integral], Y = [delta_pwm].
    """
    import csv
    X, Y = [], []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                X.append([float(row['error']), float(row['derivada']), float(row['integral'])])
                Y.append([float(row['delta_pwm'])])
            except (KeyError, ValueError):
                continue
    if not X:
        raise ValueError(f"El CSV '{csv_path}' está vacío o tiene formato incorrecto.")
    return np.array(X, dtype='float32'), np.array(Y, dtype='float32')

# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = {
    "sigmoid": {"epochs": 1500, "lr": 0.005},
    "relu":    {"epochs": 2000, "lr": 0.001},
    "tanh":    {"epochs": 1500, "lr": 0.005},
}

def entrenar(activacion: str, csv_path: str | None = None):
    """
    Entrena la red para la activacion dada.
    Si csv_path existe y tiene datos, los combina con los datos fuzzy base.
    Guarda los pesos en pesos_red_levitador.json y en el pkl individual.
    Devuelve el dict de pesos para inyectarlos en el controlador .py.
    """
    print(f"\n{'='*60}")
    print(f"ENTRENANDO RED: {activacion.upper()}")
    print('='*60)

    # Datos de entrenamiento:
    # - Base sintética: barrido del espacio (error, deriv, integral) evaluado con fuzzy centroide.
    # - Datos reales de lógica difusa: los 3 CSVs grabados en el ESP32 con los 3 métodos fuzzy.
    #   Sus entradas (error/deriv/integral) son la distribución REAL del hardware; los targets
    #   se recalculan con fuzzy centroide (nunca se usan los delta_pwm grabados para evitar
    #   ciclos viciosos si la red anterior era mala).
    # - CSV opcional del controlador RN actual (pasado como argumento).

    X_fuzzy_all, Y_fuzzy_all = _generar_datos_fuzzy()
    X_parts = [X_fuzzy_all]
    Y_parts = [Y_fuzzy_all]
    print(f"Muestras fuzzy sintéticas (base): {len(X_fuzzy_all)}")

    # ── Cargar CSVs de las 3 ejecuciones de lógica difusa ──
    FUZZY_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "logica_difusa")
    fuzzy_csvs = [
        os.path.join(FUZZY_DIR, "datos_esp32_centroide.csv"),
        os.path.join(FUZZY_DIR, "datos_esp32_bisector.csv"),
        os.path.join(FUZZY_DIR, "datos_esp32_mom.csv"),
    ]
    for fcsv in fuzzy_csvs:
        if os.path.isfile(fcsv):
            try:
                X_fc, _ = _cargar_csv_esp32(fcsv)
                Y_fc = np.array(
                    [[_fuzzy_centroide(float(X_fc[i, 0]), float(X_fc[i, 1]), float(X_fc[i, 2]))]
                     for i in range(len(X_fc))],
                    dtype='float32'
                )
                X_parts.append(X_fc)
                Y_parts.append(Y_fc)
                print(f"  + {os.path.basename(fcsv)}: {len(X_fc)} muestras reales del hardware")
            except ValueError as e:
                print(f"  [AVISO] {os.path.basename(fcsv)}: {e}")
        else:
            print(f"  [INFO] No encontrado: {os.path.basename(fcsv)} (se omite)")

    # ── CSV del controlador RN actual (opcional) ──
    if csv_path and os.path.isfile(csv_path):
        try:
            X_real, _ = _cargar_csv_esp32(csv_path)
            Y_real = np.array(
                [[_fuzzy_centroide(float(X_real[i, 0]), float(X_real[i, 1]), float(X_real[i, 2]))]
                 for i in range(len(X_real))],
                dtype='float32'
            )
            X_parts.append(X_real)
            Y_parts.append(Y_real)
            print(f"  + CSV red neuronal ({activacion}): {len(X_real)} muestras")
        except ValueError as e:
            print(f"  [AVISO] CSV red neuronal: {e}")
    else:
        print("  CSV de red neuronal no disponible (se omite).")

    X_all = np.concatenate(X_parts, axis=0)
    Y_all = np.concatenate(Y_parts, axis=0)

    print(f"Total muestras de entrenamiento: {len(X_all)}")

    # Mezclar
    idx    = np.random.permutation(len(X_all))
    X_all  = X_all[idx]
    Y_all  = Y_all[idx]

    # Normalizar  (mínimo 1e-4 para que nunca se escriba 0.000000 en el ESP32)
    X_mean = X_all.mean(axis=0)
    X_std  = np.maximum(X_all.std(axis=0), 1e-4)
    Y_mean = float(Y_all.mean())
    Y_std  = float(max(float(Y_all.std()), 1e-4))

    X_norm = (X_all - X_mean) / X_std
    Y_norm = (Y_all - Y_mean) / Y_std

    x_train = X_norm.reshape(-1, 1, 3)
    y_train = Y_norm.reshape(-1, 1, 1)

    # Construir red
    act_fn, act_prime = ACTIVACION_FNS[activacion]
    cfg    = CONFIGS[activacion]
    EPOCHS = cfg["epochs"]
    LR     = cfg["lr"]

    net = Network()
    net.add(FCLayer(3, 16));  net.add(ActivationLayer(act_fn, act_prime))
    net.add(FCLayer(16, 12)); net.add(ActivationLayer(act_fn, act_prime))
    net.add(FCLayer(12, 8));  net.add(ActivationLayer(act_fn, act_prime))
    net.add(FCLayer(8, 1));   net.add(ActivationLayer(_linear, _linear_prime))
    net.use(_mse, _mse_prime)

    history = net.fit(x_train, y_train, epochs=EPOCHS, learning_rate=LR)

    # Gráficas
    png_train = os.path.join(SCRIPT_DIR, f"entrenamiento_levitador_{activacion}.png")
    plt.figure(figsize=(10, 4))
    plt.plot(history)
    plt.title(f"MSE durante entrenamiento ({activacion})")
    plt.xlabel("Época"); plt.ylabel("MSE"); plt.grid(True); plt.tight_layout()
    plt.savefig(png_train); plt.close()
    print(f"Gráfica guardada: {png_train}")

    preds_norm = net.predict(x_train)
    preds  = np.array([p[0][0] for p in preds_norm]) * Y_std + Y_mean
    reales = Y_all.flatten()
    png_comp = os.path.join(SCRIPT_DIR, f"comparacion_fuzzy_vs_rn_{activacion}.png")
    plt.figure(figsize=(12, 4))
    plt.plot(reales[:200], label="delta_pwm Fuzzy (ref.)", alpha=0.7)
    plt.plot(preds[:200],  label=f"delta_pwm RN ({activacion})", alpha=0.7)
    plt.title(f"Comparación Fuzzy vs RN ({activacion})")
    plt.xlabel("Muestra"); plt.ylabel("delta_pwm"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(png_comp); plt.close()
    print(f"Gráfica guardada: {png_comp}")

    # Guardar pesos
    capas_fc = [(l.weights, l.bias) for l in net.layers if isinstance(l, FCLayer)]

    pkl_file = os.path.join(SCRIPT_DIR, f"pesos_levitador_{activacion}.pkl")
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'layers':     capas_fc,
            'X_mean':     X_mean,
            'X_std':      X_std,
            'Y_mean':     Y_mean,
            'Y_std':      Y_std,
            'activacion': activacion,
        }, f)
    print(f"Pesos guardados en {pkl_file}")

    # JSON compartido
    try:
        with open(PESOS_JSON, 'r') as f:
            pesos_compartidos = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pesos_compartidos = {}

    pesos_compartidos[activacion] = {
        'activacion': activacion,
        'X_mean': X_mean.tolist(),
        'X_std':  X_std.tolist(),
        'Y_mean': Y_mean,
        'Y_std':  Y_std,
        'layers': [
            {'W': W.tolist(), 'b': b.flatten().tolist()}
            for W, b in capas_fc
        ],
    }
    with open(PESOS_JSON, 'w') as f:
        json.dump(pesos_compartidos, f, indent=2)
    print(f"Pesos actualizados en {PESOS_JSON}")

    return pesos_compartidos[activacion]

# ─────────────────────────────────────────────────────────────────────────────
# INYECCIÓN DE PESOS EN EL ARCHIVO .py DEL CONTROLADOR
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_list_1d(vals):
    # Usar notación científica para que valores muy pequeños no se redondeen a 0.0
    parts = []
    for v in vals:
        parts.append(f"{v:.6e}" if abs(v) < 0.001 and v != 0.0 else f"{v:.6f}")
    return "[" + ", ".join(parts) + "]"

def _fmt_list_2d(matrix):
    rows = []
    for row in matrix:
        rows.append("    [" + ", ".join(f"{v:.6f}" for v in row) + "],")
    return "[\n" + "\n".join(rows) + "\n]"

def inyectar_pesos(activacion: str, datos_pesos: dict):
    """
    Reescribe las secciones de pesos (X_MEAN, X_STD, Y_MEAN, Y_STD, W1..Wn, B1..Bn)
    en el archivo del controlador correspondiente.
    """
    archivo = CONTROLADORES[activacion]
    with open(archivo, 'r') as f:
        contenido = f.read()

    X_mean = datos_pesos['X_mean']
    # Clamp: ningún valor de X_std puede ser 0 (causaría ZeroDivisionError en el ESP32)
    X_std  = [max(float(v), 1e-4) for v in datos_pesos['X_std']]
    Y_mean = datos_pesos['Y_mean']
    Y_std  = max(float(datos_pesos['Y_std']), 1e-4)
    layers = datos_pesos['layers']  # lista de {'W':..., 'b':...}

    # ── Reemplazar escalares de normalización ──
    contenido = re.sub(
        r"X_MEAN\s*=\s*\[.*?\]",
        f"X_MEAN = {_fmt_list_1d(X_mean)}",
        contenido, flags=re.DOTALL
    )
    contenido = re.sub(
        r"X_STD\s*=\s*\[.*?\]",
        f"X_STD  = {_fmt_list_1d(X_std)}",
        contenido, flags=re.DOTALL
    )
    contenido = re.sub(
        r"Y_MEAN\s*=\s*[-\d.]+",
        f"Y_MEAN = {Y_mean:.6f}",
        contenido
    )
    contenido = re.sub(
        r"Y_STD\s*=\s*[-\d.]+",
        f"Y_STD  = {Y_std:.6f}",
        contenido
    )

    # ── Reemplazar matrices W y vectores B de cada capa ──
    for idx, capa in enumerate(layers, start=1):
        W = capa['W']
        b = capa['b']
        # W{idx} = [ [row], [row], ... ]
        contenido = re.sub(
            rf"W{idx}\s*=\s*\[.*?\n\]",
            f"W{idx} = {_fmt_list_2d(W)}",
            contenido, flags=re.DOTALL
        )
        # B{idx} = [...]
        contenido = re.sub(
            rf"B{idx}\s*=\s*\[.*?\]",
            f"B{idx} = {_fmt_list_1d(b)}",
            contenido, flags=re.DOTALL
        )

    with open(archivo, 'w') as f:
        f.write(contenido)

    print(f"Pesos inyectados en {archivo}")

# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN DE PUERTO COM (pyserial)
# ─────────────────────────────────────────────────────────────────────────────

def _detectar_puerto() -> str | None:
    """
    Intenta detectar automáticamente el puerto COM del ESP32/MicroPython.
    Retorna el nombre del puerto (p. ej. 'COM3') o None si no lo encuentra.
    """
    try:
        from serial.tools import list_ports
        candidatos = []
        for p in list_ports.comports():
            desc = (p.description or "").lower()
            # Chips USB-Serial típicos del ESP32
            if any(k in desc for k in ("cp210", "ch340", "ch341", "ftdi", "uart",
                                        "silabs", "usb serial", "esp")):
                candidatos.append(p.device)
        if len(candidatos) == 1:
            print(f"  Puerto detectado automáticamente: {candidatos[0]}")
            return candidatos[0]
        if len(candidatos) > 1:
            print("  Se encontraron varios puertos compatibles:")
            for i, p in enumerate(candidatos, 1):
                print(f"    {i}. {p}")
            while True:
                sel = input("  Elige el número del puerto: ").strip()
                if sel.isdigit() and 1 <= int(sel) <= len(candidatos):
                    return candidatos[int(sel) - 1]
                print("  Número inválido.")
        return None
    except ImportError:
        return None

def _pedir_puerto(puerto_guardado: list) -> str | None:
    """
    Devuelve el puerto COM a usar. Detecta automáticamente o pide al usuario.
    Usa una lista de un elemento como caché mutable entre llamadas.
    """
    if puerto_guardado[0]:
        return puerto_guardado[0]
    puerto = _detectar_puerto()
    if not puerto:
        print("  No se detectó el puerto automáticamente.")
        puerto = input("  Introduce el puerto COM manualmente (ej. COM3): ").strip()
    if puerto:
        puerto_guardado[0] = puerto
    return puerto or None

# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN EN ESP32 VÍA ampy (MicroPico / pyserial)
# ─────────────────────────────────────────────────────────────────────────────

# Cache del puerto COM entre llamadas  [puerto_str | None]
_PUERTO_CACHE = [None]

def _preparar_archivo_esp32(activacion: str, setpoint: float) -> str:
    """
    Crea una copia temporal del controlador con:
      - El setpoint hardcodeado (elimina el input() interactivo).
      - La pregunta de guardar CSV eliminada: siempre guarda automáticamente.
    Devuelve la ruta del archivo temporal.
    """
    archivo_orig = CONTROLADORES[activacion]
    archivo_tmp  = os.path.join(SCRIPT_DIR, f"_tmp_esp32_{activacion}.py")

    with open(archivo_orig, 'r') as f:
        lineas = f.readlines()

    salida = []
    i = 0
    while i < len(lineas):
        linea = lineas[i]

        # ── Reemplazar bloque try/except del setpoint ──
        # Detecta la línea "try:" seguida de setpoint = float(input(...))
        if linea.strip() == "try:" and i + 1 < len(lineas) and "setpoint" in lineas[i+1] and "input(" in lineas[i+1]:
            # buscar cuántas líneas ocupa el bloque (try + setpoint + except + setpoint)
            fin = i + 1
            while fin < len(lineas) and (lineas[fin].strip().startswith("setpoint") or
                                          lineas[fin].strip().startswith("except") or
                                          lineas[fin].strip() == ""):
                fin += 1
            salida.append(f"setpoint = {setpoint:.2f}  # fijado desde control_maestro.py\n")
            i = fin
            continue

        # ── Eliminar pregunta interactiva de CSV y forzar guardado automático ──
        # Detecta:  resp = input("Guardar ... (s/n): ").strip().lower()
        if "resp" in linea and "input(" in linea and "strip().lower()" in linea:
            # Reemplazar por condición siempre verdadera
            salida.append("    resp = 's'  # guardado automático desde control_maestro.py\n")
            i += 1
            continue

        salida.append(linea)
        i += 1

    with open(archivo_tmp, 'w') as f:
        f.writelines(salida)

    return archivo_tmp


def _terminal_serie(puerto: str, baud: int = 115200):
    """
    Mini terminal serie con pyserial.
    - Muestra en tiempo real todo lo que imprime el ESP32.
    - Ctrl+C en el PC envía el byte 0x03 al ESP32 (interrupción MicroPython),
      lo que dispara el except KeyboardInterrupt del controlador y guarda el CSV.
    """
    import serial as _serial

    print("  Conectado al ESP32. Presiona Ctrl+C para detener el controlador.\n")
    print('─'*60)
    with _serial.Serial(puerto, baud, timeout=0.05) as ser:
        try:
            while True:
                data = ser.read(256)
                if data:
                    print(data.decode('utf-8', errors='replace'), end='', flush=True)
        except KeyboardInterrupt:
            # Enviar Ctrl+C al ESP32 para que dispare el except KeyboardInterrupt
            ser.write(b'\x03')
            time.sleep(1.5)
            # Leer todo lo que quede (mensaje de guardado de CSV)
            ser.timeout = 0.2
            while True:
                data = ser.read(256)
                if not data:
                    break
                print(data.decode('utf-8', errors='replace'), end='', flush=True)
    print('\n' + '─'*60)
    print("  Controlador detenido.")


def ejecutar_en_esp32(activacion: str):
    """
    Flujo:
      1. Pide setpoint en el PC.
      2. Genera archivo temporal sin input().
      3. Sube como main.py con ampy.
      4. Reinicia el ESP32 con soft-reset (Ctrl+D) via pyserial → arranca main.py.
      5. Abre mini-terminal serie para ver output y enviar Ctrl+C al ESP32.
    """
    print(f"\n{'='*60}")
    print(f"  Preparando: {os.path.basename(CONTROLADORES[activacion])}")
    print('='*60)

    # 1. Pedir setpoint en el PC
    while True:
        sp_str = input("  Setpoint (cm, ej. 20): ").strip()
        try:
            setpoint = float(sp_str) if sp_str else 20.0
            break
        except ValueError:
            print("  Valor inválido. Introduce un número.")

    print("\n  IMPORTANTE: desconecta el MicroPico vREPL de VS Code antes de continuar")
    print("  (botón 'Disconnect' en la barra inferior de VS Code).")
    input("  Presiona Enter cuando el puerto esté libre...")

    puerto = _pedir_puerto(_PUERTO_CACHE)
    if not puerto:
        print("[ERROR] No se especificó un puerto COM. Saltando ejecución.")
        return -1

    # 2. Generar temporal con setpoint fijo y guardado automático de CSV
    archivo_tmp = _preparar_archivo_esp32(activacion, setpoint)

    # 3. Subir como main.py con ampy
    print(f"\n  Subiendo controlador (setpoint={setpoint:.1f} cm) → main.py en {puerto}...")
    cmd_put = ["ampy", "--port", puerto, "put", archivo_tmp, "main.py"]
    try:
        res_put = subprocess.run(cmd_put, check=False, capture_output=True, text=True)
        if res_put.returncode != 0:
            print(f"[ERROR] ampy put falló:\n{res_put.stderr.strip()}")
            _PUERTO_CACHE[0] = None
            os.remove(archivo_tmp)
            return res_put.returncode
        print("  Archivo subido correctamente.")
    except FileNotFoundError:
        print("\n[ERROR] 'ampy' no está instalado.")
        print("  Instálalo con:  pip install adafruit-ampy")
        os.remove(archivo_tmp)
        return -1
    finally:
        # Limpiar temporal (ya fue subido, no lo necesitamos más)
        try:
            os.remove(archivo_tmp)
        except OSError:
            pass

    # 4. Soft-reset del ESP32 → ejecuta main.py automáticamente
    try:
        import serial as _serial
        print(f"\n  Reiniciando ESP32 en {puerto}...")
        with _serial.Serial(puerto, 115200, timeout=1) as ser:
            ser.write(b'\x03\x03')   # Ctrl+C x2: interrumpir cualquier script previo
            time.sleep(0.3)
            ser.read_all()           # limpiar buffer
            ser.write(b'\x04')       # Ctrl+D: soft reset → corre main.py
            time.sleep(2.0)          # esperar boot
        print("  ESP32 reiniciado. Iniciando controlador...")
    except ImportError:
        print("\n[ERROR] pyserial no está instalado.")
        print("  Instálalo con:  pip install pyserial")
        return -1
    except Exception as e:
        print(f"\n[ERROR] No se pudo reiniciar el ESP32: {e}")
        return -1

    # 5. Mini-terminal: ver output y poder enviar Ctrl+C al ESP32
    _terminal_serie(puerto)

def descargar_csv_esp32(activacion: str) -> str | None:
    """
    Descarga el CSV generado por el controlador desde el ESP32 con ampy.
    Devuelve la ruta local del CSV o None si falla.
    """
    csv_remoto = CSV_NOMBRES[activacion]
    csv_local  = os.path.join(SCRIPT_DIR, f"datos_esp32_{activacion}.csv")
    print(f"\n  Descargando /{csv_remoto} desde el ESP32...")

    puerto = _pedir_puerto(_PUERTO_CACHE)
    if not puerto:
        print("[AVISO] No hay puerto COM configurado. No se descargó el CSV.")
        return None

    cmd_get = ["ampy", "--port", puerto, "get", csv_remoto, csv_local]
    try:
        res = subprocess.run(cmd_get, check=False, capture_output=True, text=True)
        if res.returncode == 0 and os.path.isfile(csv_local):
            print(f"  CSV descargado en: {csv_local}")
            return csv_local
        else:
            print(f"  [AVISO] No se pudo descargar el CSV.")
            if res.stderr.strip():
                print(f"  Detalle: {res.stderr.strip()}")
            return None
    except FileNotFoundError:
        print("[AVISO] 'ampy' no está instalado — no se descargó el CSV.")
        print("  Instálalo con:  pip install adafruit-ampy")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE CONSOLA
# ─────────────────────────────────────────────────────────────────────────────

def _preguntar_si_no(pregunta: str) -> bool:
    """Retorna True si el usuario responde 's', False si responde 'n'."""
    while True:
        resp = input(f"{pregunta} [s/n]: ").strip().lower()
        if resp in ('s', 'si', 'sí', 'yes', 'y'):
            return True
        if resp in ('n', 'no'):
            return False
        print("  Responde 's' o 'n'.")

def _mostrar_menu():
    print("\n" + "═"*60)
    print("  CONTROL MAESTRO — Redes Neuronales para Levitador de Pelota")
    print("═"*60)
    print("  Selecciona el controlador que deseas ejecutar en el ESP32:\n")
    print("    1. Red Neuronal con activación  SIGMOID")
    print("    2. Red Neuronal con activación  RELU")
    print("    3. Red Neuronal con activación  TANH")
    print("    0. Salir")
    print("─"*60)

OPCION_A_ACTIVACION = {
    "1": "sigmoid",
    "2": "relu",
    "3": "tanh",
}

# ─────────────────────────────────────────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n  Bienvenido al Control Maestro de Redes Neuronales")
    print("  Requisitos: pip install adafruit-ampy pyserial")
    print("  Antes de ejecutar en el ESP32 desconecta el MicroPico vREPL.\n")

    activacion_actual = None   # recuerda la última red usada

    while True:
        _mostrar_menu()
        opcion = input("  Opción: ").strip()

        if opcion == "0":
            print("\nSaliendo. ¡Hasta luego!\n")
            break

        if opcion not in OPCION_A_ACTIVACION:
            print("  Opción inválida. Intenta de nuevo.")
            continue

        activacion_actual = OPCION_A_ACTIVACION[opcion]

        # ── Ciclo ejecutar → (entrenar → ejecutar)* ──
        while True:
            # 1. Ejecutar en ESP32
            ejecutar_en_esp32(activacion_actual)

            # 2. ¿Se guardó CSV? Intentar descargarlo
            csv_path = None
            if _preguntar_si_no("\n¿Se guardó el CSV en el ESP32 y quieres descargarlo?"):
                csv_path = descargar_csv_esp32(activacion_actual)

            # 3. ¿Re-entrenar?
            if not _preguntar_si_no("\n¿Deseas re-entrenar la red con los nuevos datos?"):
                print("\nEntrenamiento omitido.")
                # Preguntar si volver al menú principal o salir
                if not _preguntar_si_no("¿Quieres volver al menú principal?"):
                    print("\nSaliendo. ¡Hasta luego!\n")
                    return
                break   # salir del ciclo interno → volver al menú

            # 4. Entrenar
            datos_pesos = entrenar(activacion_actual, csv_path)

            # 5. Inyectar pesos actualizados en el archivo del controlador
            print("\nActualizando pesos en el archivo del controlador...")
            inyectar_pesos(activacion_actual, datos_pesos)
            print("  Pesos actualizados correctamente.")

            # 6. ¿Ejecutar de nuevo con los nuevos pesos?
            if not _preguntar_si_no("\n¿Deseas ejecutar de nuevo el controlador con los nuevos pesos?"):
                if not _preguntar_si_no("¿Quieres volver al menú principal?"):
                    print("\nSaliendo. ¡Hasta luego!\n")
                    return
                break   # salir del ciclo interno → volver al menú

            # Si respondió sí → vuelve al inicio del ciclo while True (ejecutar otra vez)

if __name__ == "__main__":
    main()
