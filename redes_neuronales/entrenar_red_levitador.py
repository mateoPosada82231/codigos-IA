import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import sys

# =========================
# BASE LAYER
# =========================
class Layer:
    def forward_propagation(self, input_data):
        pass

    def backward_propagation(self, output_error, learning_rate):
        pass

# =========================
# FC LAYER
# =========================
class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

# =========================
# ACTIVATION LAYER
# =========================
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        return self.activation(input_data)

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

# =========================
# FUNCIONES DE ACTIVACIÓN Y PÉRDIDA
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(float)

def tanh_act(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def linear(x):
    return x

def linear_prime(x):
    return np.ones_like(x)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# =========================
# NETWORK
# =========================
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        result = []
        for i in range(len(input_data)):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        history = []
        for i in range(epochs):
            err = 0
            for j in range(len(x_train)):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                err += self.loss(y_train[j], output)
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            err /= len(x_train)
            history.append(err)
            if (i + 1) % 50 == 0:
                print(f"  epoch {i+1}/{epochs}  MSE={err:.6f}")
        return history

# =========================
# COPIA DEL MEJOR ALGORITMO DE LÓGICA DIFUSA (Centroide)
# Traducción a Python puro del algoritmo levitacion_fuzzy_centroide.py
# Genera datos de entrenamiento sin necesidad de archivos CSV externos
# =========================

def _trapmf(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    if b < x <= c:
        return 1.0
    if c < x < d:
        return (d - x) / (d - c) if d != c else 1.0
    return 0.0

def _defuzzify_centroide(reglas):
    if not reglas:
        return 0.0
    numerador   = sum(peso * valor for peso, valor in reglas)
    denominador = sum(peso          for peso, _     in reglas)
    return (numerador / denominador) if denominador > 0 else 0.0

# Singletons de salida (idénticos al algoritmo de referencia)
_NV_out = -6.0
_NB_out = -3.0
_NM_out = -1.5
_NS_out = -0.5
_Z_out  =  0.0
_PS_out =  0.8
_PM_out =  2.5
_PB_out =  6.0
_PV_out = 18.0

# Matriz FAM (9 niveles de error × 7 niveles de derivada)
_FAM = [
    [_NV_out, _NV_out, _NB_out, _NM_out, _NS_out, _Z_out,  _Z_out ],
    [_NV_out, _NB_out, _NM_out, _NS_out, _Z_out,  _Z_out,  _PS_out],
    [_NB_out, _NM_out, _NS_out, _NS_out, _Z_out,  _PS_out, _PM_out],
    [_NM_out, _NS_out, _NS_out, _Z_out,  _Z_out,  _PS_out, _PM_out],
    [_NM_out, _NS_out, _Z_out,  _Z_out,  _Z_out,  _PS_out, _PM_out],
    [_NM_out, _NS_out, _Z_out,  _Z_out,  _PS_out, _PS_out, _PM_out],
    [_NM_out, _NS_out, _Z_out,  _PS_out, _PS_out, _PM_out, _PB_out],
    [_NS_out, _Z_out,  _Z_out,  _PS_out, _PM_out, _PB_out, _PV_out],
    [_Z_out,  _Z_out,  _PS_out, _PM_out, _PB_out, _PV_out, _PV_out],
]

_KI             = 0.10
_INTEGRAL_MAX   = 40.0

def fuzzy_centroide(error, deriv_f, integral):
    """Calcula delta_pwm usando el algoritmo fuzzy centroide de referencia."""
    e_niveles = [
        _trapmf(error, -50, -50, -15, -8),
        _trapmf(error, -12,  -9,  -7, -4),
        _trapmf(error,  -6,-4.5,-3.5,-1.5),
        _trapmf(error,-2.5,-1.5,-0.5,  0),
        _trapmf(error,  -1,-0.3, 0.3,  1),
        _trapmf(error,   0, 0.5, 1.5,2.5),
        _trapmf(error, 1.5,   3,   5,  6),
        _trapmf(error,   4,   6,  10, 12),
        _trapmf(error,   8,  15,  50, 50),
    ]
    de_niveles = [
        _trapmf(deriv_f, -80, -80, -25, -10),
        _trapmf(deriv_f, -20, -12,  -8,  -3),
        _trapmf(deriv_f,  -6,  -4,  -2,   0),
        _trapmf(deriv_f,-1.5,-0.5, 0.5, 1.5),
        _trapmf(deriv_f,   0,   2,   4,   6),
        _trapmf(deriv_f,   3,   7,  13,  20),
        _trapmf(deriv_f,  10,  25,  80,  80),
    ]
    reglas = []
    for i in range(9):
        ei = e_niveles[i]
        if ei <= 0:
            continue
        for j in range(7):
            peso = min(ei, de_niveles[j])
            if peso > 0:
                reglas.append((peso, _FAM[i][j]))

    delta_fuzzy    = _defuzzify_centroide(reglas)
    delta_integral = _KI * integral
    return delta_fuzzy + delta_integral

# =========================
# GENERAR DATOS DESDE LA LÓGICA DIFUSA
# Se barre el espacio de entradas (error, derivada, integral) para obtener
# el delta_pwm que produce el algoritmo fuzzy de referencia.
# =========================
print("=" * 60)
print("GENERANDO DATOS DE ENTRENAMIENTO DESDE LÓGICA DIFUSA CENTROIDE")
print("=" * 60)

errores   = np.linspace(-15.0,  15.0, 40)
derivadas = np.linspace(-30.0,  30.0, 40)
integrales= np.linspace(-20.0,  20.0, 20)

X_list, Y_list = [], []
for e in errores:
    for d in derivadas:
        for it in integrales:
            dp = fuzzy_centroide(float(e), float(d), float(it))
            X_list.append([e, d, it])
            Y_list.append([dp])

X_all = np.array(X_list, dtype='float32')
Y_all = np.array(Y_list, dtype='float32')
print(f"Total muestras generadas: {len(X_all)}")

# Mezclar aleatoriamente
idx = np.random.permutation(len(X_all))
X_all = X_all[idx]
Y_all = Y_all[idx]

# =========================
# NORMALIZACIÓN
# =========================
X_mean = X_all.mean(axis=0)
X_std  = X_all.std(axis=0) + 1e-8
Y_mean = Y_all.mean()
Y_std  = Y_all.std() + 1e-8

X_norm = (X_all - X_mean) / X_std
Y_norm = (Y_all - Y_mean) / Y_std

x_train = X_norm.reshape(-1, 1, 3)
y_train = Y_norm.reshape(-1, 1, 1)

# =========================
# CONFIGURACIÓN DE ACTIVACIONES A ENTRENAR
# Se entrenan las 3 variantes con los mismos datos fuzzy.
# Opcionalmente se puede pasar una como argumento para entrenar solo esa.
# Uso: python entrenar_red_levitador.py [sigmoid|relu|tanh|all] [epochs] [lr]
# =========================
ARG_ACT = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
ACTIVACIONES_VALIDAS = {"sigmoid", "relu", "tanh", "all"}
if ARG_ACT not in ACTIVACIONES_VALIDAS:
    raise ValueError(f"Activación '{ARG_ACT}' no válida. Usa: {ACTIVACIONES_VALIDAS}")

CONFIGS = {
    "sigmoid": {"epochs": 1500, "lr": 0.005, "act_fn": sigmoid, "act_prime": sigmoid_prime},
    "relu":    {"epochs": 2000, "lr": 0.001, "act_fn": relu,    "act_prime": relu_prime},
    "tanh":    {"epochs": 1500, "lr": 0.005, "act_fn": tanh_act,"act_prime": tanh_prime},
}
if ARG_ACT == "all":
    activaciones_a_entrenar = list(CONFIGS.keys())
else:
    activaciones_a_entrenar = [ARG_ACT]
    if len(sys.argv) > 2:
        CONFIGS[ARG_ACT]["epochs"] = int(sys.argv[2])
    if len(sys.argv) > 3:
        CONFIGS[ARG_ACT]["lr"] = float(sys.argv[3])

# =========================
# ARCHIVO COMPARTIDO DE PESOS
# Todos los algoritmos de redes neuronales pueden leer de aquí.
# =========================
PESOS_JSON = 'pesos_red_levitador.json'
try:
    with open(PESOS_JSON, 'r') as f:
        pesos_compartidos = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    pesos_compartidos = {}

# =========================
# ENTRENAR CADA VARIANTE
# =========================
for ACTIVACION in activaciones_a_entrenar:
    cfg    = CONFIGS[ACTIVACION]
    EPOCHS = cfg["epochs"]
    LR     = cfg["lr"]
    act_fn      = cfg["act_fn"]
    act_prime   = cfg["act_prime"]

    print(f"\n{'='*60}")
    print(f"Entrenando red: {ACTIVACION.upper()}  |  épocas: {EPOCHS}  |  lr: {LR}")
    print('='*60)

    # Modelo — FCLayer(3→16) → Act → FCLayer(16→12) → Act → FCLayer(12→8) → Act → FCLayer(8→1) → Lineal
    net = Network()
    net.add(FCLayer(3, 16))
    net.add(ActivationLayer(act_fn, act_prime))
    net.add(FCLayer(16, 12))
    net.add(ActivationLayer(act_fn, act_prime))
    net.add(FCLayer(12, 8))
    net.add(ActivationLayer(act_fn, act_prime))
    net.add(FCLayer(8, 1))
    net.add(ActivationLayer(linear, linear_prime))
    net.use(mse, mse_prime)

    history = net.fit(x_train, y_train, epochs=EPOCHS, learning_rate=LR)

    # Gráfica de entrenamiento
    plt.figure(figsize=(10, 4))
    plt.plot(history)
    plt.title(f'Error MSE durante entrenamiento ({ACTIVACION})')
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'entrenamiento_levitador_{ACTIVACION}.png')
    plt.close()
    print(f"Gráfica guardada en entrenamiento_levitador_{ACTIVACION}.png")

    # Gráfica comparación fuzzy vs red neuronal
    preds_norm = net.predict(x_train)
    preds  = np.array([p[0][0] for p in preds_norm]) * Y_std + Y_mean
    reales = Y_all.flatten()

    plt.figure(figsize=(12, 4))
    plt.plot(reales[:200], label='delta_pwm Fuzzy (referencia)', alpha=0.7)
    plt.plot(preds[:200],  label=f'delta_pwm Red Neuronal ({ACTIVACION})', alpha=0.7)
    plt.title(f'Comparación: Fuzzy vs Red Neuronal ({ACTIVACION})')
    plt.xlabel('Muestra')
    plt.ylabel('delta_pwm')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'comparacion_fuzzy_vs_rn_{ACTIVACION}.png')
    plt.close()
    print(f"Gráfica guardada en comparacion_fuzzy_vs_rn_{ACTIVACION}.png")

    # --- Guardar pesos en archivo .pkl individual (compatibilidad con exportar_pesos_esp32.py) ---
    pkl_file = f'pesos_levitador_{ACTIVACION}.pkl'
    capas_fc = [(l.weights, l.bias) for l in net.layers if isinstance(l, FCLayer)]
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'layers':    capas_fc,
            'X_mean':    X_mean,
            'X_std':     X_std,
            'Y_mean':    float(Y_mean),
            'Y_std':     float(Y_std),
            'activacion': ACTIVACION,
        }, f)
    print(f"Pesos guardados en {pkl_file}")

    # --- Guardar en archivo compartido JSON ---
    pesos_compartidos[ACTIVACION] = {
        'activacion': ACTIVACION,
        'X_mean':     X_mean.tolist(),
        'X_std':      X_std.tolist(),
        'Y_mean':     float(Y_mean),
        'Y_std':      float(Y_std),
        'layers': [
            {
                'W': W.tolist(),
                'b': b.flatten().tolist(),
            }
            for W, b in capas_fc
        ],
    }

with open(PESOS_JSON, 'w') as f:
    json.dump(pesos_compartidos, f, indent=2)
print(f"\nPesos de todas las redes guardados en {PESOS_JSON}")
print("Los algoritmos de redes neuronales pueden leer sus pesos desde ese archivo.")
