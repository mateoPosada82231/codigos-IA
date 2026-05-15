import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pickle
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
                print(f"epoch {i+1}/{epochs}  MSE={err:.6f}")
        return history

# =========================
# CARGAR DATOS CSV
# =========================
def cargar_csv(filepath):
    X, Y = [], []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row['error']), float(row['derivada']), float(row['integral'])])
            Y.append([float(row['delta_pwm'])])
    return np.array(X, dtype='float32'), np.array(Y, dtype='float32')

archivos = [
    'datos_levitacion_10cm.csv',
    'datos_levitacion_15cm.csv',
    'datos_levitacion_20cm.csv',
]

X_all, Y_all = [], []
for archivo in archivos:
    if os.path.exists(archivo):
        X, Y = cargar_csv(archivo)
        X_all.append(X)
        Y_all.append(Y)
        print(f"Cargado {archivo}: {len(X)} muestras")

if not X_all:
    raise RuntimeError("No se encontró ningún CSV de datos. Asegúrate de que los archivos existen.")

X_all = np.concatenate(X_all, axis=0)
Y_all = np.concatenate(Y_all, axis=0)
print(f"Total muestras: {len(X_all)}")

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
# ACTIVACIÓN CONFIGURABLE
# Uso: python entrenar_red_levitador.py [sigmoid|relu|tanh] [epochs] [lr]
# Por defecto: sigmoid, 500 épocas, lr=0.01 (relu usa lr=0.001 por defecto)
# =========================
ACTIVACION = sys.argv[1].lower() if len(sys.argv) > 1 else "sigmoid"
ACTIVACIONES_VALIDAS = {"sigmoid", "relu", "tanh"}
if ACTIVACION not in ACTIVACIONES_VALIDAS:
    raise ValueError(f"Activación '{ACTIVACION}' no válida. Usa: {ACTIVACIONES_VALIDAS}")

if ACTIVACION == "relu":
    act_fn, act_prime = relu, relu_prime
    DEFAULT_EPOCHS = 2000
    DEFAULT_LR = 0.001
elif ACTIVACION == "tanh":
    act_fn, act_prime = tanh_act, tanh_prime
    DEFAULT_EPOCHS = 1500
    DEFAULT_LR = 0.005
else:
    act_fn, act_prime = sigmoid, sigmoid_prime
    DEFAULT_EPOCHS = 1500
    DEFAULT_LR = 0.005

EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_EPOCHS
LR     = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_LR

print(f"Activación oculta: {ACTIVACION}  |  épocas: {EPOCHS}  |  lr: {LR}")

# =========================
# MODELO — FCLayer(3→16) → Act → FCLayer(16→12) → Act → FCLayer(12→8) → Act → FCLayer(8→1) → Lineal
# =========================
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

print("\nIniciando entrenamiento...")
history = net.fit(x_train, y_train, epochs=EPOCHS, learning_rate=LR)

# =========================
# GRÁFICA DE ENTRENAMIENTO
# =========================
plt.figure(figsize=(10, 4))
plt.plot(history)
plt.title(f'Error MSE durante entrenamiento ({ACTIVACION})')
plt.xlabel('Época')
plt.ylabel('MSE')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'entrenamiento_levitador_{ACTIVACION}.png')
plt.show()
print(f"Gráfica guardada en entrenamiento_levitador_{ACTIVACION}.png")

# =========================
# GRÁFICA COMPARACIÓN FUZZY VS RED NEURONAL
# =========================
preds_norm = net.predict(x_train)
preds = np.array([p[0][0] for p in preds_norm]) * Y_std + Y_mean
reales = Y_all.flatten()

plt.figure(figsize=(12, 4))
plt.plot(reales[:200], label='delta_pwm real (Fuzzy)', alpha=0.7)
plt.plot(preds[:200],  label='delta_pwm Red Neuronal', alpha=0.7)
plt.title(f'Comparación: Fuzzy vs Red Neuronal ({ACTIVACION})')
plt.xlabel('Muestra')
plt.ylabel('delta_pwm')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'comparacion_fuzzy_vs_rn_{ACTIVACION}.png')
plt.show()
print(f"Gráfica guardada en comparacion_fuzzy_vs_rn_{ACTIVACION}.png")

# =========================
# GUARDAR PESOS
# =========================
pkl_file = f'pesos_levitador_{ACTIVACION}.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump({
        'layers': [(l.weights, l.bias) for l in net.layers if isinstance(l, FCLayer)],
        'X_mean': X_mean,
        'X_std':  X_std,
        'Y_mean': float(Y_mean),
        'Y_std':  float(Y_std),
        'activacion': ACTIVACION,
    }, f)
print(f"Pesos guardados en {pkl_file}")
