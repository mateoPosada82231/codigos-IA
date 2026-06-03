import pickle
import json
import numpy as np
import sys
import os

# =========================
# LEER PESOS GUARDADOS
# Uso: python exportar_pesos_esp32.py [archivo.pkl|pesos_red_levitador.json] [activacion]
#
# Ejemplos:
#   python exportar_pesos_esp32.py                          → lee pesos_red_levitador.json, activ. sigmoid
#   python exportar_pesos_esp32.py pesos_red_levitador.json relu
#   python exportar_pesos_esp32.py pesos_levitador_relu.pkl
# =========================

fuente   = sys.argv[1] if len(sys.argv) > 1 else 'pesos_red_levitador.json'
ACTIVACION = sys.argv[2].lower() if len(sys.argv) > 2 else 'sigmoid'

# Determinar formato por extensión
ext = os.path.splitext(fuente)[1].lower()

if ext == '.json':
    with open(fuente, 'r') as f:
        todos = json.load(f)
    if ACTIVACION not in todos:
        disponibles = list(todos.keys())
        raise KeyError(
            f"Activación '{ACTIVACION}' no encontrada en {fuente}. "
            f"Disponibles: {disponibles}"
        )
    datos_json = todos[ACTIVACION]
    X_mean = np.array(datos_json['X_mean'])
    X_std  = np.array(datos_json['X_std'])
    Y_mean = datos_json['Y_mean']
    Y_std  = datos_json['Y_std']
    # Convertir capas a tuplas (W numpy, b numpy) igual que pkl
    capas = [
        (np.array(capa['W']), np.array(capa['b']).reshape(1, -1))
        for capa in datos_json['layers']
    ]
else:
    # Formato .pkl original
    with open(fuente, 'rb') as f:
        datos = pickle.load(f)
    X_mean = datos['X_mean']
    X_std  = datos['X_std']
    Y_mean = datos['Y_mean']
    Y_std  = datos['Y_std']
    ACTIVACION = datos.get('activacion', ACTIVACION)
    capas = datos['layers']

# =========================
# GENERAR CÓDIGO MICROPYTHON
# =========================
print("# =============================================")
print("# PESOS GENERADOS AUTOMÁTICAMENTE - NO EDITAR")
print(f"# Generado por exportar_pesos_esp32.py  (fuente: {fuente}, activacion: {ACTIVACION})")
print("# =============================================\n")

print(f"X_MEAN = [{', '.join(f'{v:.6f}' for v in X_mean)}]")
print(f"X_STD  = [{', '.join(f'{v:.6f}' for v in X_std)}]")
print(f"Y_MEAN = {Y_mean:.6f}")
print(f"Y_STD  = {Y_std:.6f}\n")

for idx, (W, b) in enumerate(capas):
    W = np.array(W)
    b_flat = np.array(b).flatten()
    rows, cols = W.shape
    print(f"W{idx+1} = [")
    for r in range(rows):
        vals = ', '.join(f'{W[r, c]:.6f}' for c in range(cols))
        print(f"    [{vals}],")
    print("]")
    print(f"B{idx+1} = [{', '.join(f'{b_flat[c]:.6f}' for c in range(cols))}]\n")
