import pickle
import numpy as np

# =========================
# LEER PESOS GUARDADOS
# =========================
with open('pesos_levitador.pkl', 'rb') as f:
    datos = pickle.load(f)

layers = datos['layers']
X_mean = datos['X_mean']
X_std  = datos['X_std']
Y_mean = datos['Y_mean']
Y_std  = datos['Y_std']

# =========================
# GENERAR CÓDIGO MICROPYTHON
# =========================
print("# =============================================")
print("# PESOS GENERADOS AUTOMÁTICAMENTE - NO EDITAR")
print("# Generado por exportar_pesos_esp32.py")
print("# =============================================\n")

print(f"X_MEAN = [{', '.join(f'{v:.6f}' for v in X_mean)}]")
print(f"X_STD  = [{', '.join(f'{v:.6f}' for v in X_std)}]")
print(f"Y_MEAN = {Y_mean:.6f}")
print(f"Y_STD  = {Y_std:.6f}\n")

for idx, (W, b) in enumerate(layers):
    rows, cols = W.shape
    print(f"W{idx+1} = [")
    for r in range(rows):
        vals = ', '.join(f'{W[r, c]:.6f}' for c in range(cols))
        print(f"    [{vals}],")
    print("]")
    print(f"B{idx+1} = [{', '.join(f'{b.flatten()[c]:.6f}' for c in range(cols))}]\n")
