"""
Exporta los pesos del modelo DQN entrenado a un archivo MicroPython.
Si el modelo no existe, lo entrena primero automaticamente.
Uso: python exportar_pesos_dqn.py
Genera: pesos_dqn.py  (copiar al ESP32)
"""
import os
import torch
from dqn_tres_sensores import DQN, TresSensoresEnv, entrenar_dqn

ARCHIVO_MODELO = "dqn_tres_sensores.pth"
ARCHIVO_SALIDA = "pesos_dqn.py"

if not os.path.exists(ARCHIVO_MODELO):
    print(f"Modelo '{ARCHIVO_MODELO}' no encontrado. Entrenando...")
    env = TresSensoresEnv(distancia_objetivo=30)
    entrenar_dqn(env, episodios=1000, archivo_modelo=ARCHIVO_MODELO)
    print("Entrenamiento completado.")
else:
    print(f"Modelo '{ARCHIVO_MODELO}' encontrado.")

model = DQN(input_size=3, output_size=5)
model.load_state_dict(torch.load(ARCHIVO_MODELO))
model.eval()

def tensor_a_lista(t):
    return t.detach().numpy().tolist()

W1 = tensor_a_lista(model.fc1.weight)
B1 = tensor_a_lista(model.fc1.bias)
W2 = tensor_a_lista(model.fc2.weight)
B2 = tensor_a_lista(model.fc2.bias)
W3 = tensor_a_lista(model.fc3.weight)
B3 = tensor_a_lista(model.fc3.bias)

with open(ARCHIVO_SALIDA, "w") as f:
    f.write("# Pesos DQN exportados automaticamente - no editar\n\n")
    f.write(f"W1 = {W1}\n")
    f.write(f"B1 = {B1}\n\n")
    f.write(f"W2 = {W2}\n")
    f.write(f"B2 = {B2}\n\n")
    f.write(f"W3 = {W3}\n")
    f.write(f"B3 = {B3}\n")

print(f"Pesos exportados a {ARCHIVO_SALIDA}")
print("Copia pesos_dqn.py al ESP32 con MicroPico.")
