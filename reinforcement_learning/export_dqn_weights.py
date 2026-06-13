"""
Exporta los pesos del modelo DQN del levitador a un archivo MicroPython.

Uso:
    python export_dqn_weights.py

Genera:
    dqn_weights.py   (copiar al ESP32 junto con dqn_esp32.py)

El archivo contiene W1, B1, W2, B2, W3, B3, ademas de los parametros de
normalizacion (POS_NORM, VEL_NORM) y la lista de acciones (PWM).
"""
import os
import torch

from dqn_train import DQNNet, LevitadorEnv, normalize_state, POS_NORM, VEL_NORM

ARCHIVO_MODELO = "dqn_model.pth"
ARCHIVO_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dqn_weights.py")

if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), ARCHIVO_MODELO)):
    raise FileNotFoundError(
        f"No se encontro '{ARCHIVO_MODELO}'. "
        "Ejecuta primero: python dqn_train.py"
    )

model = DQNNet()
model.load_state_dict(
    torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), ARCHIVO_MODELO),
               map_location="cpu")
)
model.eval()

ACTIONS = LevitadorEnv.ACTIONS


def tensor_a_lista(t):
    return t.detach().cpu().numpy().tolist()


W1 = tensor_a_lista(model.fc1.weight)
B1 = tensor_a_lista(model.fc1.bias)
W2 = tensor_a_lista(model.fc2.weight)
B2 = tensor_a_lista(model.fc2.bias)
W3 = tensor_a_lista(model.fc3.weight)
B3 = tensor_a_lista(model.fc3.bias)

with open(ARCHIVO_SALIDA, "w") as f:
    f.write("# Pesos DQN levitador exportados automaticamente - no editar\n\n")
    f.write(f"POS_NORM = {POS_NORM}\n")
    f.write(f"VEL_NORM = {VEL_NORM}\n")
    f.write(f"PWM_ACTIONS = {list(map(int, ACTIONS.tolist()))}\n\n")
    f.write(f"W1 = {W1}\n")
    f.write(f"B1 = {B1}\n\n")
    f.write(f"W2 = {W2}\n")
    f.write(f"B2 = {B2}\n\n")
    f.write(f"W3 = {W3}\n")
    f.write(f"B3 = {B3}\n")

print(f"Pesos exportados a {ARCHIVO_SALIDA}")
print(f"  W1: {len(W1)}x{len(W1[0])}  W2: {len(W2)}x{len(W2[0])}  W3: {len(W3)}x{len(W3[0])}")
print("Copia 'dqn_weights.py' y 'dqn_esp32.py' al ESP32 con MicroPico.")
