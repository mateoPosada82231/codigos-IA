# dqn_esp32.py - Inferencia DQN para ESP32 con MicroPython
# Requiere pesos_dqn.py en el ESP32
# Sensores HC-SR04: izquierdo=GPIO5, frente=GPIO18, derecho=GPIO19

import machine
import time
import math
import pesos_dqn as P

# --- ReLU ---
def relu(x):
    return [max(0.0, v) for v in x]

# --- Multiplicacion matriz-vector + bias ---
def linear(W, b, x):
    out = []
    for i in range(len(W)):
        s = b[i]
        for j in range(len(x)):
            s += W[i][j] * x[j]
        out.append(s)
    return out

# --- Forward pass de la red DQN ---
def forward(state):
    x = [float(v) for v in state]
    x = relu(linear(P.W1, P.B1, x))
    x = relu(linear(P.W2, P.B2, x))
    x = linear(P.W3, P.B3, x)
    return x

# --- Argmax ---
def argmax(lista):
    return lista.index(max(lista))

# --- Leer sensor HC-SR04 ---
def leer_hcsr04(trigger_pin, echo_pin, timeout_us=30000):
    trigger = machine.Pin(trigger_pin, machine.Pin.OUT)
    echo = machine.Pin(echo_pin, machine.Pin.IN)
    trigger.off()
    time.sleep_us(2)
    trigger.on()
    time.sleep_us(10)
    trigger.off()
    t_start = time.ticks_us()
    while echo.value() == 0:
        if time.ticks_diff(time.ticks_us(), t_start) > timeout_us:
            return 100  # Sin respuesta: lejos
    t_start = time.ticks_us()
    while echo.value() == 1:
        if time.ticks_diff(time.ticks_us(), t_start) > timeout_us:
            return 100
    duracion = time.ticks_diff(time.ticks_us(), t_start)
    distancia = (duracion * 0.0343) / 2
    return min(distancia, 100)

# --- Convertir distancia a estado discreto (igual que en el entorno) ---
def distancia_a_estado(d):
    if d >= 40:
        return 0  # Muy lejos
    elif d >= 20:
        return 1  # Lejos
    elif d >= 5:
        return 2  # Cerca
    else:
        return 3  # Muy cerca

# --- Nombres de acciones ---
ACCIONES = ["Avanzar", "Retroceder", "Girar izquierda", "Girar derecha", "Mantener"]

# --- Pines HC-SR04 (ajusta segun tu cableado) ---
TRIG_IZQ, ECHO_IZQ = 5, 4
TRIG_FRT, ECHO_FRT = 18, 19
TRIG_DER, ECHO_DER = 21, 22

# --- Bucle principal ---
print("DQN ESP32 iniciado")
while True:
    d_izq = leer_hcsr04(TRIG_IZQ, ECHO_IZQ)
    d_frt = leer_hcsr04(TRIG_FRT, ECHO_FRT)
    d_der = leer_hcsr04(TRIG_DER, ECHO_DER)

    estado = [
        distancia_a_estado(d_izq),
        distancia_a_estado(d_frt),
        distancia_a_estado(d_der)
    ]

    q_values = forward(estado)
    accion = argmax(q_values)

    print("Dist: izq={:.1f} frt={:.1f} der={:.1f} | Estado:{} | Accion: {}".format(
        d_izq, d_frt, d_der, estado, ACCIONES[accion]))

    time.sleep(0.5)
