import machine
import time
import random
import math

# ── Parámetros del entorno ──────────────────────────────────────────────────
NUM_STATES  = 11   # Alturas discretas: 5, 6, ..., 15 cm
NUM_ACTIONS = 6    # Acciones discretas de PWM

STATES  = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
ACTIONS = [0, 50, 100, 150, 200, 255]

# ── Tabla Q (inicializada en 0) ─────────────────────────────────────────────
Q = [[0.0] * NUM_ACTIONS for _ in range(NUM_STATES)]

# ── Parámetros de aprendizaje ───────────────────────────────────────────────
ALPHA   = 0.1   # Tasa de aprendizaje
GAMMA   = 0.9   # Factor de descuento
EPSILON = 0.2   # Probabilidad de exploración

# ── Setpoint ────────────────────────────────────────────────────────────────
SETPOINT = 10.0  # Altura deseada en cm

# ── Pines ESP32 ─────────────────────────────────────────────────────────────
TRIGGER_PIN = 12
ECHO_PIN    = 13
VALVE_PIN   = 25  # Pin PWM para la válvula solenoide

# ── Configuración de hardware ───────────────────────────────────────────────
trigger = machine.Pin(TRIGGER_PIN, machine.Pin.OUT)
echo    = machine.Pin(ECHO_PIN,    machine.Pin.IN)

# PWM en ESP32: frecuencia 1000 Hz, duty 0–1023 (10 bits por defecto)
valve_pwm = machine.PWM(machine.Pin(VALVE_PIN), freq=1000, duty=0)

def pwm_duty(value_0_255: int) -> int:
    """Convierte rango 0-255 (Arduino) a 0-1023 (MicroPython ESP32)."""
    return int(value_0_255 / 255 * 1023)

# ── Sensor ultrasónico HC-SR04 ───────────────────────────────────────────────
def read_ultrasonic_distance() -> float:
    """Devuelve la distancia en cm medida por el sensor HC-SR04."""
    # Pulso de disparo
    trigger.value(0)
    time.sleep_us(2)
    trigger.value(1)
    time.sleep_us(10)
    trigger.value(0)

    # Esperar flanco de subida del echo (timeout 30 ms)
    timeout = 30_000  # µs
    t0 = time.ticks_us()
    while echo.value() == 0:
        if time.ticks_diff(time.ticks_us(), t0) > timeout:
            return 10.0  # Valor por defecto si hay timeout

    # Medir duración del pulso echo
    t_start = time.ticks_us()
    while echo.value() == 1:
        if time.ticks_diff(time.ticks_us(), t_start) > timeout:
            return 10.0

    duration = time.ticks_diff(time.ticks_us(), t_start)
    distance = duration * 0.034 / 2  # Convertir a cm

    # Filtrar lecturas inválidas
    if distance <= 0 or distance > 200:
        return 10.0
    return distance

# ── Discretización de estado ────────────────────────────────────────────────
def discretize_state(current: float) -> int:
    """Retorna el índice del estado más cercano al valor actual."""
    for i, s in enumerate(STATES):
        if abs(current - s) < 0.5:
            return i
    return 5  # Estado por defecto → 10 cm

# ── Selección de acción (ε-greedy) ──────────────────────────────────────────
def select_action(state_idx: int) -> int:
    """Exploración aleatoria o explotación de la tabla Q."""
    if random.random() < EPSILON:
        return random.randint(0, NUM_ACTIONS - 1)  # Exploración
    else:
        # Explotación: acción con mayor Q
        best = 0
        for i in range(1, NUM_ACTIONS):
            if Q[state_idx][i] > Q[state_idx][best]:
                best = i
        return best

# ── Función de recompensa ────────────────────────────────────────────────────
def get_reward(state: float, setpoint: float) -> float:
    return -abs(state - setpoint)

# ── Bucle principal ──────────────────────────────────────────────────────────
print("Iniciando Q-Learning en ESP32...")

while True:
    # 1. Leer estado actual
    current_state = read_ultrasonic_distance()
    state_idx     = discretize_state(current_state)

    # 2. Seleccionar acción
    action_idx = select_action(state_idx)
    action     = ACTIONS[action_idx]

    # 3. Aplicar acción a la válvula (PWM)
    valve_pwm.duty(pwm_duty(action))

    # 4. Leer siguiente estado
    time.sleep_ms(50)                          # Pequeña pausa de asentamiento
    next_state     = read_ultrasonic_distance()
    next_state_idx = discretize_state(next_state)

    # 5. Calcular recompensa
    reward = get_reward(next_state, SETPOINT)

    # 6. Actualizar tabla Q (Q-Learning)
    max_next_q = max(Q[next_state_idx])
    Q[state_idx][action_idx] += ALPHA * (
        reward + GAMMA * max_next_q - Q[state_idx][action_idx]
    )

    # 7. Depuración por consola serial
    print(f"Dist: {current_state:.1f} cm | Acción: {action} PWM | Recompensa: {reward:.2f}")

    time.sleep_ms(100)