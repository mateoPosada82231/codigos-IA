from machine import Pin, PWM
import time
import random
import ujson

# =========================
# Parámetros Q-learning
# =========================
NUM_STATES  = 11
NUM_ACTIONS = 6

# Estados discretos (cm)
STATES = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Rango PWM solicitado
PWM_MIN = 210
PWM_MAX = 600

# Acciones discretas dentro de 200..400
ACTIONS = [200, 280, 360, 440, 520, 600]

QTABLE_FILE = 'qtable.json'
SAVE_EVERY  = 100  # guardar cada N pasos

Q = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]

def load_qtable():
    try:
        with open(QTABLE_FILE, 'r') as f:
            data = ujson.load(f)
        for i in range(NUM_STATES):
            for j in range(NUM_ACTIONS):
                Q[i][j] = data[i][j]
        print("Tabla Q cargada desde", QTABLE_FILE)
    except Exception:
        print("No se encontró tabla Q previa, iniciando desde cero")

def save_qtable():
    try:
        with open(QTABLE_FILE, 'w') as f:
            ujson.dump(Q, f)
    except Exception as e:
        print("Error guardando tabla Q:", e)

ALPHA   = 0.10
GAMMA   = 0.90
EPSILON = 0.20

# Setpoint fijo solicitado
SETPOINT = 15.0

# =========================
# Pines (según tu repo)
# =========================
TRIG_PIN = 27
ECHO_PIN = 26
FAN_PIN  = 14

trig = Pin(TRIG_PIN, Pin.OUT)
echo = Pin(ECHO_PIN, Pin.IN)
fan  = PWM(Pin(FAN_PIN), freq=25000, duty=0)

def clamp_pwm(p):
    if p < PWM_MIN:
        return PWM_MIN
    if p > PWM_MAX:
        return PWM_MAX
    return p

def read_raw_distance():
    """Lectura HC-SR04 en cm con timeout."""
    trig.value(0)
    time.sleep_us(2)
    trig.value(1)
    time.sleep_us(10)
    trig.value(0)

    timeout_us = 30000

    t0 = time.ticks_us()
    while echo.value() == 0:
        if time.ticks_diff(time.ticks_us(), t0) > timeout_us:
            return None  # lectura invalida

    t1 = time.ticks_us()
    while echo.value() == 1:
        if time.ticks_diff(time.ticks_us(), t1) > timeout_us:
            return None  # lectura invalida

    dur = time.ticks_diff(time.ticks_us(), t1)
    dist = (dur * 0.0343) / 2.0

    if dist <= 0 or dist > 200:
        return None
    return dist

# Estado del filtro EMA (alpha: 0=muy suave, 1=sin filtro)
EMA_ALPHA  = 0.3
_ema_value = 15.0  # valor inicial

def read_ultrasonic_distance():
    """Mediana de 3 lecturas + filtro EMA para reducir ruido."""
    global _ema_value

    # Tomar 3 muestras
    samples = []
    for _ in range(3):
        d = read_raw_distance()
        if d is not None:
            samples.append(d)
        time.sleep_us(500)  # pequeña pausa entre disparos

    if len(samples) == 0:
        return _ema_value  # fallback al ultimo valor filtrado

    # Mediana
    samples.sort()
    median = samples[len(samples) // 2]

    # Filtro EMA
    _ema_value = EMA_ALPHA * median + (1.0 - EMA_ALPHA) * _ema_value

    return _ema_value

def discretize_state(x):
    # índice del estado más cercano
    best_i = 0
    best_d = abs(x - STATES[0])
    for i in range(1, NUM_STATES):
        d = abs(x - STATES[i])
        if d < best_d:
            best_d = d
            best_i = i
    return best_i

def select_action(state_idx):
    # epsilon=0 -> explora siempre | epsilon=1 -> explota siempre
    if random.random() < EPSILON:
        # explotar: mejor accion conocida
        best = 0
        best_q = Q[state_idx][0]
        for i in range(1, NUM_ACTIONS):
            if Q[state_idx][i] > best_q:
                best_q = Q[state_idx][i]
                best = i
        return best
    else:
        # explorar: accion aleatoria
        return random.randint(0, NUM_ACTIONS - 1)

def reward_fn(dist):
    # Penaliza alejamiento del setpoint
    return -abs(dist - SETPOINT)

load_qtable()

print("Iniciando control Q-learning")
print("Setpoint fijo:", SETPOINT, "cm")
print("PWM rango:", PWM_MIN, "a", PWM_MAX)

# Fase inicial para levantar: usar PWM 400
fan.duty(PWM_MAX)
time.sleep(1.2)

MAX_STEPS    = 800
EPSILON_STEP = 0.20   # cuanto sube cada 100 pasos

step = 0
try:
    while True:
        # Estado actual
        dist_now = read_ultrasonic_distance()
        s = discretize_state(dist_now)

        # Acción
        a = select_action(s)
        pwm_cmd = clamp_pwm(ACTIONS[a])
        fan.duty(pwm_cmd)

        # Espera corta y siguiente estado
        time.sleep_ms(60)
        dist_next = read_ultrasonic_distance()
        s_next = discretize_state(dist_next)

        # Recompensa y actualización Q
        r = reward_fn(dist_next)
        max_next_q = Q[s_next][0]
        for i in range(1, NUM_ACTIONS):
            if Q[s_next][i] > max_next_q:
                max_next_q = Q[s_next][i]

        Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * max_next_q - Q[s][a])

        step += 1

        # Aumentar epsilon cada 100 pasos (tope 1.0)
        if step % 100 == 0:
            EPSILON = min(1.0, EPSILON + EPSILON_STEP)
            save_qtable()
            print("[paso {}] Tabla Q guardada | epsilon={:.1f}".format(step, EPSILON))

        print("dist={:.2f}cm | pwm={} | reward={:.2f} | eps={:.1f}".format(dist_now, pwm_cmd, r, EPSILON))

        time.sleep_ms(100)

        if step >= MAX_STEPS:
            print("800 pasos completados. Finalizando.")
            break

except KeyboardInterrupt:
    fan.duty(0)
    save_qtable()
    print("Detenido. Tabla Q guardada en '{}'.".format(QTABLE_FILE))