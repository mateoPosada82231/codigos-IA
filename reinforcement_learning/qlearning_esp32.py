from machine import Pin, PWM
import time
import random
import ujson
import array

# =========================
# Parámetros Q-learning
# =========================
NUM_STATES  = 11
NUM_ACTIONS = 6

# Estados discretos (cm) — cubre el rango físico completo del tubo.
# Antes era [10-20], lo que impedía distinguir "pelota pegada al sensor"
# de "pelota en zona de control".
STATES = [3, 5, 7, 9, 11, 13, 15, 17, 19, 22, 26]

# Hash simple del espacio de estados: detecta cambios de STATES al cargar qtable.json
_STATES_HASH = sum(STATES)

# Rango PWM solicitado
PWM_MIN = 275
PWM_MAX = 750

# Acciones discretas dentro de 275..750
ACTIONS = [275, 330, 410, 490, 600, 750]

QTABLE_FILE  = 'qtable.json'
SAVE_EVERY   = 100   # guardar cada N pasos
RESET_QTABLE = False # True = borrar tabla y empezar de cero; False = continuar aprendizaje

Q = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]

def _init_qtable_fisica():
    """
    Inicializa la tabla Q con priors físicos cuando no hay datos previos.
    Principio: pelota alta (dist < setpoint) → PWM bajo para bajar;
               pelota baja (dist > setpoint) → PWM alto para subir.
    """
    print("Inicializando tabla Q con priors físicos...")
    for i in range(NUM_STATES):
        dist = float(STATES[i])
        err  = dist - SETPOINT
        # Índice de acción ideal mapeado por error:
        # err=-12 (pelota arriba del todo) → j_ideal=0 (PWM mínimo)
        # err= 0  (en setpoint)            → j_ideal=2.5 (PWM medio)
        # err=+11 (pelota abajo del todo)  → j_ideal=4.8 (PWM máximo)
        j_ideal = (err + 12.0) / 24.0 * (NUM_ACTIONS - 1)
        if j_ideal < 0.0:            j_ideal = 0.0
        if j_ideal > NUM_ACTIONS - 1: j_ideal = float(NUM_ACTIONS - 1)
        for j in range(NUM_ACTIONS):
            Q[i][j] = 2.0 - abs(float(j) - j_ideal) * 0.8

def load_qtable():
    import os
    if RESET_QTABLE:
        try:
            os.remove(QTABLE_FILE)
        except Exception:
            pass
        print("Tabla Q reiniciada desde cero")
        _init_qtable_fisica()
        return
    try:
        with open(QTABLE_FILE, 'r') as f:
            data = ujson.load(f)
        # Nuevo formato: {'hash': int, 'Q': [[...], ...]}
        # Formato legacy: [[...], ...] (lista directa)
        try:
            saved_hash = data['hash']
            Q_data     = data['Q']
        except (TypeError, KeyError):
            # Formato legacy o incompatible
            print("Formato qtable legacy/incompatible, reiniciando con priors físicos")
            _init_qtable_fisica()
            return
        if saved_hash != _STATES_HASH:
            print("Espacio de estados cambiado, reiniciando con priors físicos")
            _init_qtable_fisica()
            return
        for i in range(NUM_STATES):
            for j in range(NUM_ACTIONS):
                Q[i][j] = Q_data[i][j]
        print("Tabla Q cargada desde", QTABLE_FILE)
    except Exception:
        print("No se encontró tabla Q previa, iniciando con priors físicos")
        _init_qtable_fisica()

def save_qtable():
    try:
        with open(QTABLE_FILE, 'w') as f:
            ujson.dump({'hash': _STATES_HASH, 'Q': Q}, f)
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

def select_action(state_idx, dist_cm):
    # Convención estándar: epsilon = probabilidad de explorar aleatoriamente.
    #   epsilon=0.20 -> 20% aleatorio, 80% greedy (Q-table)
    #   epsilon=0.00 -> 100% greedy (modo ejecución)
    #
    # Anulación de seguridad física: si la pelota está en zona peligrosa,
    # forzar siempre la acción correcta independientemente de epsilon.
    if dist_cm < 7.0:       # pelota muy alta -> PWM mínimo obligatorio
        return 0
    if dist_cm > 28.0:      # pelota muy abajo -> PWM máximo obligatorio
        return NUM_ACTIONS - 1

    if random.random() < EPSILON:
        # Explorar: acción aleatoria (restringida a zona segura según posición)
        return random.randint(0, NUM_ACTIONS - 1)
    else:
        # Explotar: mejor acción conocida
        best = 0
        best_q = Q[state_idx][0]
        for i in range(1, NUM_ACTIONS):
            if Q[state_idx][i] > best_q:
                best_q = Q[state_idx][i]
                best = i
        return best

def reward_fn(dist):
    # Zonas de peligro: penalización fuerte para que el agente aprenda
    # a NUNCA aplicar PWM alto cuando la pelota está pegada al sensor.
    if dist < 5.0:        return -25.0          # peligroso: pelota contra el sensor
    if dist > 30.0:       return -20.0          # peligroso: pelota en el fondo
    err = abs(dist - SETPOINT)
    if err < 0.5:         return  5.0           # excelente: muy cerca del setpoint
    if err < 1.5:         return  2.0           # bien
    if err < 3.0:         return  0.5           # aceptable
    return -err * 1.5                           # malo: proporcional al error

# --- Gestión de memoria para CSV (ring buffer en array, sin fragmentar heap) ---
MAX_LOGS      = 300
WARMUP_STEPS  = 100   # descartar primeros 100 samples (estabilización inicial)
_LOG_N        = 8
_log_buf      = array.array('f', [0.0] * (MAX_LOGS * _LOG_N))
_log_idx      = 0
_log_count    = 0
_warmup_steps = WARMUP_STEPS
_LOG_FILENAME = "datos_qlearning.csv"

def save_csv():
    resp = input("Guardar {} datos en CSV? (s/n): ".format(_log_count)).strip().lower()
    if resp != 's':
        return
    try:
        _start = (_log_idx - _log_count) % MAX_LOGS if _log_count == MAX_LOGS else 0
        with open(_LOG_FILENAME, "w") as f:
            f.write("tiempo,distancia,setpoint,error,pwm,accion,recompensa,epsilon\n")
            for i in range(_log_count):
                _b = ((_start + i) % MAX_LOGS) * _LOG_N
                f.write("{:.3f},{:.2f},{:.2f},{:.2f},{},{:.0f},{:.2f},{:.2f}\n".format(
                    _log_buf[_b], _log_buf[_b+1], _log_buf[_b+2], _log_buf[_b+3],
                    int(_log_buf[_b+4]), _log_buf[_b+5], _log_buf[_b+6], _log_buf[_b+7]))
        print("Guardado con exito en el ESP32:", _LOG_FILENAME)
    except Exception as e:
        print("Error al guardar:", e)

load_qtable()

print("Iniciando control Q-learning")
print("Setpoint fijo:", SETPOINT, "cm")
print("PWM rango:", PWM_MIN, "a", PWM_MAX)

# Fase inicial: maximo PWM 3 s para subir rapido
print("Elevacion inicial al maximo PWM (3 s)...")
fan.duty(PWM_MAX)
time.sleep(3.0)
# Rampa suave de bajada: 3 s en 60 pasos de 50 ms
print("Bajando suavemente hacia zona de control...")
_pwm_ramp_fin = PWM_MIN + (PWM_MAX - PWM_MIN) * 45 // 100
_pasos_ramp   = 60
_delta_ramp   = (PWM_MAX - _pwm_ramp_fin) / _pasos_ramp
for _i in range(_pasos_ramp):
    fan.duty(int(PWM_MAX - _delta_ramp * _i))
    time.sleep_ms(50)
fan.duty(_pwm_ramp_fin)
print("Rampa completada. Iniciando Q-learning...")

MAX_STEPS    = 800
EPSILON_STEP = 0.20   # cuanto sube cada 100 pasos
EPSILON_FIXED = False  # True = epsilon no cambia nunca (fijado desde control_maestro_rl.py)

step = 0
t_inicio = time.ticks_ms()
try:
    while True:
        # Estado actual
        dist_now = read_ultrasonic_distance()
        s = discretize_state(dist_now)

        # Acción
        a = select_action(s, dist_now)
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

        # CSV logging (omitido durante el warmup inicial)
        if _warmup_steps > 0:
            _warmup_steps -= 1
        else:
            tiempo_actual = time.ticks_diff(time.ticks_ms(), t_inicio) / 1000.0
            error = dist_now - SETPOINT
            _base = _log_idx * _LOG_N
            _log_buf[_base]   = tiempo_actual
            _log_buf[_base+1] = dist_now
            _log_buf[_base+2] = SETPOINT
            _log_buf[_base+3] = error
            _log_buf[_base+4] = float(pwm_cmd)
            _log_buf[_base+5] = float(a)
            _log_buf[_base+6] = r
            _log_buf[_base+7] = EPSILON
            _log_idx = (_log_idx + 1) % MAX_LOGS
            if _log_count < MAX_LOGS:
                _log_count += 1

        # Aumentar epsilon cada 100 pasos (tope 1.0), solo si no está fijo
        if step % 100 == 0:
            if not EPSILON_FIXED:
                EPSILON = min(1.0, EPSILON + EPSILON_STEP)
            save_qtable()
            print("[paso {}] Tabla Q guardada | epsilon={:.2f}".format(step, EPSILON))

        print("dist={:.2f}cm | pwm={} | reward={:.2f} | eps={:.1f}".format(dist_now, pwm_cmd, r, EPSILON))

        time.sleep_ms(100)

        if step >= MAX_STEPS:
            print("800 pasos completados. Finalizando.")
            break

except KeyboardInterrupt:
    fan.duty(0)
    save_qtable()
    print("Detenido. Tabla Q guardada en '{}'.".format(QTABLE_FILE))
    save_csv()

# Guardar también al final normal (sin KeyboardInterrupt)
if step >= MAX_STEPS:
    fan.duty(0)
    save_qtable()
    print("Sesión finalizada. Tabla Q guardada en '{}'.".format(QTABLE_FILE))
    save_csv()
