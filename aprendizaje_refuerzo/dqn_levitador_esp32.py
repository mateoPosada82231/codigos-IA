"""
dqn_levitador_esp32.py
Inferencia DQN para levitador vertical en ESP32 (MicroPython).

Hardware (identico al resto del proyecto):
  - 1 x HC-SR04  -> TRIG GPIO27, ECHO GPIO26
  - Ventilador DC -> GPIO14, PWM 25 kHz

Archivos requeridos en el ESP32:
  - pesos_dqn_levitador.py   (generado por exportar_pesos_dqn_levitador.py)
  - dqn_levitador_esp32.py   (este archivo)

La red es 2 -> 24 -> 24 -> 6 (ReLU en las 2 capas ocultas). Recibe
[pos_normalizada, vel_normalizada] y produce los Q-values de las 6
acciones PWM [200, 280, 360, 440, 520, 600].
"""
import time
import machine
import array
import pesos_dqn_levitador as P


# ============================================================
# Parametros de la red y normalizacion (inyectados desde pesos_*.py)
# ============================================================
POS_NORM = P.POS_NORM
VEL_NORM = P.VEL_NORM
PWM_ACTIONS = P.PWM_ACTIONS

# Setpoint del levitador (referencia para columna "error" en el CSV)
SETPOINT = 15.0

# ============================================================
# Pines (mismo cableado que el resto del proyecto)
# ============================================================
TRIG_PIN = 27
ECHO_PIN = 26
FAN_PIN  = 14

trig = machine.Pin(TRIG_PIN, machine.Pin.OUT)
echo = machine.Pin(ECHO_PIN, machine.Pin.IN)
fan  = machine.PWM(machine.Pin(FAN_PIN), freq=25000, duty=0)

PWM_MIN = PWM_ACTIONS[0]
PWM_MAX = PWM_ACTIONS[-1]

# ============================================================
# Filtros del sensor (mediana 3 + EMA sobre la posicion)
# ============================================================
EMA_ALPHA = 0.40
_pos_filt = 15.0
_vel_filt = 0.0
_ultimo_t = None


def _leer_crudo():
    """Lectura cruda del HC-SR04 en cm. None si timeout."""
    trig.value(0)
    time.sleep_us(2)
    trig.value(1)
    time.sleep_us(10)
    trig.value(0)

    timeout_us = 30000
    t0 = time.ticks_us()
    while echo.value() == 0:
        if time.ticks_diff(time.ticks_us(), t0) > timeout_us:
            return None
    t1 = time.ticks_us()
    while echo.value() == 1:
        if time.ticks_diff(time.ticks_us(), t1) > timeout_us:
            return None
    dur = time.ticks_diff(time.ticks_us(), t1)
    d = (dur * 0.0343) / 2.0
    if d <= 0 or d > 200:
        return None
    return d


def leer_distancia():
    """Mediana de 3 lecturas crudas + EMA. Devuelve (pos_cm, vel_cm_s)."""
    global _pos_filt, _vel_filt, _ultimo_t

    muestras = []
    for _ in range(3):
        d = _leer_crudo()
        if d is not None:
            muestras.append(d)
        time.sleep_us(500)
    if not muestras:
        return _pos_filt, _vel_filt
    muestras.sort()
    mediana = muestras[len(muestras) // 2]

    pos_prev = _pos_filt
    _pos_filt = EMA_ALPHA * mediana + (1.0 - EMA_ALPHA) * _pos_filt

    ahora = time.ticks_ms()
    if _ultimo_t is None:
        dt = 0.05
    else:
        dt = time.ticks_diff(ahora, _ultimo_t) / 1000.0
        if dt <= 0 or dt > 0.5:
            dt = 0.05
    _ultimo_t = ahora
    _vel_filt = (_pos_filt - pos_prev) / dt

    if _pos_filt < 1.0:
        _pos_filt = 1.0
    elif _pos_filt > 39.0:
        _pos_filt = 39.0
    return _pos_filt, _vel_filt


# ============================================================
# Forward pass manual (mat * vec + bias, ReLU)
# ============================================================
def _linear(W, b, x):
    out = [0.0] * len(W)
    for i in range(len(W)):
        s = b[i]
        Wi = W[i]
        for j in range(len(x)):
            s += Wi[j] * x[j]
        out[i] = s
    return out


def _relu(v):
    return [vv if vv > 0.0 else 0.0 for vv in v]


def _argmax(lista):
    bi = 0
    bv = lista[0]
    for i in range(1, len(lista)):
        if lista[i] > bv:
            bv = lista[i]
            bi = i
    return bi


def forward(pos_cm, vel_cm_s):
    p = pos_cm / POS_NORM
    if p < 0.0: p = 0.0
    elif p > 1.0: p = 1.0
    v = vel_cm_s / VEL_NORM
    if v < -1.0: v = -1.0
    elif v > 1.0: v = 1.0
    x = [p, v]
    x = _relu(_linear(P.W1, P.B1, x))
    x = _relu(_linear(P.W2, P.B2, x))
    x = _linear(P.W3, P.B3, x)
    return x


# ============================================================
# CSV logging (ring buffer + warmup, sin fragmentar heap)
# ============================================================
MAX_LOGS      = 300
WARMUP_STEPS  = 100   # descartar primeros 100 samples (estabilización inicial)
_LOG_N        = 8
_log_buf      = array.array('f', [0.0] * (MAX_LOGS * _LOG_N))
_log_idx      = 0
_log_count    = 0
_warmup_steps = WARMUP_STEPS
_LOG_FILENAME = "datos_dqn.csv"


def save_csv():
    resp = input("Guardar {} datos en CSV? (s/n): ".format(_log_count)).strip().lower()
    if resp != 's':
        return
    try:
        _start = (_log_idx - _log_count) % MAX_LOGS if _log_count == MAX_LOGS else 0
        with open(_LOG_FILENAME, "w") as f:
            f.write("tiempo,distancia,velocidad,setpoint,error,pwm,accion,q_max\n")
            for i in range(_log_count):
                _b = ((_start + i) % MAX_LOGS) * _LOG_N
                f.write("{:.3f},{:.2f},{:.2f},{:.2f},{:.2f},{},{:.0f},{:.2f}\n".format(
                    _log_buf[_b], _log_buf[_b+1], _log_buf[_b+2], _log_buf[_b+3],
                    _log_buf[_b+4], int(_log_buf[_b+5]), _log_buf[_b+6], _log_buf[_b+7]))
        print("Guardado con exito en el ESP32:", _LOG_FILENAME)
    except Exception as e:
        print("Error al guardar:", e)


# ============================================================
# Bucle principal (lazo a 20 Hz)
# ============================================================
PERIODO_MS = 50
TIEMPO_ELEVACION_MS = 3000

print("DQN levitador - ESP32")
print("PWM_ACTIONS:", PWM_ACTIONS)

print("Elevacion inicial al maximo PWM ({} ms)...".format(TIEMPO_ELEVACION_MS))
fan.duty(PWM_MAX)
time.sleep_ms(TIEMPO_ELEVACION_MS)

print("Inferencia DQN activa. Setpoint 15 cm.")
t0 = time.ticks_ms()
try:
    while True:
        pos, vel = leer_distancia()
        q = forward(pos, vel)
        a = _argmax(q)
        q_max = q[a]
        pwm_cmd = PWM_ACTIONS[a]
        fan.duty(pwm_cmd)

        # CSV logging (omitido durante el warmup inicial)
        if _warmup_steps > 0:
            _warmup_steps -= 1
        else:
            t = time.ticks_diff(time.ticks_ms(), t0)
            error = pos - SETPOINT
            _base = _log_idx * _LOG_N
            _log_buf[_base]   = t / 1000.0
            _log_buf[_base+1] = pos
            _log_buf[_base+2] = vel
            _log_buf[_base+3] = SETPOINT
            _log_buf[_base+4] = error
            _log_buf[_base+5] = float(pwm_cmd)
            _log_buf[_base+6] = float(a)
            _log_buf[_base+7] = q_max
            _log_idx = (_log_idx + 1) % MAX_LOGS
            if _log_count < MAX_LOGS:
                _log_count += 1

        print("[{:6d}ms] pos={:5.2f}cm vel={:6.1f}cm/s | Q=[{:5.1f},{:5.1f},{:5.1f},{:5.1f},{:5.1f},{:5.1f}] | PWM={}".format(
            time.ticks_diff(time.ticks_ms(), t0), pos, vel,
            q[0], q[1], q[2], q[3], q[4], q[5], pwm_cmd))

        time.sleep_ms(PERIODO_MS)

except KeyboardInterrupt:
    fan.duty(0)
    print("\nMotor detenido.")
    save_csv()
