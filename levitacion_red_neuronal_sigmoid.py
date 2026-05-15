from machine import Pin, PWM, time_pulse_us
import time
import gc
import math

# =====================================================================
#  LEVITACIÓN DE PELOTA — Controlador Red Neuronal (reemplaza Fuzzy)
#  Red FCLayer(3→10→6→1) entrenada en PC con datos reales
#
#  INSTRUCCIONES:
#  1. Corre entrenar_red_levitador.py en PC para obtener pesos_levitador.pkl
#  2. Corre exportar_pesos_esp32.py en PC y copia la salida en la sección
#     "PESOS DE LA RED NEURONAL" más abajo (reemplaza los valores placeholder)
#  3. Carga este archivo en el ESP32 como main.py o boot.py
#
#  El resto de la lógica (sensor, integral, anti-windup, logging y CSV)
#  es idéntica a levitacion7niveles.py.
# =====================================================================

# Pines
TRIG_PIN = 27
ECHO_PIN = 26
FAN_PIN  = 14

trig = Pin(TRIG_PIN, Pin.OUT)
echo = Pin(ECHO_PIN, Pin.IN)
fan = PWM(Pin(FAN_PIN), freq=25000, duty=0)

# Variables de control
DT_TARGET = 0.05        # Período objetivo (20 Hz)
PWM_MAX   = 900
PWM_MIN   = 230
ELEVACION_PWM = PWM_MAX
ELEVACION_SEGUNDOS = 1.8
ACTIVACION_OCULTA = "sigmoid"

# Sensor
SENSOR_MIN  = 3.0
SENSOR_MAX  = 40.0
BUF_SIZE    = 7
OUTLIER_THR = 5.0

# Filtros
ALFA_DERIV = 0.40       # EMA de derivada

# Integral
INTEGRAL_MAX   = 40.0
INTEGRAL_DECAY = 0.998

buf = []
rechazos = 0
MAX_RECHAZOS = 5

# --- Gestión de memoria para CSV ---
data_log = []
MAX_LOGS = 1200

# =====================================================================
#  PESOS DE LA RED NEURONAL
#  *** REEMPLAZA ESTOS VALORES con la salida de exportar_pesos_esp32.py ***
# =====================================================================

X_MEAN = [3.215049, 2.197991, 6.101390]
X_STD  = [7.078470, 32.847725, 3.533574]

Y_MEAN = 0.053318
Y_STD  = 2.952577

W1 = [
    [0.475902, 4.880586, 2.922797, 5.753262, 0.543515, 0.844230, 0.048973, 0.425906, -4.330852, 3.419555],
    [-7.564624, 3.903639, -2.396129, 3.200053, 7.944611, -4.841145, 4.934371, -0.109516, -1.894149, -4.537044],
    [1.075050, 0.748568, 2.797146, 0.193852, 1.326005, -4.350423, 0.851812, -1.381793, 1.097492, -0.605272],
]
B1 = [0.709039, 3.407781, 0.738640, 0.607807, 1.358147, 1.754136, 3.802847, 0.591069, 3.861248, 0.098806]

W2 = [
    [-0.311386, 5.352912, 1.246851, -1.051062, 0.557074, -0.449300],
    [-1.757307, 4.983955, -0.448541, -2.954020, -0.519852, 0.174493],
    [0.506930, 0.354480, -0.659546, -0.140265, -1.627290, -3.962422],
    [0.906494, 0.545979, 0.513800, -3.742983, -0.565489, 1.252151],
    [0.754592, -4.862587, -0.585657, 0.722798, -0.744749, -0.634266],
    [2.234596, -3.950755, 0.507867, -0.066621, 0.046318, 0.008018],
    [-1.020512, -3.304977, -0.038483, 3.005325, -1.874357, -3.926049],
    [1.320539, 1.945387, -0.412634, -1.267027, -0.323327, -0.285502],
    [-4.133133, 5.740283, -0.678767, -0.265658, -0.094936, 4.088856],
    [2.334613, -3.214960, -0.870682, -0.072695, -0.703697, -1.899160],
]
B2 = [-0.169513, 1.787265, 0.246507, 0.888705, -0.564607, -0.664958]

W3 = [
    [2.437305],
    [-4.491347],
    [1.581161],
    [5.878410],
    [-1.178106],
    [-4.913285],
]
B3 = [1.310841]

# =====================================================================
#  FORWARD PASS DE LA RED NEURONAL (sin numpy, compatible MicroPython)
# =====================================================================

def sigmoid(x):
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def tanh(x):
    if x > 10.0:
        return 1.0
    if x < -10.0:
        return -1.0
    return math.tanh(x)

def relu(x):
    return x if x > 0.0 else 0.0

def activar(x):
    if ACTIVACION_OCULTA == "tanh":
        return tanh(x)
    if ACTIVACION_OCULTA == "relu":
        return relu(x)
    return sigmoid(x)

def _dense(inputs, weights, bias):
    """Multiplica vector inputs × matriz weights + bias. Retorna lista."""
    n_out = len(bias)
    n_in  = len(inputs)
    out = []
    for j in range(n_out):
        s = bias[j]
        for i in range(n_in):
            s += inputs[i] * weights[i][j]
        out.append(s)
    return out

def red_neuronal(error, deriv_f, integral):
    """Calcula delta_pwm usando la red neuronal entrenada.

    Pasos:
      1. Normalizar entradas
      2. Capa oculta 1 (3→10, activación configurable)
      3. Capa oculta 2 (10→6, activación configurable)
      4. Capa de salida (6→1, lineal)
      5. Desnormalizar salida
    """
    # 1. Normalizar entradas
    x = [
        (error    - X_MEAN[0]) / X_STD[0],
        (deriv_f  - X_MEAN[1]) / X_STD[1],
        (integral - X_MEAN[2]) / X_STD[2],
    ]

    # 2. Capa oculta 1: 3→10, activación configurable
    h1 = _dense(x, W1, B1)
    h1 = [activar(v) for v in h1]

    # 3. Capa oculta 2: 10→6
    h2 = _dense(h1, W2, B2)
    h2 = [activar(v) for v in h2]

    # 4. Capa de salida: 6→1, lineal
    out = _dense(h2, W3, B3)

    # 5. Desnormalizar salida
    delta_pwm = out[0] * Y_STD + Y_MEAN
    return delta_pwm

# =====================================================================
#  FUNCIÓN DE MEDICIÓN (idéntica a levitacion7niveles.py)
# =====================================================================

def medir_cm():
    global buf, rechazos
    trig.off()
    time.sleep_us(2)
    trig.on()
    time.sleep_us(10)
    trig.off()
    dur = time_pulse_us(echo, 1, 30000)

    if dur < 0:
        rechazos += 1
        if rechazos >= MAX_RECHAZOS:
            buf = []
            rechazos = 0
        if buf:
            return round(sorted(buf)[len(buf) // 2], 2)
        return -1.0

    d = dur * 0.034 / 2

    if d < SENSOR_MIN or d > SENSOR_MAX:
        rechazos += 1
        if rechazos >= MAX_RECHAZOS:
            buf = []
            rechazos = 0
        if buf:
            return round(sorted(buf)[len(buf) // 2], 2)
        return -1.0

    # Rechazo de outliers
    if len(buf) >= 3:
        mediana = sorted(buf)[len(buf) // 2]
        if abs(d - mediana) > OUTLIER_THR:
            rechazos += 1
            if rechazos >= MAX_RECHAZOS:
                buf = [d]
                rechazos = 0
                return round(d, 2)
            return round(mediana, 2)

    rechazos = 0
    buf.append(d)
    if len(buf) > BUF_SIZE:
        buf.pop(0)
    return round(sorted(buf)[len(buf) // 2], 2)

# =====================================================================
#  INICIO
# =====================================================================
print("=" * 60)
print("CONTROLADOR RED NEURONAL — levitador de pelota")
print("Activación oculta:", ACTIVACION_OCULTA)
print("=" * 60)

try:
    setpoint = float(input("Setpoint (cm, ej 20): ").strip())
except:
    setpoint = 20.0

pwm_actual = float(ELEVACION_PWM)
error_ant  = 0.0
deriv_f    = 0.0
integral   = 0.0
fan.duty(int(ELEVACION_PWM))
print("Elevación inicial al máximo PWM...")
time.sleep(ELEVACION_SEGUNDOS)
pwm_actual = 400.0
fan.duty(int(pwm_actual))
time.sleep(0.3)

t_inicio   = time.ticks_ms()
t_anterior = time.ticks_ms()
ciclos     = 0

gc.collect()

# =====================================================================
#  LOOP PRINCIPAL
# =====================================================================
try:
    while True:
        # Medir dt real
        t_ahora  = time.ticks_ms()
        dt_real  = time.ticks_diff(t_ahora, t_anterior) / 1000.0
        t_anterior = t_ahora
        if dt_real < 0.01:
            dt_real = 0.01
        elif dt_real > 0.5:
            dt_real = 0.5

        dist = medir_cm()
        if dist < 0:
            print("Sin lectura válida del sensor | PWM: {:7.2f} | rechazos: {}".format(pwm_actual, rechazos))
            time.sleep(DT_TARGET)
            continue

        tiempo_actual = time.ticks_diff(t_ahora, t_inicio) / 1000.0

        # 1. Error y Derivada
        error   = dist - setpoint
        deriv   = (error - error_ant) / dt_real
        deriv_f = ALFA_DERIV * deriv + (1.0 - ALFA_DERIV) * deriv_f
        error_ant = error

        # 2. Integral con anti-windup
        if abs(error) < 10.0:
            integral_inc = error * dt_real
            # Anti-windup: no acumular en la dirección de saturación
            if not (pwm_actual >= PWM_MAX and integral_inc > 0) and \
               not (pwm_actual <= PWM_MIN and integral_inc < 0):
                integral += integral_inc
            integral *= INTEGRAL_DECAY
            if integral > INTEGRAL_MAX:
                integral = INTEGRAL_MAX
            elif integral < -INTEGRAL_MAX:
                integral = -INTEGRAL_MAX

        # 3. Inferencia: Red Neuronal (reemplaza el bloque Fuzzy PD + integral)
        delta_pwm = red_neuronal(error, deriv_f, integral)

        # 4. Aplicar cambios
        pwm_actual += delta_pwm
        if pwm_actual > PWM_MAX:
            pwm_actual = float(PWM_MAX)
        if pwm_actual < PWM_MIN:
            pwm_actual = float(PWM_MIN)
        fan.duty(int(pwm_actual))

        # Logging protegido
        data_log.append((tiempo_actual, dist, setpoint, error, deriv_f, integral, delta_pwm, pwm_actual))

        if len(data_log) > MAX_LOGS:
            data_log.pop(0)

        ciclos += 1
        if ciclos % 100 == 0:
            gc.collect()

        print("PWM: {:7.2f} | Distancia: {:6.2f} | Error: {:+7.2f}".format(
            pwm_actual, dist, error))

        # Compensar tiempo de loop
        transcurrido = time.ticks_diff(time.ticks_ms(), t_ahora) / 1000.0
        espera = DT_TARGET - transcurrido
        if espera > 0:
            time.sleep(espera)

except KeyboardInterrupt:
    fan.duty(0)
    fan.deinit()
    print("\nMotor detenido. Aterrizaje seguro.")

    resp = input("Guardar {} datos en CSV? (s/n): ".format(len(data_log))).strip().lower()
    if resp == 's':
        try:
            with open("datos_levitacion.csv", "w") as f:
                f.write("tiempo,distancia,setpoint,error,derivada,integral,delta_pwm,pwm\n")
                for d in data_log:
                    f.write("{:.3f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(
                        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]))
            print("Guardado con éxito en el ESP32.")
        except Exception as e:
            print("Error al guardar:", e)
