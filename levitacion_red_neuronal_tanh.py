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
ACTIVACION_OCULTA = "tanh"

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
    [0.695464, -0.523091, 0.569441, -0.342460, -1.243676, -0.115670, 0.348954, 3.177498, -2.182946, -0.138129],
    [-1.750033, -7.916194, 1.734870, 3.295787, -1.608698, 3.183644, -6.296974, -1.717811, 0.153351, -0.651055],
    [0.987543, 0.641589, -0.310316, 2.617954, 0.159863, 1.355020, 0.581378, 0.183456, -0.134578, -1.282007],
]
B1 = [0.236737, 2.636660, 0.550263, -2.751939, 1.654391, 4.059143, 0.399191, -1.589510, -0.984294, -0.459572]

W2 = [
    [0.593245, 0.229270, -0.379988, -0.101011, -0.102843, 0.300980],
    [1.836147, -1.750507, 0.564649, 1.031463, -0.258026, 1.838709],
    [-0.207129, 0.091672, 0.184335, 0.928021, 0.415407, 0.888380],
    [0.168804, -3.163981, 0.207458, -0.190266, 0.702213, 0.432599],
    [-1.262996, -0.468941, 1.220850, 1.488654, -0.238904, 0.879854],
    [-1.504295, -0.911056, -1.565702, -1.530733, -0.337587, -2.839393],
    [1.236082, 0.545598, -0.032463, -0.879914, -1.234873, 1.716261],
    [1.561315, 1.213725, -0.420841, -1.000693, -0.243129, 1.837776],
    [0.700007, -1.445516, 1.960304, 0.128578, 0.027383, 0.442123],
    [0.814463, 0.494061, -0.416938, -0.048104, -0.436778, -0.558014],
]
B2 = [-0.472800, -1.840011, -0.266970, -1.126245, 0.236305, -0.320453]

W3 = [
    [-0.884469],
    [2.000909],
    [1.700060],
    [-1.649107],
    [-0.340049],
    [-1.368670],
]
B3 = [0.156721]

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
