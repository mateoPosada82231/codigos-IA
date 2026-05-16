from machine import Pin, PWM, time_pulse_us
import time
import gc
import math

# =====================================================================
#  LEVITACIÓN DE PELOTA — Controlador Red Neuronal (reemplaza Fuzzy)
#  Red FCLayer(3→16→12→8→1) entrenada en PC con datos reales
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
DT_TARGET = 0.02  # Período objetivo (20 Hz)
PWM_MAX   = 310
PWM_MIN   = 205
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
    [-0.336977, -1.100056, -2.072914, -1.305943, 0.932461, -0.886965, -0.516632, -2.988986, -0.919044, 2.568051, -0.365067, -1.451283, 4.052559, -3.452499, 3.512136, 0.762228],
    [5.034006, -4.225824, 3.676621, -0.716412, 6.336661, -0.912275, 3.606602, 0.390942, 3.494121, -2.068714, -0.925750, 1.200825, 2.421424, 0.554454, 1.962403, -1.182831],
    [0.323842, -2.672730, -0.652476, -0.785944, 0.791926, 0.990332, 3.088618, 1.573190, 3.673136, 0.330307, 1.414552, -0.388239, 0.293838, 1.575305, -0.727081, -0.871890],
]
B1 = [1.941282, -1.043577, 2.639736, -3.416066, -0.006989, -2.157502, -2.287784, -0.024834, -0.299027, -1.827786, 0.290092, -0.075051, 1.222892, -3.101284, -3.366696, 0.141464]

W2 = [
    [0.434005, 0.404976, -1.411125, 0.004152, -0.478102, -0.207241, 3.126933, 0.146588, -1.258493, -0.495443, -0.646918, -0.005571],
    [0.436840, -0.671773, 0.177167, 0.173925, -0.493027, 0.222307, -1.942553, 0.006009, 0.975473, 1.069206, -0.051697, 1.438890],
    [-0.609024, 0.292190, -0.636963, -0.551713, -0.189720, 0.420542, 2.527719, -0.130593, -1.364584, -0.753149, -1.206701, -0.633617],
    [0.785145, -0.766214, 1.039510, 0.441460, 0.241200, -1.498807, -0.878794, -0.076483, 1.173244, 0.363546, 0.936927, 1.182713],
    [-0.324177, 0.103805, 0.012531, 0.641548, -0.053049, -0.024862, 4.053089, -0.213762, -0.426331, 0.064741, 0.062934, -1.655072],
    [-1.591578, -0.402230, -0.599106, -0.711958, -0.146940, 0.460267, 0.425323, 0.417360, -0.590607, -0.854310, -0.411983, -0.998439],
    [1.972791, -0.552571, 0.289741, 1.385172, 0.211333, -2.799243, 0.510143, 0.441901, 0.622506, -0.406182, -0.473017, 2.381197],
    [1.325284, -0.571871, 1.200052, -0.117598, -0.167065, -1.432102, -0.857747, -0.241953, 0.818853, 0.216784, 0.430132, 0.602141],
    [1.932461, 0.246039, -0.800750, 0.629827, -0.083829, -2.695754, 0.962394, 0.161423, -0.278787, -0.364900, -0.395096, 2.072182],
    [-1.732098, 0.081506, -0.906449, -0.818606, 0.445536, 2.255376, -0.176331, -0.329245, -0.690269, -1.054974, -0.014851, -2.105878],
    [-0.637402, -0.814917, -1.759416, -0.783441, 0.476810, -0.249336, 0.538073, -0.146124, 0.370954, -0.789502, -0.556621, -0.184706],
    [0.710063, 0.302993, 0.183962, -0.068696, -0.291449, -0.934290, 1.574440, 0.394512, -0.629614, 0.622036, 0.192235, -0.070867],
    [1.457209, -1.042971, 1.906584, 1.013565, -0.063168, -1.222898, -3.495694, -1.387901, 3.507646, 0.949463, 1.794263, 4.035397],
    [-1.797771, -0.246311, -0.970802, -0.768676, -0.324020, 0.809964, 0.217293, 0.955830, -0.194631, -0.769883, -0.834082, -2.939138],
    [-1.914929, 0.843503, -1.169712, -1.407400, -0.371899, 3.254414, 0.526176, 0.334739, -1.449976, -1.400351, -0.651643, -2.337904],
    [-0.236275, -0.650885, 0.282590, 0.594714, 0.109914, 0.148782, -0.332975, -1.099967, 0.441443, 0.312744, 1.076701, 0.280450],
]
B2 = [-0.352737, 0.104355, -0.629546, -0.621758, -0.001205, 0.103111, 0.318679, 0.129354, -0.695435, -0.501531, -0.041516, -0.278148]

W3 = [
    [-1.041929, -0.849237, -0.421976, -0.012181, -0.710553, 3.025520, 0.339284, -0.752126],
    [1.193167, -0.143387, -1.134226, -0.075481, 0.136416, -0.802323, -0.312653, 0.879398],
    [-1.667472, -0.314795, 0.162425, -0.189340, 0.103120, 1.631670, 0.806907, -1.212164],
    [-0.120456, -0.383507, -0.282250, -0.186717, -1.641410, 2.034457, 0.038538, 0.355855],
    [-0.364534, -0.548421, -0.293653, -0.241374, 0.493990, -0.281183, -0.617830, -0.256420],
    [1.192104, 0.006078, 0.821239, -0.207303, 0.996700, -4.855623, -1.944889, -0.202468],
    [5.372768, 0.077622, -0.813220, -0.703177, 0.338369, -1.334350, -1.149985, 1.197563],
    [0.871635, 0.438151, -0.096056, 0.208342, -0.268654, -1.911934, -0.641848, 0.687885],
    [-3.317178, -0.991347, 0.056482, -0.472122, -0.807293, 0.200660, -0.220214, 0.131122],
    [-0.883336, -0.276100, -0.238173, -0.450212, -0.442014, 1.527753, 0.059463, -0.454805],
    [-2.020212, -0.106221, 0.369511, -0.601045, -0.456635, 0.431554, -0.205818, -0.349146],
    [-3.456298, -0.137599, 0.328337, 0.514905, -0.846832, 3.237047, 0.658042, -0.386412],
]
B3 = [0.393848, -0.115406, -0.403443, -0.103587, -0.136808, -2.202840, -1.599290, 0.325642]

W4 = [
    [4.216313],
    [-0.240566],
    [0.053509],
    [0.668549],
    [-1.437866],
    [-4.292853],
    [-1.152320],
    [0.540776],
]
B4 = [0.259655]

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
      2. Capa oculta 1 (3→16, activación configurable)
      3. Capa oculta 2 (16→12, activación configurable)
      4. Capa oculta 3 (12→8, activación configurable)
      5. Capa de salida (8→1, lineal)
      6. Desnormalizar salida
    """
    # 1. Normalizar entradas
    x = [
        (error    - X_MEAN[0]) / X_STD[0],
        (deriv_f  - X_MEAN[1]) / X_STD[1],
        (integral - X_MEAN[2]) / X_STD[2],
    ]

    # 2. Capa oculta 1: 3→16, activación configurable
    h1 = _dense(x, W1, B1)
    h1 = [activar(v) for v in h1]

    # 3. Capa oculta 2: 16→12
    h2 = _dense(h1, W2, B2)
    h2 = [activar(v) for v in h2]

    # 4. Capa oculta 3: 12→8
    h3 = _dense(h2, W3, B3)
    h3 = [activar(v) for v in h3]

    # 5. Capa de salida: 8→1, lineal
    out = _dense(h3, W4, B4)

    # 6. Desnormalizar salida
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
