from machine import Pin, PWM, time_pulse_us
import time
import gc
import math
import array

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
DT_TARGET = 0.02  # Período objetivo (200 Hz)
PWM_MAX   = 750
PWM_MIN   = 275
ELEVACION_PWM = PWM_MAX
ELEVACION_SEGUNDOS = 1.8
ACTIVACION_OCULTA = "relu"

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

# --- Gestión de memoria para CSV (ring buffer en array, sin fragmentar heap) ---
MAX_LOGS      = 300
WARMUP_STEPS  = 100   # descartar primeros 100 samples (estabilización inicial)
_LOG_N        = 8
_log_buf      = array.array('f', [0.0] * (MAX_LOGS * _LOG_N))
_log_idx      = 0
_log_count    = 0
_warmup_steps = WARMUP_STEPS
_LOG_FILENAME = "datos_rn_relu.csv"

# =====================================================================
#  PESOS DE LA RED NEURONAL
#  *** REEMPLAZA ESTOS VALORES con la salida de exportar_pesos_esp32.py ***
# =====================================================================

X_MEAN = [-0.879959, -0.471249, -5.012318]
X_STD  = [8.210899, 16.428917, 14.101883]
Y_MEAN = 0.548125
Y_STD  = 5.751217

W1 = [
    [1.145278, -1.536911, -0.772608, 0.738129, 0.798030, 0.474519, -0.501410, 0.214485, 0.478430, 0.066722, -0.447544, -0.350974, -0.018877, 0.877038, 0.069824, -1.183737],
    [-0.308520, 0.808892, 0.070519, -0.038908, 1.454667, 0.771034, -0.146162, 0.085519, -0.839381, -0.788322, -0.147746, 1.386514, -2.486316, 0.892992, 0.082632, -0.305828],
    [-0.005446, -0.014885, -0.006112, 0.001566, 0.011039, 0.829072, -1.717927, 1.017094, 0.003700, -1.597371, -1.458132, 0.333814, 0.425012, -0.847385, 0.351372, 0.011920],
]
B1 = [0.378481, 0.283896, 0.052923, 0.246866, -0.766717, 0.286317, 0.160242, -0.031401, -0.212217, -0.166943, 0.095347, 0.591390, 0.045790, -0.474181, -0.086160, -0.522108]

W2 = [
    [-0.024446, -0.809731, -0.445934, 0.226351, -0.076616, -0.426062, 0.625696, 0.168744, 0.067423, 0.075180, -0.390704, -0.385251],
    [-0.458888, -0.579824, -1.138487, 0.125959, -0.274411, 0.143016, -0.200204, -0.038298, -0.522089, -0.433136, 0.506436, -0.252895],
    [0.354714, 0.383983, 0.195374, -0.469021, -0.083224, -0.124678, 0.496571, -0.078005, 0.649840, -0.083345, -0.475308, 0.448796],
    [0.086970, -0.118724, 0.330975, -0.726611, -0.235179, 0.350789, -0.113778, 0.816650, 0.479123, 0.085427, 0.613832, 0.779405],
    [-0.061916, -0.237261, 1.442538, -0.162622, -0.417774, -0.202913, 0.171394, -0.051718, 0.244514, -0.889938, 0.070067, -0.418408],
    [-0.725242, -0.213162, 0.045128, 0.115086, 0.024420, -0.153078, 0.424034, 0.437891, 0.016247, -0.205110, -0.207381, 0.343816],
    [0.029574, 0.041056, -0.162740, -0.278335, -0.211765, -0.055439, 0.019017, 0.059210, 0.193852, -0.799328, 0.052264, 0.346826],
    [-0.120225, 0.035586, 0.170624, 0.263788, 0.008220, 0.343183, 0.163418, -0.112489, -0.286582, 0.032938, 0.090582, -0.043503],
    [-0.162296, -0.241064, 0.040103, 0.378701, 0.257602, 0.557805, -0.218428, 1.063121, -0.143442, 0.006660, -0.545588, 0.298426],
    [-0.254860, -0.087970, -0.114229, 0.144228, -0.553331, 0.066273, -0.031155, -0.065618, -0.122142, -0.000553, -0.210766, 0.036530],
    [0.331603, -0.010552, 0.100881, 0.152981, -0.207923, -0.409391, 0.028904, 0.136240, 0.169979, -0.325521, 0.149155, -0.275198],
    [-0.250464, 0.933862, -0.385859, 0.028743, -0.154846, 0.032115, -0.000372, -0.741799, 0.169399, 0.075537, 0.783979, -0.183882],
    [0.238168, -0.110745, -1.047493, -0.073100, 0.108950, -0.156010, 0.167717, -0.804778, -0.976706, -0.781246, -0.332627, 0.127112],
    [-0.184245, 0.172878, 0.264442, -0.770046, -0.838713, -0.216486, -0.016222, -0.109746, -0.263553, -0.094250, 0.187580, -0.028874],
    [-0.103733, 0.018224, 0.277327, -0.532112, -0.055947, 0.163732, -0.621385, -0.069573, -0.102320, -0.466285, -0.067667, -0.582139],
    [0.482178, 0.451116, -0.029941, 0.447097, 0.032750, -0.085482, -0.173872, -0.280055, 0.382407, 0.066823, -0.463060, 0.345059],
]
B2 = [-0.601464, 0.666736, -0.408118, -0.131965, 0.059633, -0.056961, -0.061064, -0.109356, -0.360207, -0.037888, -0.118084, 0.739726]

W3 = [
    [-0.639849, -0.691730, 0.249825, -0.469360, -0.106173, -0.354842, 0.161988, 0.080385],
    [-0.510268, -0.217884, 0.625791, 0.386463, -0.743444, 0.519605, 0.747432, 0.734254],
    [-1.099101, 0.320457, 0.447982, 0.159037, -0.856399, -0.868904, -0.645217, 0.200953],
    [0.382843, 0.883266, 0.288617, 0.598050, -0.053234, -0.207211, 0.182486, 0.369938],
    [0.136946, 0.225784, -0.002770, 0.251574, 0.006985, 0.127067, 0.349075, 0.308739],
    [0.646696, -0.217990, -0.275199, 0.069134, 0.458891, 0.033579, 0.803401, 0.072326],
    [-0.111796, -0.232009, -0.157495, -0.230511, 0.038157, -0.158300, -0.207187, 0.624785],
    [0.179679, 0.403025, 0.755415, 0.249823, -0.310824, 1.073463, -0.043702, 0.635498],
    [-0.615514, 0.479611, -0.271360, 0.421135, -0.534372, -0.102864, -0.425361, -0.030027],
    [-0.226315, 0.872488, 0.008956, 0.171424, -0.399438, -0.850208, 0.515538, 0.930782],
    [0.799737, 0.325829, -0.177768, -0.666165, 0.888887, -0.455006, 0.310918, 0.213101],
    [0.028991, 0.240700, -0.687929, 0.200115, 0.399013, 0.381641, 0.147950, 0.147283],
]
B3 = [0.533216, -0.053231, -0.566022, -0.336220, 0.987090, -0.315492, 0.077450, 0.269920]

W4 = [
    [1.184836],
    [-0.204242],
    [-0.924512],
    [-0.729799],
    [-1.362989],
    [-1.160969],
    [-0.804244],
    [0.963703],
]
B4 = [0.443333]

W2 = [
    [-0.024446, -0.809731, -0.445934, 0.226351, -0.076616, -0.426062, 0.625696, 0.168744, 0.067423, 0.075180, -0.390704, -0.385251],
    [-0.458888, -0.579824, -1.138487, 0.125959, -0.274411, 0.143016, -0.200204, -0.038298, -0.522089, -0.433136, 0.506436, -0.252895],
    [0.354714, 0.383983, 0.195374, -0.469021, -0.083224, -0.124678, 0.496571, -0.078005, 0.649840, -0.083345, -0.475308, 0.448796],
    [0.086970, -0.118724, 0.330975, -0.726611, -0.235179, 0.350789, -0.113778, 0.816650, 0.479123, 0.085427, 0.613832, 0.779405],
    [-0.061916, -0.237261, 1.442538, -0.162622, -0.417774, -0.202913, 0.171394, -0.051718, 0.244514, -0.889938, 0.070067, -0.418408],
    [-0.725242, -0.213162, 0.045128, 0.115086, 0.024420, -0.153078, 0.424034, 0.437891, 0.016247, -0.205110, -0.207381, 0.343816],
    [0.029574, 0.041056, -0.162740, -0.278335, -0.211765, -0.055439, 0.019017, 0.059210, 0.193852, -0.799328, 0.052264, 0.346826],
    [-0.120225, 0.035586, 0.170624, 0.263788, 0.008220, 0.343183, 0.163418, -0.112489, -0.286582, 0.032938, 0.090582, -0.043503],
    [-0.162296, -0.241064, 0.040103, 0.378701, 0.257602, 0.557805, -0.218428, 1.063121, -0.143442, 0.006660, -0.545588, 0.298426],
    [-0.254860, -0.087970, -0.114229, 0.144228, -0.553331, 0.066273, -0.031155, -0.065618, -0.122142, -0.000553, -0.210766, 0.036530],
    [0.331603, -0.010552, 0.100881, 0.152981, -0.207923, -0.409391, 0.028904, 0.136240, 0.169979, -0.325521, 0.149155, -0.275198],
    [-0.250464, 0.933862, -0.385859, 0.028743, -0.154846, 0.032115, -0.000372, -0.741799, 0.169399, 0.075537, 0.783979, -0.183882],
    [0.238168, -0.110745, -1.047493, -0.073100, 0.108950, -0.156010, 0.167717, -0.804778, -0.976706, -0.781246, -0.332627, 0.127112],
    [-0.184245, 0.172878, 0.264442, -0.770046, -0.838713, -0.216486, -0.016222, -0.109746, -0.263553, -0.094250, 0.187580, -0.028874],
    [-0.103733, 0.018224, 0.277327, -0.532112, -0.055947, 0.163732, -0.621385, -0.069573, -0.102320, -0.466285, -0.067667, -0.582139],
    [0.482178, 0.451116, -0.029941, 0.447097, 0.032750, -0.085482, -0.173872, -0.280055, 0.382407, 0.066823, -0.463060, 0.345059],
]
B2 = [-0.601464, 0.666736, -0.408118, -0.131965, 0.059633, -0.056961, -0.061064, -0.109356, -0.360207, -0.037888, -0.118084, 0.739726]

W3 = [
    [-0.639849, -0.691730, 0.249825, -0.469360, -0.106173, -0.354842, 0.161988, 0.080385],
    [-0.510268, -0.217884, 0.625791, 0.386463, -0.743444, 0.519605, 0.747432, 0.734254],
    [-1.099101, 0.320457, 0.447982, 0.159037, -0.856399, -0.868904, -0.645217, 0.200953],
    [0.382843, 0.883266, 0.288617, 0.598050, -0.053234, -0.207211, 0.182486, 0.369938],
    [0.136946, 0.225784, -0.002770, 0.251574, 0.006985, 0.127067, 0.349075, 0.308739],
    [0.646696, -0.217990, -0.275199, 0.069134, 0.458891, 0.033579, 0.803401, 0.072326],
    [-0.111796, -0.232009, -0.157495, -0.230511, 0.038157, -0.158300, -0.207187, 0.624785],
    [0.179679, 0.403025, 0.755415, 0.249823, -0.310824, 1.073463, -0.043702, 0.635498],
    [-0.615514, 0.479611, -0.271360, 0.421135, -0.534372, -0.102864, -0.425361, -0.030027],
    [-0.226315, 0.872488, 0.008956, 0.171424, -0.399438, -0.850208, 0.515538, 0.930782],
    [0.799737, 0.325829, -0.177768, -0.666165, 0.888887, -0.455006, 0.310918, 0.213101],
    [0.028991, 0.240700, -0.687929, 0.200115, 0.399013, 0.381641, 0.147950, 0.147283],
]
B3 = [0.533216, -0.053231, -0.566022, -0.336220, 0.987090, -0.315492, 0.077450, 0.269920]

W4 = [
    [1.184836],
    [-0.204242],
    [-0.924512],
    [-0.729799],
    [-1.362989],
    [-1.160969],
    [-0.804244],
    [0.963703],
]
B4 = [0.443333]

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

# Buffers pre-asignados: evitan allocar heap en cada ciclo del loop
_x  = array.array('f', [0.0, 0.0, 0.0])
_h1 = array.array('f', [0.0] * 16)
_h2 = array.array('f', [0.0] * 12)
_h3 = array.array('f', [0.0] * 8)

def _dense_into(inputs, weights, bias, out):
    """Multiplica inputs × weights + bias, escribe resultado en out (sin heap)."""
    n_out = len(bias)
    n_in  = len(inputs)
    for j in range(n_out):
        s = bias[j]
        for i in range(n_in):
            s += inputs[i] * weights[i][j]
        out[j] = s

def red_neuronal(error, deriv_f, integral):
    # 1. Normalizar entradas
    _x[0] = (error    - X_MEAN[0]) / (X_STD[0] if X_STD[0] != 0.0 else 1e-4)
    _x[1] = (deriv_f  - X_MEAN[1]) / (X_STD[1] if X_STD[1] != 0.0 else 1e-4)
    _x[2] = (integral - X_MEAN[2]) / (X_STD[2] if X_STD[2] != 0.0 else 1e-4)

    # 2. Capa oculta 1: 3→16
    _dense_into(_x, W1, B1, _h1)
    for j in range(16): _h1[j] = activar(_h1[j])

    # 3. Capa oculta 2: 16→12
    _dense_into(_h1, W2, B2, _h2)
    for j in range(12): _h2[j] = activar(_h2[j])

    # 4. Capa oculta 3: 12→8
    _dense_into(_h2, W3, B3, _h3)
    for j in range(8): _h3[j] = activar(_h3[j])

    # 5. Capa de salida: 8→1, lineal
    s = B4[0]
    for i in range(8):
        s += _h3[i] * W4[i][0]

    # 6. Desnormalizar
    return s * Y_STD + Y_MEAN

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
print("Elevacion inicial al maximo PWM (5 s)...")
time.sleep(5.0)
pwm_actual = float(PWM_MAX)
print("Iniciando control por error...")

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

        # Logging protegido (omitido durante el warmup inicial)
        if _warmup_steps > 0:
            _warmup_steps -= 1
        else:
            _base = _log_idx * _LOG_N
            _log_buf[_base]   = tiempo_actual
            _log_buf[_base+1] = dist
            _log_buf[_base+2] = setpoint
            _log_buf[_base+3] = error
            _log_buf[_base+4] = deriv_f
            _log_buf[_base+5] = integral
            _log_buf[_base+6] = delta_pwm
            _log_buf[_base+7] = pwm_actual
            _log_idx = (_log_idx + 1) % MAX_LOGS
            if _log_count < MAX_LOGS:
                _log_count += 1

        ciclos += 1
        if ciclos % 100 == 0:
            gc.collect()

        print("PWM: {:7.2f} | Dist: {:6.2f} | Error: {:+7.2f} | dPWM: {:+7.2f}".format(
            pwm_actual, dist, error, delta_pwm))

        # Compensar tiempo de loop
        transcurrido = time.ticks_diff(time.ticks_ms(), t_ahora) / 1000.0
        espera = DT_TARGET - transcurrido
        if espera > 0:
            time.sleep(espera)

except KeyboardInterrupt:
    fan.duty(0)
    fan.deinit()
    print("\nMotor detenido. Aterrizaje seguro.")

    resp = input("Guardar {} datos en CSV? (s/n): ".format(_log_count)).strip().lower()
    if resp == 's':
        try:
            _start = (_log_idx - _log_count) % MAX_LOGS if _log_count == MAX_LOGS else 0
            with open(_LOG_FILENAME, "w") as f:
                f.write("tiempo,distancia,setpoint,error,derivada,integral,delta_pwm,pwm\n")
                for i in range(_log_count):
                    _b = ((_start + i) % MAX_LOGS) * _LOG_N
                    f.write("{:.3f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(
                        _log_buf[_b], _log_buf[_b+1], _log_buf[_b+2], _log_buf[_b+3],
                        _log_buf[_b+4], _log_buf[_b+5], _log_buf[_b+6], _log_buf[_b+7]))
            print("Guardado con exito en el ESP32.")
        except Exception as e:
            print("Error al guardar:", e)
