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
DT_TARGET = 0.02  # Período objetivo (50 Hz)
PWM_MAX   = 550
PWM_MIN   = 235
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

# --- Gestión de memoria para CSV (ring buffer en array, sin fragmentar heap) ---
MAX_LOGS      = 300
WARMUP_STEPS  = 100   # descartar primeros 100 samples (estabilización inicial)
_LOG_N        = 8
_log_buf      = array.array('f', [0.0] * (MAX_LOGS * _LOG_N))
_log_idx      = 0
_log_count    = 0
_warmup_steps = WARMUP_STEPS
_LOG_FILENAME = "datos_rn_tanh.csv"

# =====================================================================
#  PESOS DE LA RED NEURONAL
#  *** REEMPLAZA ESTOS VALORES con la salida de exportar_pesos_esp32.py ***
# =====================================================================

X_MEAN = [-0.837678, -0.040126, -0.960568]
X_STD  = [9.003469, 17.884190, 12.277892]
Y_MEAN = 1.244673
Y_STD  = 6.197051

W1 = [
    [-2.275366, 0.156348, -0.085588, 1.437565, -1.972456, -0.328583, 0.804543, -0.280829, -0.107111, 0.843791, 1.574890, -0.379818, 0.401082, -0.564961, 2.937765, 0.222777],
    [0.517261, 0.245647, -0.424363, 1.497918, 0.975904, 0.197681, -2.246546, 0.412148, -0.136788, 1.545790, -0.176253, -1.175435, 0.781540, -1.755634, 0.742576, -1.142748],
    [-0.108875, -0.453172, -0.522626, -0.044247, 0.016683, 0.377251, -0.005675, -0.382637, -0.364402, -0.050267, 0.036280, 0.032457, -0.455856, 0.318556, -0.060639, -0.738208],
]
B1 = [-0.082291, -0.082545, -0.345867, 0.160520, 0.435100, -0.023199, -0.027209, -0.175026, 0.356871, -1.967003, -0.007051, -0.773319, 0.903668, -0.302032, 0.163420, -0.043972]

W2 = [
    [-0.051885, 0.193294, -0.458635, 0.163626, -0.864577, -0.632543, -0.208422, 0.757938, -0.052855, 0.050151, -1.010969, -0.206721],
    [0.060977, -0.895964, -0.264136, 0.130326, -0.069453, 0.589349, -0.479287, -0.221120, -0.127624, -0.119615, 0.421632, -0.178297],
    [-0.129163, -0.512121, 0.533159, 0.067695, -0.179145, 0.346115, -0.260190, -0.004063, 0.043055, -0.095483, -0.019679, -0.130205],
    [-0.077868, -0.150445, 0.397987, -0.066199, -1.061761, 0.319514, -0.419212, 0.307323, 0.173653, 0.063229, 0.005448, -0.582983],
    [0.386785, -0.150890, -0.606353, 0.453163, -2.257182, 0.160308, -0.724236, -0.938524, -0.097830, 0.027471, -0.412259, -0.545604],
    [0.270109, -0.768862, -0.422248, 0.032955, -0.077750, 0.301521, -0.623843, -0.061403, 0.525132, 0.184532, 0.036515, -0.120671],
    [-1.223751, 0.296723, -0.111772, -0.308143, -1.553347, 0.338026, -0.134343, -1.253952, 0.729448, 1.463694, 0.315680, -1.054474],
    [-0.251425, 0.130201, -0.059445, 0.377024, 0.180636, -1.198259, -0.170834, -0.007156, -0.403587, -0.384243, -0.445049, 0.147591],
    [0.440952, 0.414681, 0.260065, -0.106066, -0.177030, -0.508961, 0.458085, 0.147712, -0.589155, -0.059966, -0.147567, 0.275816],
    [-1.965930, -0.396141, -1.324433, -0.355688, 0.651862, -0.299539, -0.498361, -1.467389, -0.398948, -0.883903, 0.615649, 0.001850],
    [-0.278168, -0.885998, -0.246822, 0.681370, 1.043665, -0.090885, -0.141884, -0.822887, 0.121483, -0.013591, 0.884824, 0.278645],
    [-0.399360, 0.010554, 0.877200, 0.267483, -0.181753, -0.101678, 0.076891, -0.357795, -1.097072, 0.432029, -0.757243, -0.212311],
    [0.210664, -0.178971, 0.034073, 0.634172, 0.365843, 0.373188, 0.278136, 0.219670, 0.878537, 0.293082, -0.335485, 0.027260],
    [0.087987, -0.009440, 0.281931, 0.180323, 0.729844, -0.279263, -0.415041, 0.281182, -0.270119, 0.351828, -0.316262, 0.007140],
    [0.298282, 0.387591, 0.318138, -0.328124, -1.434995, 0.474912, 0.044996, -0.516962, -0.064865, -0.624722, 0.158201, -0.507573],
    [0.103670, 0.183625, -0.184647, -0.063626, 0.093677, 0.031857, 0.184365, -0.056327, 0.137560, 0.544188, 0.228816, -0.043740],
]
B2 = [0.287480, 0.454473, -0.313430, -0.125183, 0.329551, 0.226809, 0.005127, 0.562178, 0.250380, -0.331905, -0.432149, -0.023861]

W3 = [
    [0.512008, -0.970070, -1.125892, -0.584434, -1.127455, -0.063652, -0.405088, -0.849780],
    [-0.319465, -0.671828, 0.820309, -0.285602, -0.188004, -0.340514, 0.065808, -0.875587],
    [0.744438, -0.153880, -0.797275, -0.941635, -0.000479, -0.527318, 0.695142, -1.155310],
    [0.111528, -0.033405, -0.477620, 0.124710, -0.550295, 0.177656, 0.462413, 0.646582],
    [0.047076, 1.281627, -0.643455, 1.891154, -1.180611, 0.588161, 0.055354, -0.177840],
    [-0.012864, 0.351139, -0.025650, -1.285582, -0.133630, -0.020721, -0.149235, -0.167012],
    [0.513937, -0.147450, 0.562193, -0.617900, 0.569221, -0.161861, 0.427512, 0.401171],
    [1.526174, -0.013377, -0.564672, -0.965989, -0.210774, 0.712726, -0.949713, -0.105253],
    [-0.200022, -0.257055, -0.006753, -0.380172, -0.663790, 0.412021, -0.212045, -0.163175],
    [0.431788, -0.234444, 0.192072, -1.101276, -0.386453, -0.456094, 0.035678, 0.370369],
    [-0.306500, 0.412889, 0.242697, 0.275664, -0.261331, 0.511241, 0.428835, -0.224479],
    [0.592477, 0.226346, -0.302928, -0.737483, -0.439487, 0.814891, 0.491883, 0.080781],
]
B3 = [-0.597582, -0.165483, 0.585041, -0.926207, 0.408692, 0.123620, -0.174600, 0.256498]

W4 = [
    [-0.541923],
    [0.600665],
    [0.547469],
    [0.840169],
    [-0.592011],
    [0.508534],
    [-0.853848],
    [-0.601234],
]
B4 = [0.314540]

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
