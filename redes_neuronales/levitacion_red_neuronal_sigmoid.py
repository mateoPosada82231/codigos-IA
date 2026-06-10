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
DT_TARGET = 0.02  # Período objetivo (20 Hz)
PWM_MAX   = 750
PWM_MIN   = 275
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

# --- Gestión de memoria para CSV (ring buffer en array, sin fragmentar heap) ---
MAX_LOGS      = 300
WARMUP_STEPS  = 100   # descartar primeros 100 samples (estabilización inicial)
_LOG_N        = 8
_log_buf      = array.array('f', [0.0] * (MAX_LOGS * _LOG_N))
_log_idx      = 0
_log_count    = 0
_warmup_steps = WARMUP_STEPS
_LOG_FILENAME = "datos_rn_sigmoid.csv"

# =====================================================================
#  PESOS DE LA RED NEURONAL
#  *** REEMPLAZA ESTOS VALORES con la salida de exportar_pesos_esp32.py ***
# =====================================================================

X_MEAN = [-0.766939, -0.549105, -4.885519]
X_STD  = [8.565711, 16.818464, 14.155710]
Y_MEAN = 0.671598
Y_STD  = 5.910508

W1 = [
    [-7.816041, 3.330975, -0.017980, 2.672906, 0.874333, -0.115644, -2.018160, 1.686006, -0.809993, -1.635448, -1.183013, -0.257260, -1.393073, 0.234633, -0.631510, -2.616760],
    [2.519984, -4.597332, 0.030978, 0.537119, 0.742064, -0.115354, -0.377505, 0.420417, 0.160029, 2.641609, 7.519824, -1.414998, -0.883107, -1.558617, -4.426317, 1.845828],
    [0.007675, 0.008289, 0.605008, 0.018356, -0.529351, 0.594017, -0.232318, -0.031915, 0.393973, 0.060432, -0.011990, 0.159446, 0.007116, -0.136340, 0.059276, 0.037516],
]
B1 = [1.617962, 2.002485, -0.532655, 0.592364, 0.727082, 0.358516, 1.667902, -0.285947, 0.423762, 0.450898, 0.145386, 0.951737, 1.898125, 0.263704, -0.601238, 2.069458]

W2 = [
    [-0.166993, -1.844880, 0.625950, 1.199131, -0.904900, -1.548514, 1.527588, -0.388780, -1.380492, 0.342381, 2.984511, -0.955263],
    [-0.821895, -2.581148, 0.844582, 1.483379, -0.663397, -1.565870, -1.603561, 0.689443, -1.840186, -0.038796, 2.822395, -0.528799],
    [0.000709, 0.628154, -0.285520, -0.880681, -0.501256, 0.346851, -1.231533, 0.266786, -0.249826, -0.417441, 0.166426, -0.239942],
    [0.337262, 1.087182, -0.463235, -1.280167, 0.767605, 1.128451, -2.354439, 0.277881, 0.615781, 0.020668, -1.227123, 0.012772],
    [-0.928426, 0.531727, 0.709309, 0.687168, 0.154160, -0.457578, -1.616302, -0.412026, 0.078894, 0.373549, -0.578571, 0.274496],
    [0.245968, 0.005405, -1.223646, -0.193287, -0.399672, 0.676586, -0.680421, 0.540561, -0.395132, -1.078972, 0.200141, -0.514101],
    [-0.792142, -2.749024, 0.251820, 0.618427, 1.203413, -1.162667, 1.103991, -0.927142, -0.643896, 0.155585, 0.974366, 0.029968],
    [0.122911, 1.051566, -0.369294, 0.269296, 0.431177, 0.315407, -1.501885, 0.341926, 0.956195, -0.477083, -0.980013, 0.049225],
    [0.494238, -0.824210, -0.215005, -0.036737, -0.289452, -0.432366, -0.249387, 0.073966, -0.485353, -1.075860, -0.197411, -0.343946],
    [0.257194, -0.221408, 0.418527, 0.474386, -0.337133, -1.215062, -0.593902, 0.479069, -0.514354, -0.299996, 0.169453, -0.818080],
    [0.897277, 2.545164, -1.111933, -1.508588, 1.106386, 1.496547, -1.903676, -0.360756, 1.368015, 0.388916, -1.412931, -0.212310],
    [-0.173627, -1.395714, -0.009700, 0.523789, -0.072043, -0.056043, -0.211708, -0.272934, -0.090043, -0.468508, 1.287622, 0.466825],
    [-0.259066, -2.347692, -0.894883, -0.332573, 0.001223, -0.649444, 0.046512, 0.082497, -0.137390, 0.406505, 1.086381, 0.493425],
    [-0.198597, -1.115529, 0.442949, 0.438815, 1.191280, 0.465942, 0.116321, 0.008446, -0.199037, -0.866256, 0.039264, 0.226703],
    [-0.020357, -1.870656, 0.084989, -0.159244, -0.791688, -2.319894, 1.388500, 0.037900, -0.986111, -0.713417, 1.013854, -0.310556],
    [-0.521628, -1.692394, -0.742239, 0.090756, -0.399163, -1.823686, 0.731821, 0.217787, -0.446525, 0.338601, 1.572784, -0.135782],
]
B2 = [-0.591127, -0.455576, -0.080462, 0.412511, 0.372015, -0.133521, -0.815295, -0.335891, -0.473694, -0.215931, 0.263043, -0.060921]

W3 = [
    [0.438331, 0.119170, -0.620263, 0.309892, -0.726267, 0.400256, -0.428239, -1.257601],
    [-0.217241, -0.047582, 0.549236, 3.846167, 0.069112, -0.582237, 0.190259, 0.659016],
    [0.959279, -0.338312, 0.269318, -1.288894, 0.048146, -0.464563, 0.638274, 0.935539],
    [0.051262, 0.504369, 0.907137, -1.926543, 0.308799, -0.468382, 0.909848, 0.193161],
    [-0.026896, 0.170177, 0.513386, 2.564188, -0.219122, -1.082677, 1.106923, 0.358745],
    [-0.758311, 0.610626, -0.289873, 2.859714, -0.073211, 0.740539, -1.145028, -0.240520],
    [0.428427, -0.066439, 0.778633, -0.782861, 0.271180, -0.778390, 0.912208, 2.042080],
    [-0.299980, -0.126931, -0.687187, -0.373125, -0.519218, -0.279164, -0.297439, -1.346967],
    [-0.976842, -0.457697, 0.046256, 2.034759, -0.037429, 0.512725, -1.105190, 0.231075],
    [0.369264, 0.593280, 0.931094, 0.421591, -0.335870, -0.031689, 0.712524, 0.743814],
    [-0.703470, -0.883554, -1.378904, -3.733097, -0.022711, 0.644323, -0.862164, -0.934526],
    [0.878897, -0.206918, 0.408902, 0.869419, -0.327911, 0.055834, 0.438110, 0.406136],
]
B3 = [-0.479374, -0.251033, -0.532666, -0.496192, -0.296055, -0.040224, -0.279862, 0.076956]

W4 = [
    [-0.729669],
    [0.032279],
    [-1.377161],
    [3.233260],
    [0.151841],
    [1.415976],
    [-1.583000],
    [-1.883502],
]
B4 = [1.478891]

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
