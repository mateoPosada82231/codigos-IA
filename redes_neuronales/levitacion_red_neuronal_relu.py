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
PWM_MAX   = 600
PWM_MIN   = 235
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

X_MEAN = [1.219968, 0.073431, -3.187884]
X_STD  = [10.202658, 18.457283, 14.837739]
Y_MEAN = 1.461528
Y_STD  = 6.315547

W1 = [
    [-0.379376, -0.353437, -0.301129, 0.960029, 2.037536, -0.630553, 0.528909, 0.692609, 0.142779, -0.951157, -0.914095, 1.557949, 0.892424, 0.410779, -0.373645, 2.162074],
    [2.297158, -0.288119, 0.324557, 0.263527, 0.149707, 1.383334, 0.110472, -0.609276, 0.723884, -1.410197, 0.216800, 0.130799, 0.791307, 1.487237, -0.351295, -1.185016],
    [-0.119124, -0.311783, -0.873501, 0.611565, 0.043126, 0.486254, 0.529675, 0.295895, -0.376782, 1.243600, 0.145764, -0.069426, 0.118133, -0.110808, -0.021899, -0.009561],
]
B1 = [-0.424851, 0.561474, 0.567962, 0.402329, -0.066380, -0.150355, -0.297107, 0.623203, -0.174173, 0.073945, -0.121485, -0.540823, 0.492380, -0.218505, -0.786314, 0.228206]

W2 = [
    [0.184994, 0.074554, 0.102252, -0.088095, -1.323265, -0.787618, -0.160600, 0.883870, 0.729391, -0.183881, 0.200412, -0.511791],
    [-0.164578, 0.499621, 0.815851, -0.611263, -0.169805, 0.790456, 0.423167, -0.462759, 0.854094, -0.064267, -0.196209, -0.210085],
    [0.370079, 0.088198, 0.419888, 0.192852, 0.257635, -0.097286, -0.390745, 0.307861, -0.287948, -0.716244, 0.174915, 0.207441],
    [0.177243, 0.072831, -0.492214, 0.268033, 0.313516, 0.224942, -0.091718, 0.101224, -0.126705, -0.542930, -0.111466, 0.098347],
    [-0.741916, 0.997871, -0.251671, 0.223618, 0.205150, -0.463436, 0.092476, -0.124487, -0.567644, 0.026534, -0.549185, -1.139780],
    [0.189851, 0.509378, -0.061521, 0.531401, 0.408596, 0.010736, -0.222630, 0.268562, 0.394688, -0.023727, -0.123419, -0.002207],
    [-0.165321, -0.230763, 0.232221, 0.092312, 0.350760, -0.179654, -0.434961, 0.472824, -0.512216, -0.114141, 0.324514, -0.002557],
    [-0.267389, -0.148163, -0.134643, -0.614209, -0.364136, -0.032090, 0.370937, -0.645184, 0.182247, -0.584644, 0.229753, 0.073298],
    [-0.050629, 0.155649, -0.222878, 0.557285, 0.311200, -0.208727, -0.663205, -0.684739, -0.241988, 0.386886, 0.162315, 0.014856],
    [0.220792, -0.130747, 0.066051, -0.252821, 0.068401, 0.048859, -0.694731, 0.692910, 0.264089, -0.756307, -0.517609, -0.057253],
    [0.122347, 0.036907, -0.273103, 0.854607, 0.241765, 0.075644, 0.134880, -0.483709, 0.131238, -0.092238, 0.238606, -0.154173],
    [-1.098044, 0.340663, 0.789718, 0.203440, -0.078073, 0.266414, 0.192332, 0.569911, -0.221428, -0.311788, -0.874741, -0.885214],
    [0.471828, -0.611339, 0.176220, -0.195527, 0.780243, -0.241669, -0.029639, -0.219536, -0.258874, 0.306491, 0.526287, 0.528353],
    [-0.404942, -0.082799, -0.841582, -0.193458, 1.221171, -0.036833, -0.386075, -0.209728, -0.021308, -0.003920, -0.323739, 0.551726],
    [-0.636812, 0.112030, -0.777552, -0.467957, 0.547487, -0.159253, 0.024926, -0.081324, -0.460292, 0.280659, -0.053008, -0.593637],
    [-0.419988, -0.660797, 0.276095, -0.543701, -0.794881, -0.242918, -0.521027, 0.147408, -0.535555, 0.233133, -0.428706, 0.243934],
]
B2 = [0.054695, -0.997937, 0.048387, 0.055397, 0.540661, -0.632263, -0.090619, 0.186502, -0.706912, -0.028580, 0.865332, 1.547097]

W3 = [
    [0.482197, 0.303908, -0.915021, -0.538876, 0.221509, 0.245931, 0.574544, 0.087671],
    [0.415989, -0.307253, -0.348449, -0.974786, 0.371700, 0.652570, -0.471965, -0.254331],
    [-0.306478, 0.076890, -0.027644, -0.388281, 0.632606, -0.119812, 0.364148, 0.260433],
    [-0.548126, -0.174920, -0.079583, 0.271141, -0.788786, 0.258630, -0.883652, -0.378811],
    [-0.064949, -0.130251, 0.377519, 0.471412, -0.443946, -0.494284, -0.504814, 0.522375],
    [0.385670, -0.247064, 0.120677, -0.707885, 0.338599, -0.806742, -0.528376, 0.044170],
    [-0.323337, 0.102952, -0.182153, -0.207689, 0.002696, -0.453569, 0.563565, 0.073300],
    [0.130036, 0.549782, -0.020548, 0.145023, 0.243666, -0.336836, 0.408573, -0.785244],
    [0.598488, -0.260929, -0.265372, -0.626489, 0.203544, -0.929892, 0.727746, -0.142881],
    [0.168382, -0.731180, 0.138247, -0.297926, 0.749926, -0.332378, -0.047454, 0.746094],
    [-0.614309, -0.050195, 0.798655, -0.313608, 0.475884, 0.221798, 0.207248, 0.333763],
    [0.850268, -0.083593, -0.984428, 0.943655, 0.265076, -0.151862, 0.264131, -0.938660],
]
B3 = [0.005990, 0.025889, 0.128568, 0.077574, 0.924058, -0.016398, 0.016373, -0.589371]

W4 = [
    [-0.649126],
    [0.043481],
    [0.850642],
    [0.488078],
    [-0.873626],
    [0.255833],
    [0.782369],
    [1.381240],
]
B4 = [0.757200]

W2 = [
    [0.184994, 0.074554, 0.102252, -0.088095, -1.323265, -0.787618, -0.160600, 0.883870, 0.729391, -0.183881, 0.200412, -0.511791],
    [-0.164578, 0.499621, 0.815851, -0.611263, -0.169805, 0.790456, 0.423167, -0.462759, 0.854094, -0.064267, -0.196209, -0.210085],
    [0.370079, 0.088198, 0.419888, 0.192852, 0.257635, -0.097286, -0.390745, 0.307861, -0.287948, -0.716244, 0.174915, 0.207441],
    [0.177243, 0.072831, -0.492214, 0.268033, 0.313516, 0.224942, -0.091718, 0.101224, -0.126705, -0.542930, -0.111466, 0.098347],
    [-0.741916, 0.997871, -0.251671, 0.223618, 0.205150, -0.463436, 0.092476, -0.124487, -0.567644, 0.026534, -0.549185, -1.139780],
    [0.189851, 0.509378, -0.061521, 0.531401, 0.408596, 0.010736, -0.222630, 0.268562, 0.394688, -0.023727, -0.123419, -0.002207],
    [-0.165321, -0.230763, 0.232221, 0.092312, 0.350760, -0.179654, -0.434961, 0.472824, -0.512216, -0.114141, 0.324514, -0.002557],
    [-0.267389, -0.148163, -0.134643, -0.614209, -0.364136, -0.032090, 0.370937, -0.645184, 0.182247, -0.584644, 0.229753, 0.073298],
    [-0.050629, 0.155649, -0.222878, 0.557285, 0.311200, -0.208727, -0.663205, -0.684739, -0.241988, 0.386886, 0.162315, 0.014856],
    [0.220792, -0.130747, 0.066051, -0.252821, 0.068401, 0.048859, -0.694731, 0.692910, 0.264089, -0.756307, -0.517609, -0.057253],
    [0.122347, 0.036907, -0.273103, 0.854607, 0.241765, 0.075644, 0.134880, -0.483709, 0.131238, -0.092238, 0.238606, -0.154173],
    [-1.098044, 0.340663, 0.789718, 0.203440, -0.078073, 0.266414, 0.192332, 0.569911, -0.221428, -0.311788, -0.874741, -0.885214],
    [0.471828, -0.611339, 0.176220, -0.195527, 0.780243, -0.241669, -0.029639, -0.219536, -0.258874, 0.306491, 0.526287, 0.528353],
    [-0.404942, -0.082799, -0.841582, -0.193458, 1.221171, -0.036833, -0.386075, -0.209728, -0.021308, -0.003920, -0.323739, 0.551726],
    [-0.636812, 0.112030, -0.777552, -0.467957, 0.547487, -0.159253, 0.024926, -0.081324, -0.460292, 0.280659, -0.053008, -0.593637],
    [-0.419988, -0.660797, 0.276095, -0.543701, -0.794881, -0.242918, -0.521027, 0.147408, -0.535555, 0.233133, -0.428706, 0.243934],
]
B2 = [0.054695, -0.997937, 0.048387, 0.055397, 0.540661, -0.632263, -0.090619, 0.186502, -0.706912, -0.028580, 0.865332, 1.547097]

W3 = [
    [0.482197, 0.303908, -0.915021, -0.538876, 0.221509, 0.245931, 0.574544, 0.087671],
    [0.415989, -0.307253, -0.348449, -0.974786, 0.371700, 0.652570, -0.471965, -0.254331],
    [-0.306478, 0.076890, -0.027644, -0.388281, 0.632606, -0.119812, 0.364148, 0.260433],
    [-0.548126, -0.174920, -0.079583, 0.271141, -0.788786, 0.258630, -0.883652, -0.378811],
    [-0.064949, -0.130251, 0.377519, 0.471412, -0.443946, -0.494284, -0.504814, 0.522375],
    [0.385670, -0.247064, 0.120677, -0.707885, 0.338599, -0.806742, -0.528376, 0.044170],
    [-0.323337, 0.102952, -0.182153, -0.207689, 0.002696, -0.453569, 0.563565, 0.073300],
    [0.130036, 0.549782, -0.020548, 0.145023, 0.243666, -0.336836, 0.408573, -0.785244],
    [0.598488, -0.260929, -0.265372, -0.626489, 0.203544, -0.929892, 0.727746, -0.142881],
    [0.168382, -0.731180, 0.138247, -0.297926, 0.749926, -0.332378, -0.047454, 0.746094],
    [-0.614309, -0.050195, 0.798655, -0.313608, 0.475884, 0.221798, 0.207248, 0.333763],
    [0.850268, -0.083593, -0.984428, 0.943655, 0.265076, -0.151862, 0.264131, -0.938660],
]
B3 = [0.005990, 0.025889, 0.128568, 0.077574, 0.924058, -0.016398, 0.016373, -0.589371]

W4 = [
    [-0.649126],
    [0.043481],
    [0.850642],
    [0.488078],
    [-0.873626],
    [0.255833],
    [0.782369],
    [1.381240],
]
B4 = [0.757200]

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
