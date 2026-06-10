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
PWM_MAX   = 750
PWM_MIN   = 275
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

X_MEAN = [-0.862001, -0.364707, -5.148597]
X_STD  = [8.485796, 16.821253, 14.123256]
Y_MEAN = 0.608819
Y_STD  = 5.851730

W1 = [
    [-1.714415, 2.338630, 2.126723, 0.936850, 1.442208, -0.639045, -1.095611, -1.599906, -0.369056, -0.743584, 0.026041, 0.933889, 1.051787, 0.126054, 0.389421, -0.818113],
    [-0.233787, -1.045513, -0.565360, 1.409728, -0.014459, -0.210335, -1.867998, 0.329179, -0.562698, 0.873573, 0.093398, -2.687587, -0.670795, -0.808912, 1.293372, 1.795048],
    [-0.208823, -0.005865, -0.021882, -0.003940, 0.025312, -0.144151, 0.031279, -0.020349, -0.187464, 0.670898, -0.260932, -0.009930, -0.598806, 0.052139, -0.017126, -0.043265],
]
B1 = [0.127464, -1.023366, 0.105412, -2.064735, -1.389574, -0.740369, 0.067747, -0.843901, 0.629808, -0.521404, 0.078685, 0.598890, -0.063105, -0.420746, -0.954019, 0.987059]

W2 = [
    [0.602178, 0.191685, -0.602709, -0.316820, 0.476786, 0.894538, 0.307108, 0.040191, -0.451813, 0.192293, -0.185514, -0.741898],
    [-0.431140, 0.802249, -0.082603, -0.607329, -0.486831, -1.995290, 0.152124, 1.933392, 0.378723, -0.003626, 0.371297, 0.322213],
    [-1.246502, 0.368144, 0.603946, -0.082213, -0.299830, -1.015391, 0.567817, 0.285481, -1.034283, -0.116950, -0.170836, -0.052333],
    [-0.470027, 0.705503, -0.112368, 1.081712, 0.167604, -0.181040, 0.047154, -1.144288, -1.754395, 0.401119, -0.406056, -0.392406],
    [0.223832, -0.164403, 0.207595, -0.354604, 0.809278, -1.022870, -0.613870, -0.963621, -0.545099, 0.152365, 0.123048, -0.731605],
    [0.315057, 0.350575, -1.090710, -0.261258, -0.496218, -0.656046, -0.401101, 0.628765, -0.506812, 0.232207, -0.246956, 0.163969],
    [-0.738841, -0.100985, 1.046102, 1.018839, -0.454615, -0.241235, -0.152150, 0.543805, -0.509953, -0.060815, -0.171751, -0.357900],
    [0.153981, -0.039112, -1.177636, -0.489620, -0.899063, 0.591972, 0.826368, -1.171586, 0.261697, -0.150529, -0.210912, -0.120885],
    [0.000614, 0.467737, 0.451938, -0.740943, -0.357405, 0.151731, 0.552140, 0.688874, -0.012903, 0.131778, -0.020068, 0.237770],
    [0.075685, -0.054389, -0.015315, 0.077576, -0.326679, -0.156905, -0.002923, 0.146761, -0.971664, -0.189932, -0.172429, -0.294996],
    [-0.057663, 0.057164, 0.569137, 0.240502, 0.221046, -0.325176, 0.384716, 0.001624, -0.304380, 0.746523, -0.321119, 0.819053],
    [-0.498103, 0.102252, 0.074959, -0.279539, 0.420461, 1.063902, 0.540942, -1.272713, -0.284021, -0.148181, 0.186101, 0.269897],
    [-0.059013, -0.347179, -0.180529, -0.072002, 0.183512, -0.182157, 0.190025, -0.055495, -0.925328, 0.030553, -0.344176, -0.153621],
    [0.283606, -0.229907, -0.761843, -1.087708, -0.321906, 0.210850, 0.143607, 0.662642, 0.220316, 0.009378, 0.684230, -0.316734],
    [-0.153204, -0.337311, 0.321950, 1.246321, 0.070172, -0.383757, -0.169606, -1.118076, -0.189224, -0.379422, -0.109357, -0.468808],
    [0.615835, 1.097571, 0.867550, 0.902821, 0.629083, -0.187914, -0.079675, 0.070049, 0.799854, -0.383645, 0.015925, 0.099127],
]
B2 = [-0.481995, -0.073644, 0.489460, -0.225708, 0.315560, 0.799421, 0.153237, -0.153245, 0.305785, -0.245070, 0.231098, 0.419263]

W3 = [
    [-1.078155, 0.323875, -0.643416, 0.036224, -0.150389, 0.460846, 0.415110, 0.598522],
    [0.917032, -0.330367, 0.101774, -0.810049, -0.162919, -0.547030, 0.747372, -0.369507],
    [0.649963, -0.540954, -0.439201, 0.128661, 0.405194, 0.588314, -1.439978, 0.328847],
    [0.792662, -0.399019, -0.269110, 0.161873, 0.398144, -0.391977, 1.551175, -1.966124],
    [-0.618999, 0.223699, -0.832578, 0.137180, -0.581429, -0.522925, 0.359097, -0.233094],
    [-0.163499, -0.607433, 0.168234, 2.040050, -0.033215, 0.373574, 0.179073, -0.595949],
    [0.107262, -0.599011, 0.200212, 0.240151, 0.489276, -0.017304, -0.340637, -0.155981],
    [-0.240587, -0.103066, -0.026007, 0.304574, 0.640145, -0.356443, 2.214850, -0.316143],
    [0.128336, 0.140405, -0.335787, 1.194353, 0.354052, 0.765943, 1.200018, -0.468135],
    [-0.452973, -0.323954, 0.565278, 0.091254, -0.108875, -0.164159, -0.170949, 0.314146],
    [0.015087, 0.359215, 0.051871, 0.599408, 0.307847, 0.273660, 0.046790, -0.314472],
    [-0.730797, -0.368547, 0.346963, 0.627171, 0.230837, -0.465947, 0.487583, 0.047071],
]
B3 = [-0.370313, 0.352864, -0.043748, 0.751540, 0.132746, 0.145764, -0.289934, -0.471402]

W4 = [
    [0.616100],
    [0.822395],
    [-0.397861],
    [-0.983625],
    [0.495671],
    [0.564895],
    [-0.598091],
    [-0.841488],
]
B4 = [0.421894]

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
