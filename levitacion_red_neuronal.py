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
PWM_MIN   = 170

# Sensor
SENSOR_MIN  = 3.0
SENSOR_MAX  = 40.0
BUF_SIZE    = 7
OUTLIER_THR = 5.0

# Filtros
ALFA_DERIV = 0.40       # EMA de derivada

# Integral
KI             = 0.10
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

# Parámetros de normalización de entradas [error, derivada, integral]
X_MEAN = [0.000000, 0.000000, 0.000000]
X_STD  = [1.000000, 1.000000, 1.000000]

# Parámetros de normalización de salida [delta_pwm]
Y_MEAN = 0.000000
Y_STD  = 1.000000

# Capa 1: 3 entradas → 10 neuronas (sigmoid)
W1 = [
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
]
B1 = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]

# Capa 2: 10 entradas → 6 neuronas (sigmoid)
W2 = [
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
]
B2 = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]

# Capa 3: 6 entradas → 1 salida (lineal)
W3 = [
    [0.000000],
    [0.000000],
    [0.000000],
    [0.000000],
    [0.000000],
    [0.000000],
]
B3 = [0.000000]

# =====================================================================
#  FORWARD PASS DE LA RED NEURONAL (sin numpy, compatible MicroPython)
# =====================================================================

def sigmoid(x):
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))

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
      2. Capa oculta 1 (3→10, sigmoid)
      3. Capa oculta 2 (10→6, sigmoid)
      4. Capa de salida (6→1, lineal)
      5. Desnormalizar salida
    """
    # 1. Normalizar entradas
    x = [
        (error    - X_MEAN[0]) / X_STD[0],
        (deriv_f  - X_MEAN[1]) / X_STD[1],
        (integral - X_MEAN[2]) / X_STD[2],
    ]

    # 2. Capa oculta 1: 3→10, sigmoid
    h1 = _dense(x, W1, B1)
    h1 = [sigmoid(v) for v in h1]

    # 3. Capa oculta 2: 10→6, sigmoid
    h2 = _dense(h1, W2, B2)
    h2 = [sigmoid(v) for v in h2]

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
print("=" * 60)

try:
    setpoint = float(input("Setpoint (cm, ej 20): ").strip())
except:
    setpoint = 20.0

pwm_actual = 400.0
error_ant  = 0.0
deriv_f    = 0.0
integral   = 0.0
fan.duty(int(pwm_actual))
time.sleep(1)

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
            integral += error * dt_real
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

        print("{:7.2f} | {:6.2f} | {:+7.2f} | {:+7.2f} | {:+7.2f} | {:+7.2f} | {:7.2f}".format(
            tiempo_actual, dist, error, deriv_f, integral, delta_pwm, pwm_actual))

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
            print("Guardado con exito en el ESP32.")
        except Exception as e:
            print("Error al guardar:", e)
