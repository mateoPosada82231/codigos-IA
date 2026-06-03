from machine import Pin, PWM, time_pulse_us
import time
import gc
import array

# =====================================================================
#  LEVITACIÓN DE PELOTA — Fuzzy PID con 9 niveles de error
#  Fuzzy PD + Integral con anti-windup + rechazo de outliers
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
PWM_MIN   = 200
ELEVACION_PWM = PWM_MAX
ELEVACION_SEGUNDOS = 1.8
DEFUZZ_METHOD = "centroid"

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
rechazos = 0           # Contador de lecturas rechazadas consecutivas
MAX_RECHAZOS = 5        # Tras este número, limpiar búfer para re-adaptar

# --- Gestión de memoria para CSV (ring buffer en array, sin fragmentar heap) ---
MAX_LOGS   = 300
_LOG_N     = 8
_log_buf   = array.array('f', [0.0] * (MAX_LOGS * _LOG_N))
_log_idx   = 0
_log_count = 0

# --- Funciones de pertenencia ---
def trapmf(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    if b < x <= c:
        return 1.0
    if c < x < d:
        return (d - x) / (d - c) if d != c else 1.0
    return 0.0

def trimf(x, a, b, c):
    return trapmf(x, a, b, b, c)

def defuzzify_singletons(reglas, metodo):
    if not reglas:
        return 0.0

    if metodo == "bisector":
        reglas_ordenadas = sorted(reglas, key=lambda r: r[1])
        total = sum(peso for peso, _ in reglas_ordenadas)
        if total <= 0:
            return 0.0
        acumulado = 0.0
        mitad = total / 2.0
        for peso, valor in reglas_ordenadas:
            acumulado += peso
            if acumulado >= mitad:
                return valor
        return reglas_ordenadas[-1][1]

    if metodo == "mom":
        max_peso = max(peso for peso, _ in reglas)
        if max_peso <= 0:
            return 0.0
        valores_max = [valor for peso, valor in reglas if abs(peso - max_peso) < 1e-6]
        return sum(valores_max) / len(valores_max)

    # Centroide para singletons (promedio ponderado)
    numerador = sum(peso * valor for peso, valor in reglas)
    denominador = sum(peso for peso, _ in reglas)
    return (numerador / denominador) if denominador > 0 else 0.0

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

print("=" * 60)
print("CONTROL DIFUSO PID — 9 niveles (con integral)")
print("Método de desfusificación:", DEFUZZ_METHOD)
print("=" * 60)

try:
    setpoint = float(input("Setpoint (cm, ej 20): ").strip())
except:
    setpoint = 20.0

pwm_actual = float(ELEVACION_PWM)
error_ant = 0.0
deriv_f = 0.0
integral = 0.0
fan.duty(int(ELEVACION_PWM))
print("Elevacion inicial al maximo PWM (3 s)...")
time.sleep(3.0)
print("Bajando suavemente hacia zona de control...")
_pwm_ramp_fin = float(PWM_MIN + (PWM_MAX - PWM_MIN) * 0.45)
_pasos_ramp   = 60   # 60 x 50 ms = 3 s de rampa
_delta_ramp   = (float(PWM_MAX) - _pwm_ramp_fin) / _pasos_ramp
for _i in range(_pasos_ramp):
    pwm_actual = float(PWM_MAX) - _delta_ramp * _i
    fan.duty(int(pwm_actual))
    time.sleep_ms(50)
pwm_actual = _pwm_ramp_fin
fan.duty(int(pwm_actual))
print("Rampa completada. Iniciando control...")

# Deltas PWM
NV_out = -6.0
NB_out = -3.0
NM_out = -1.5
NS_out = -0.5
Z_out  =  0.0
PS_out =  0.8
PM_out =  2.5
PB_out =  6.0
PV_out =  18.0

# Matriz FAM Asimétrica
FAM = [
    [NV_out, NV_out, NB_out, NM_out, NS_out, Z_out,  Z_out ],
    [NV_out, NB_out, NM_out, NS_out, Z_out,  Z_out,  PS_out],
    [NB_out, NM_out, NS_out, NS_out, Z_out,  PS_out, PM_out],
    [NM_out, NS_out, NS_out, Z_out,  Z_out,  PS_out, PM_out],
    [NM_out, NS_out, Z_out,  Z_out,  Z_out,  PS_out, PM_out],
    [NM_out, NS_out, Z_out,  Z_out,  PS_out, PS_out, PM_out],
    [NM_out, NS_out, Z_out,  PS_out, PS_out, PM_out, PB_out],
    [NS_out, Z_out,  Z_out,  PS_out, PM_out, PB_out, PV_out],
    [Z_out,  Z_out,  PS_out, PM_out, PB_out, PV_out, PV_out]
]

t_inicio = time.ticks_ms()
t_anterior = time.ticks_ms()
ciclos = 0

gc.collect()

try:
    while True:
        # Medir dt real
        t_ahora = time.ticks_ms()
        dt_real = time.ticks_diff(t_ahora, t_anterior) / 1000.0
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
        error = dist - setpoint
        deriv = (error - error_ant) / dt_real
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

        # 3. Funciones de Error (trapecios para mayor precisión)
        e_niveles = [
            trapmf(error, -50, -50, -15, -8),       # NV
            trapmf(error, -12, -9, -7, -4),          # NB
            trapmf(error, -6, -4.5, -3.5, -1.5),    # NM
            trapmf(error, -2.5, -1.5, -0.5, 0),     # NS
            trapmf(error, -1.0, -0.3, 0.3, 1.0),    # Z
            trapmf(error, 0, 0.5, 1.5, 2.5),        # PS
            trapmf(error, 1.5, 3, 5, 6),             # PM
            trapmf(error, 4, 6, 10, 12),             # PB
            trapmf(error, 8, 15, 50, 50)             # PV
        ]

        # 4. Funciones de Derivada (trapecios para mayor precisión)
        de_niveles = [
            trapmf(deriv_f, -80, -80, -25, -10),     # NV
            trapmf(deriv_f, -20, -12, -8, -3),       # NB
            trapmf(deriv_f, -6, -4, -2, 0),          # NS
            trapmf(deriv_f, -1.5, -0.5, 0.5, 1.5),  # Z
            trapmf(deriv_f, 0, 2, 4, 6),             # PS
            trapmf(deriv_f, 3, 7, 13, 20),           # PB
            trapmf(deriv_f, 10, 25, 80, 80)          # PV
        ]

        # 5. Inferencia Fuzzy (PD)
        reglas = []
        for i in range(9):
            ei = e_niveles[i]
            if ei <= 0:
                continue
            for j in range(7):
                peso = min(ei, de_niveles[j])
                if peso > 0:
                    reglas.append((peso, FAM[i][j]))

        delta_fuzzy = defuzzify_singletons(reglas, DEFUZZ_METHOD)

        # 6. Contribución integral
        delta_integral = KI * integral
        delta_pwm = delta_fuzzy + delta_integral

        # 7. Aplicar cambios
        pwm_actual += delta_pwm
        if pwm_actual > PWM_MAX:
            pwm_actual = float(PWM_MAX)
        if pwm_actual < PWM_MIN:
            pwm_actual = float(PWM_MIN)
        fan.duty(int(pwm_actual))

        # Logging protegido
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

    resp = input("Guardar {} datos en CSV? (s/n): ".format(_log_count)).strip().lower()
    if resp == 's':
        try:
            _start = (_log_idx - _log_count) % MAX_LOGS if _log_count == MAX_LOGS else 0
            with open("datos_levitacion.csv", "w") as f:
                f.write("tiempo,distancia,setpoint,error,derivada,integral,delta_pwm,pwm\n")
                for i in range(_log_count):
                    _b = ((_start + i) % MAX_LOGS) * _LOG_N
                    f.write("{:.3f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(
                        _log_buf[_b], _log_buf[_b+1], _log_buf[_b+2], _log_buf[_b+3],
                        _log_buf[_b+4], _log_buf[_b+5], _log_buf[_b+6], _log_buf[_b+7]))
            print("Guardado con exito en el ESP32.")
        except Exception as e:
            print("Error al guardar:", e)
