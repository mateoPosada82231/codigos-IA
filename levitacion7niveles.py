from machine import Pin, PWM, time_pulse_us
import time
import gc

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

# --- Gestión de memoria para CSV ---
data_log = []
MAX_LOGS = 1200

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

def medir_cm():
    global buf
    trig.off()
    time.sleep_us(2)
    trig.on()
    time.sleep_us(10)
    trig.off()
    dur = time_pulse_us(echo, 1, 30000)

    if dur < 0:
        if buf:
            return round(sorted(buf)[len(buf) // 2], 2)
        return -1.0

    d = dur * 0.034 / 2

    if d < SENSOR_MIN or d > SENSOR_MAX:
        if buf:
            return round(sorted(buf)[len(buf) // 2], 2)
        return -1.0

    # Rechazo de outliers
    if len(buf) >= 3:
        mediana = sorted(buf)[len(buf) // 2]
        if abs(d - mediana) > OUTLIER_THR:
            return round(mediana, 2)

    buf.append(d)
    if len(buf) > BUF_SIZE:
        buf.pop(0)
    return round(sorted(buf)[len(buf) // 2], 2)

print("=" * 60)
print("CONTROL DIFUSO PID — 9 niveles (con integral)")
print("=" * 60)

try:
    setpoint = float(input("Setpoint (cm, ej 20): ").strip())
except:
    setpoint = 20.0

pwm_actual = 400.0
error_ant = 0.0
deriv_f = 0.0
integral = 0.0
fan.duty(int(pwm_actual))
time.sleep(1)

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

        # 3. Funciones de Error
        e_niveles = [
            trapmf(error, -50, -50, -15, -8),
            trimf(error, -12, -8, -4),
            trimf(error, -6, -4, -1.5),
            trimf(error, -2.5, -1.0, 0),
            trimf(error, -1.0, 0.0, 1.0),
            trimf(error, 0, 1.0, 2.5),
            trimf(error, 1.5, 4, 6),
            trimf(error, 4, 8, 12),
            trapmf(error, 8, 15, 50, 50)
        ]

        # 4. Funciones de Derivada
        de_niveles = [
            trapmf(deriv_f, -80, -80, -25, -10),
            trimf(deriv_f, -20, -10, -3),
            trimf(deriv_f, -6, -3, 0),
            trimf(deriv_f, -1.5, 0, 1.5),
            trimf(deriv_f, 0, 3, 6),
            trimf(deriv_f, 3, 10, 20),
            trapmf(deriv_f, 10, 25, 80, 80)
        ]

        # 5. Inferencia Fuzzy (PD)
        numerador = 0.0
        denominador = 0.0
        for i in range(9):
            ei = e_niveles[i]
            if ei <= 0:
                continue
            for j in range(7):
                peso = min(ei, de_niveles[j])
                if peso > 0:
                    numerador += peso * FAM[i][j]
                    denominador += peso

        delta_fuzzy = (numerador / denominador) if denominador > 0 else 0.0

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