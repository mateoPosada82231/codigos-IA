from machine import Pin, PWM, time_pulse_us
import time
import sys

# ============================================================
#  LEVITACIÓN DE PELOTA — Fuzzy PID v5
#  Control: Fuzzy PD + Integral con anti-windup
# ============================================================
#
#  Mejoras respecto a v4:
#  1. Integral con anti-windup  → elimina error en estado estable
#  2. Sin zona muerta (impedía convergencia al setpoint)
#  3. dt real medido  → derivada e integral precisas
#  4. Rechazo de outliers en sensor
#  5. Limitador de cambio adaptativo
#  6. Compensación de tiempo de loop
# ============================================================

# -------- Pines --------
TRIG_PIN = 27
ECHO_PIN = 26
FAN_PIN  = 14

trig = Pin(TRIG_PIN, Pin.OUT)
echo = Pin(ECHO_PIN, Pin.IN)
pwm  = PWM(Pin(FAN_PIN), freq=25000)

# -------- Configuración --------
TUBO_CM        = 40.0
DISTANCIA_MIN  = 2.0
DISTANCIA_MAX  = 38.0
VOLTAJE_FUENTE = 12.0
DT_TARGET      = 0.2        # Período objetivo del loop
ALFA_PWM       = 0.30       # EMA del PWM (más responsivo que 0.25)
ALFA_DERIV     = 0.25       # EMA de derivada
PWM_MINIMO     = 170
PWM_MAXIMO     = 900
MAX_MUESTRAS   = 150

# Integral (pieza clave para estabilización)
KI             = 0.10       # Ganancia integral
INTEGRAL_MAX   = 60.0       # Anti-windup
INTEGRAL_DECAY = 0.998      # Decaimiento leve

# Limitador adaptativo
MAX_CAMBIO_CERCA = 30       # PWM/ciclo cuando cerca del setpoint
MAX_CAMBIO_LEJOS = 80       # PWM/ciclo cuando lejos
UMBRAL_CERCA     = 3.0      # cm

# Sensor
BUF_SIZE    = 7
OUTLIER_THR = 5.0

datos = []
distancia_deseada = 20.0

# ============================================================
#  SENSOR con mediana + rechazo de outliers
# ============================================================
_buf = []
_rechazos = 0           # Contador de lecturas rechazadas consecutivas
MAX_RECHAZOS = 5        # Tras este número, limpiar búfer para re-adaptar

def medir_raw():
    trig.off()
    time.sleep_us(2)
    trig.on()
    time.sleep_us(10)
    trig.off()
    dur = time_pulse_us(echo, 1, 30000)
    if dur < 0:
        return -1.0
    return round(dur * 0.034 / 2, 1)

def medir():
    global _buf, _rechazos
    d = medir_raw()
    if d < DISTANCIA_MIN or d > DISTANCIA_MAX:
        _rechazos += 1
        if _rechazos >= MAX_RECHAZOS:
            _buf = []
            _rechazos = 0
        if _buf:
            return round(sorted(_buf)[len(_buf) // 2], 1)
        return -1.0

    # Rechazo de outliers
    if len(_buf) >= 3:
        mediana = sorted(_buf)[len(_buf) // 2]
        if abs(d - mediana) > OUTLIER_THR:
            _rechazos += 1
            if _rechazos >= MAX_RECHAZOS:
                _buf = [d]
                _rechazos = 0
                return round(d, 1)
            return round(mediana, 1)

    _rechazos = 0
    _buf.append(d)
    if len(_buf) > BUF_SIZE:
        _buf.pop(0)
    if len(_buf) >= 3:
        s = sorted(_buf)
        return round(s[len(s) // 2], 1)
    return round(sum(_buf) / len(_buf), 1)

# ============================================================
#  CALIBRACIÓN
# ============================================================
def calibrar():
    sys.stdout.write("\nCALIBRACION: pon la pelota en el fondo\n")
    sys.stdout.write("Iniciando en 3 segundos...\n")
    time.sleep(3)
    sys.stdout.write("Subiendo PWM...\n")

    p = 150
    while p <= 800:
        pwm.duty(p)
        time.sleep(0.3)
        d = medir_raw()
        v = round((p / 1023) * VOLTAJE_FUENTE, 1)
        if d < 0:
            sys.stdout.write("PWM " + str(p) + " (" + str(v) + "V) -> sin lectura\n")
        else:
            sys.stdout.write("PWM " + str(p) + " (" + str(v) + "V) -> " + str(d) + "cm\n")
        if 0 < d < DISTANCIA_MAX - 5:
            sys.stdout.write("PWM base encontrado: " + str(p) + "\n")
            time.sleep(1)
            return p
        p += 15

    sys.stdout.write("No se detecto movimiento, usando 350\n")
    return 350

# ============================================================
#  MEMBRESÍA
# ============================================================
def mb(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif x < b:
        return (x - a) / (b - a)
    else:
        return (d - x) / (d - c)

# ============================================================
#  FUZZY PD
# ============================================================
def fuzzy(error, deriv, base):
    # Error: 5 niveles
    e1 = mb(error,   6,  10,  99,  99)   # Muy abajo
    e2 = mb(error,   1,   4,   8,  12)   # Abajo
    e3 = mb(error,  -3,  -0.5, 0.5, 3)   # En punto (zona más estrecha)
    e4 = mb(error, -12,  -8,  -4,  -1)   # Arriba
    e5 = mb(error, -99, -99, -10,  -6)   # Muy arriba

    # Derivada: 3 niveles
    d1 = mb(deriv,  2,   5,  99,  99)    # Cayendo
    d2 = mb(deriv, -2,  -0.5, 0.5,  2)    # Quieta (trapecio)
    d3 = mb(deriv, -99, -99, -5,  -2)    # Subiendo

    # Salidas como offsets del base
    s5 = min(base + 320, PWM_MAXIMO)
    s4 = min(base + 200, PWM_MAXIMO)
    s3 = base + 80
    s2 = max(base - 40,  PWM_MINIMO)
    s1 = max(base - 120, PWM_MINIMO)

    r = [
        (min(e1, d3), s4), (min(e1, d2), s5), (min(e1, d1), s5),
        (min(e2, d3), s3), (min(e2, d2), s4), (min(e2, d1), s5),
        (min(e3, d3), s2), (min(e3, d2), s3), (min(e3, d1), s4),
        (min(e4, d3), s1), (min(e4, d2), s2), (min(e4, d1), s3),
        (min(e5, d3), s1), (min(e5, d2), s1), (min(e5, d1), s2),
    ]

    num = sum(a * s for a, s in r)
    den = sum(a     for a, _ in r)
    if den == 0:
        return s3
    return int(num / den)

# ============================================================
#  ESTADÍSTICAS
# ============================================================
def stats():
    global distancia_deseada
    if not datos:
        return
    sys.stdout.write("\n--- DATOS ---\n")
    sys.stdout.write("t,pos,des,error,deriv,integral,duty,volt\n")
    for d in datos:
        t, pos, err, der, intg, dty, vlt = d
        sys.stdout.write(str(t) + "," + str(pos) + "," +
                         str(distancia_deseada) + "," + str(err) +
                         "," + str(der) + "," + str(intg) +
                         "," + str(dty) + "," + str(vlt) + "\n")

    errores = [abs(d[2]) for d in datos]
    pwms    = [d[5]      for d in datos]
    pos     = [d[1]      for d in datos]
    n       = len(datos)

    sys.stdout.write("\n--- ESTADISTICAS ---\n")
    sys.stdout.write("Muestras: "          + str(n) + "\n")
    sys.stdout.write("Tiempo: "            + str(datos[-1][0]) + "s\n")
    sys.stdout.write("Error prom: "        + str(round(sum(errores)/n, 2)) + "cm\n")
    sys.stdout.write("Error max: "         + str(round(max(errores), 2))   + "cm\n")
    sys.stdout.write("Error min: "         + str(round(min(errores), 2))   + "cm\n")
    sys.stdout.write("Posicion prom: "     + str(round(sum(pos)/n, 2))     + "cm\n")
    sys.stdout.write("PWM prom: "          + str(int(sum(pwms)/n))         + "\n")

    e1 = sum(1 for e in errores if e <= 1.0)
    e2 = sum(1 for e in errores if e <= 2.0)
    e3 = sum(1 for e in errores if e <= 3.0)
    sys.stdout.write("En +-1cm: " + str(e1) + "/" + str(n) + " (" + str(100*e1//n) + "%)\n")
    sys.stdout.write("En +-2cm: " + str(e2) + "/" + str(n) + " (" + str(100*e2//n) + "%)\n")
    sys.stdout.write("En +-3cm: " + str(e3) + "/" + str(n) + " (" + str(100*e3//n) + "%)\n")

# ============================================================
#  INICIO
# ============================================================
sys.stdout.write("LEVITACION DE PELOTA v5 (Fuzzy PID)\n")
sys.stdout.write("Fuente: " + str(VOLTAJE_FUENTE) + "V\n")
sys.stdout.write("Rango sensor: " + str(DISTANCIA_MIN) + " a " + str(DISTANCIA_MAX) + "cm\n\n")

# Verificar sensor
sys.stdout.write("Verificando sensor...\n")
ok = 0
for _ in range(5):
    d = medir_raw()
    sys.stdout.write("  " + str(d) + "cm\n")
    if DISTANCIA_MIN <= d <= DISTANCIA_MAX:
        ok += 1
    time.sleep(0.3)
sys.stdout.write("Lecturas validas: " + str(ok) + "/5\n")

# Calibración
sys.stdout.write("\nCalibracion automatica? (s/n): ")
if input().strip().lower() == "s":
    PWM_BASE = calibrar()
else:
    PWM_BASE = 350
sys.stdout.write("PWM base: " + str(PWM_BASE) + "\n")

# Setpoint
lim_inf = int(DISTANCIA_MIN + 2)
lim_sup = int(DISTANCIA_MAX - 2)
while True:
    try:
        sys.stdout.write("Distancia deseada (" + str(lim_inf) + "-" + str(lim_sup) + "cm): ")
        distancia_deseada = float(input().strip())
        if lim_inf <= distancia_deseada <= lim_sup:
            break
        sys.stdout.write("Fuera de rango\n")
    except:
        sys.stdout.write("Numero invalido\n")

sys.stdout.write("Setpoint: " + str(distancia_deseada) + "cm\n")
sys.stdout.write("Ctrl+C para detener y ver datos\n\n")
sys.stdout.write("t(s) | pos | error | deriv | integ | duty | volt | estado\n")
sys.stdout.write("-" * 65 + "\n")

# Variables de estado
err_ant   = 0.0
der_fil   = 0.0
integral  = 0.0       # NUEVO: acumulador integral
pwm_suav  = float(PWM_BASE)
inv       = 0
duty      = PWM_BASE
t0        = time.time()
t_ant_ms  = time.ticks_ms()  # Para medir dt real

# Arranque progresivo
sys.stdout.write("Arranque...\n")
p = 150
while p < PWM_BASE:
    pwm.duty(p)
    time.sleep(0.05)
    p += 20
pwm.duty(PWM_BASE)
time.sleep(1.5)

# ============================================================
#  LOOP PRINCIPAL
# ============================================================
while True:
    try:
        # Medir dt real
        t_ahora_ms = time.ticks_ms()
        dt_real = time.ticks_diff(t_ahora_ms, t_ant_ms) / 1000.0
        t_ant_ms = t_ahora_ms
        if dt_real < 0.01:
            dt_real = 0.01
        elif dt_real > 1.0:
            dt_real = 1.0

        t = round(time.time() - t0, 1)
        dist = medir()

        if dist < 0:
            inv += 1
            sys.stdout.write("Lectura invalida #" + str(inv) + "\n")
            if inv > 5:
                sys.stdout.write("Demasiadas lecturas invalidas\n")
                break
            time.sleep(DT_TARGET)
            continue

        inv = 0
        error  = round(dist - distancia_deseada, 1)
        dr_raw = (error - err_ant) / dt_real
        der_fil = round(ALFA_DERIV * dr_raw + (1 - ALFA_DERIV) * der_fil, 2)
        err_ant = error

        # Integral con anti-windup (NUEVO)
        if abs(error) < 10.0:
            integral += error * dt_real
            integral *= INTEGRAL_DECAY
            if integral > INTEGRAL_MAX:
                integral = INTEGRAL_MAX
            elif integral < -INTEGRAL_MAX:
                integral = -INTEGRAL_MAX

        # Fuzzy PD + Integral (sin zona muerta — la integral converge sola)
        pwm_fuzzy = fuzzy(error, der_fil, PWM_BASE)
        pwm_r = pwm_fuzzy + KI * integral

        # Suavizado EMA
        pwm_suav = ALFA_PWM * pwm_r + (1 - ALFA_PWM) * pwm_suav

        # Limitador adaptativo
        if abs(error) > UMBRAL_CERCA:
            max_cambio = MAX_CAMBIO_LEJOS
        else:
            max_cambio = MAX_CAMBIO_CERCA
        cambio = pwm_suav - duty
        cambio = max(-max_cambio, min(max_cambio, cambio))
        pwm_suav = max(float(PWM_MINIMO), min(float(PWM_MAXIMO), pwm_suav))
        duty = int(max(PWM_MINIMO, min(PWM_MAXIMO, duty + cambio)))

        pwm.duty(duty)

        volt = round((duty / 1023) * VOLTAJE_FUENTE, 1)

        if len(datos) < MAX_MUESTRAS:
            datos.append([t, dist, error, der_fil, integral, duty, volt])

        if abs(error) <= 1.0:
            estado = "EN EL PUNTO"
        elif error > 0:
            estado = "ABAJO"
        else:
            estado = "ARRIBA"

        sys.stdout.write(str(t) + "s | " +
                         str(dist) + "cm | " +
                         str(error) + "cm | " +
                         str(der_fil) + " | " +
                         str(round(integral, 1)) + " | " +
                         str(duty) + " | " +
                         str(volt) + "V | " +
                         estado + "\n")

        # Compensar tiempo de loop
        transcurrido = time.ticks_diff(time.ticks_ms(), t_ahora_ms) / 1000.0
        espera = DT_TARGET - transcurrido
        if espera > 0:
            time.sleep(espera)

    except KeyboardInterrupt:
        sys.stdout.write("\nDeteniendo...\n")
        p = duty
        while p > 150:
            pwm.duty(p)
            time.sleep(0.04)
            p -= 25
        pwm.duty(0)
        pwm.deinit()
        stats()
        break