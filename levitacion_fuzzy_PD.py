from machine import Pin, PWM, time_pulse_us
import time
import sys

# ============================================================
#  LEVITACIÓN DE PELOTA — Fuzzy PD v4
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
DT             = 0.2
ALFA_PWM       = 0.25
ALFA_DERIV     = 0.20
ZONA_MUERTA    = 1.5
PWM_MINIMO     = 200
PWM_MAXIMO     = 900
MAX_MUESTRAS   = 150

datos = []
distancia_deseada = 20.0

# ============================================================
#  SENSOR
# ============================================================
_buf = []

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
    global _buf
    d = medir_raw()
    if d < DISTANCIA_MIN or d > DISTANCIA_MAX:
        return -1.0
    _buf.append(d)
    if len(_buf) > 5:
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
    e1 = mb(error,   6,  10,  99,  99)
    e2 = mb(error,   1,   4,   8,  12)
    e3 = mb(error,  -5,  -1,   1,   5)
    e4 = mb(error, -12,  -8,  -4,  -1)
    e5 = mb(error, -99, -99, -10,  -6)

    d1 = mb(deriv,  2,   5,  99,  99)
    d2 = mb(deriv, -2,   0,   0,   2)
    d3 = mb(deriv, -99, -99, -5,  -2)

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
    sys.stdout.write("t,pos,des,error,deriv,duty,volt\n")
    for d in datos:
        t, pos, err, der, dty, vlt = d
        sys.stdout.write(str(t) + "," + str(pos) + "," +
                         str(distancia_deseada) + "," + str(err) +
                         "," + str(der) + "," + str(dty) +
                         "," + str(vlt) + "\n")

    errores = [abs(d[2]) for d in datos]
    pwms    = [d[4]      for d in datos]
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
sys.stdout.write("LEVITACION DE PELOTA v4\n")
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
sys.stdout.write("t(s) | pos | error | deriv | duty | volt | estado\n")
sys.stdout.write("-" * 55 + "\n")

# Variables
err_ant  = 0.0
der_fil  = 0.0
pwm_suav = float(PWM_BASE)
inv      = 0
duty     = PWM_BASE
t0       = time.time()

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
#  LOOP
# ============================================================
while True:
    try:
        t = round(time.time() - t0, 1)
        dist = medir()

        if dist < 0:
            inv += 1
            sys.stdout.write("Lectura invalida #" + str(inv) + "\n")
            if inv > 5:
                sys.stdout.write("Demasiadas lecturas invalidas\n")
                break
            time.sleep(DT)
            continue

        inv = 0
        error  = round(dist - distancia_deseada, 1)
        dr_raw = (error - err_ant) / DT
        der_fil = round(ALFA_DERIV * dr_raw + (1 - ALFA_DERIV) * der_fil, 2)
        err_ant = error

        if abs(error) < ZONA_MUERTA:
            pwm_r = pwm_suav
        else:
            pwm_r = fuzzy(error, der_fil, PWM_BASE)

        pwm_suav = ALFA_PWM * pwm_r + (1 - ALFA_PWM) * pwm_suav
        duty = int(max(PWM_MINIMO, min(PWM_MAXIMO, pwm_suav)))
        pwm.duty(duty)

        volt = round((duty / 1023) * VOLTAJE_FUENTE, 1)

        if len(datos) < MAX_MUESTRAS:
            datos.append([t, dist, error, der_fil, duty, volt])

        if abs(error) <= ZONA_MUERTA:
            estado = "EN PUNTO"
        elif error > 0:
            estado = "ABAJO"
        else:
            estado = "ARRIBA"

        sys.stdout.write(str(t) + "s | " +
                         str(dist) + "cm | " +
                         str(error) + "cm | " +
                         str(der_fil) + " | " +
                         str(duty) + " | " +
                         str(volt) + "V | " +
                         estado + "\n")

        time.sleep(DT)

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