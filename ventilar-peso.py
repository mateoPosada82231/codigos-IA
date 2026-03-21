from machine import Pin, PWM, time_pulse_us
import time
import gc

# =====================================================================
#  LEVITACIÓN DE PELOTA — Control Difuso PID (Fuzzy PD + Integral)
#  Para pelota de icopor ~0.5 g en tubo con ventilador 12 V / ESP32
# =====================================================================
#
#  Mejoras clave respecto a la versión anterior (solo Fuzzy PD):
#
#  1. INTEGRAL con anti-windup  →  elimina error en estado estable.
#     Sin integral, el PWM nunca converge al valor exacto que sostiene
#     la pelota en el setpoint; siempre queda un offset residual.
#
#  2. SIN zona muerta  →  la zona muerta impedía corregir errores
#     pequeños. Ahora el sistema fuzzy + integral se encarga de todo.
#
#  3. dt real medido  →  la derivada y la integral usan el tiempo
#     real transcurrido, no un DT fijo que puede ser inexacto.
#
#  4. Rechazo de outliers en el sensor  →  lecturas que difieren
#     demasiado de la mediana actual se descartan.
#
#  5. Limitador de cambio adaptativo  →  permite correcciones rápidas
#     cuando la pelota está lejos, pero finas cuando está cerca.
#
#  6. Compensación de tiempo de loop  →  el sleep se ajusta para
#     mantener la frecuencia objetivo de 20 Hz.
# =====================================================================

# ---- Pines ----
TRIG_PIN = 27
ECHO_PIN = 26
FAN_PIN  = 14

trig = Pin(TRIG_PIN, Pin.OUT)
echo = Pin(ECHO_PIN, Pin.IN)
fan  = PWM(Pin(FAN_PIN), freq=25000, duty=0)

# ---- Parámetros del controlador ----
DT_TARGET = 0.05        # Período objetivo del loop (20 Hz)
PWM_MAX   = 900
PWM_MIN   = 80

# Sensor
SENSOR_MIN  = 3.0       # Distancia mínima válida (cm)
SENSOR_MAX  = 40.0      # Distancia máxima válida (cm)
BUF_SIZE    = 7          # Tamaño del búfer de mediana
OUTLIER_THR = 5.0        # Umbral para rechazar outliers (cm)
MAX_LECTURAS_INV = 10    # Lecturas inválidas antes de parar

# Filtros
ALFA_DERIV = 0.35        # EMA de derivada (0 = todo anterior, 1 = todo nuevo)
ALFA_PWM   = 0.70        # EMA de PWM (responsivo pero sin saltos)

# Integral (la pieza clave que faltaba)
KI             = 0.12    # Ganancia integral — pequeña pero persistente
INTEGRAL_MAX   = 40.0    # Límite anti-windup (±)
INTEGRAL_DECAY = 0.998   # Decaimiento leve para evitar acumulación infinita

# Limitador de cambio adaptativo
MAX_CAMBIO_CERCA = 4.0   # PWM/ciclo cuando |error| < UMBRAL_CERCA
MAX_CAMBIO_LEJOS = 12.0  # PWM/ciclo cuando |error| >= UMBRAL_CERCA
UMBRAL_CERCA     = 3.0   # cm

# Logging
MAX_LOGS = 1200          # ~60 s a 20 Hz

# ---- Sensor ultrasónico con mediana + rechazo de outliers ----
buf = []

def medir_cm():
    global buf
    trig.off()
    time.sleep_us(2)
    trig.on()
    time.sleep_us(10)
    trig.off()

    dur = time_pulse_us(echo, 1, 30000)

    # Lectura fallida → devolver mediana del búfer (o -1 si vacío)
    if dur < 0:
        if buf:
            return round(sorted(buf)[len(buf) // 2], 2)
        return -1.0

    d = dur * 0.034 / 2

    if d < SENSOR_MIN or d > SENSOR_MAX:
        if buf:
            return round(sorted(buf)[len(buf) // 2], 2)
        return -1.0

    # Rechazo de outliers: si ya hay suficientes muestras, rechazar
    # lecturas que se alejen demasiado de la mediana actual.
    if len(buf) >= 3:
        mediana = sorted(buf)[len(buf) // 2]
        if abs(d - mediana) > OUTLIER_THR:
            return round(mediana, 2)

    buf.append(d)
    if len(buf) > BUF_SIZE:
        buf.pop(0)
    if len(buf) >= 3:
        return round(sorted(buf)[len(buf) // 2], 2)
    return round(sum(buf) / len(buf), 2)


# ---- Funciones de pertenencia ----
def trapmf(x, a, b, c, d):
    """Trapezoidal: 0 fuera de [a,d], sube en [a,b], 1 en [b,c], baja en [c,d]."""
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
    """Triangular: pico en b."""
    return trapmf(x, a, b, b, c)


# ---- Deltas de PWM (singletons de salida) ----
# Asimétricos: las correcciones hacia arriba son más fuertes porque
# la gravedad ayuda a bajar pero se opone a subir.
DELTA_NV = -10.0    # Frenar fuerte (pelota muy arriba + subiendo)
DELTA_NB =  -5.0    # Bajar rápido
DELTA_NM =  -2.5    # Bajar moderado
DELTA_NS =  -1.0    # Ajuste fino hacia abajo
DELTA_Z  =   0.0    # Mantener
DELTA_PS =   1.5    # Ajuste fino hacia arriba
DELTA_PM =   4.0    # Empujón moderado
DELTA_PB =   8.0    # Subida rápida
DELTA_PV =  20.0    # Rescate fuerte (pelota muy abajo + cayendo)

# ---- Matriz FAM (Fuzzy Associative Memory) ----
# Filas  = nivel de error    (9): NV NB NM NS Z PS PM PB PV
# Columnas = nivel de derivada (7): NV NB NS Z  PS PB PV
#
# Convención de signos:
#   error > 0  →  pelota ABAJO del setpoint  →  necesita MÁS fan
#   error < 0  →  pelota ARRIBA del setpoint →  necesita MENOS fan
#   deriv > 0  →  pelota CAYENDO (error crece) →  necesita MÁS fan
#   deriv < 0  →  pelota SUBIENDO (error decrece) →  necesita MENOS fan
#
# Diseño:
#   • Diagonal: error y derivada del mismo signo → corrección fuerte
#   • Anti-diagonal: se oponen → corrección moderada (ya se auto-corrige)
#   • Centro (Z,Z): DELTA_Z = 0 → la integral se encarga del fino
#   • Filas NS/Z/PS: más entradas Z para amortiguar cerca del setpoint

FAM = [
    # dE:  NVd     NBd     NSd      Zd     PSd     PBd     PVd
    [DELTA_NV, DELTA_NV, DELTA_NB, DELTA_NM, DELTA_NS, DELTA_Z,  DELTA_Z ],  # eNV: muy arriba
    [DELTA_NV, DELTA_NB, DELTA_NM, DELTA_NS, DELTA_Z,  DELTA_Z,  DELTA_PS],  # eNB
    [DELTA_NB, DELTA_NM, DELTA_NS, DELTA_NS, DELTA_Z,  DELTA_PS, DELTA_PM],  # eNM
    [DELTA_NM, DELTA_NS, DELTA_NS, DELTA_Z,  DELTA_Z,  DELTA_PS, DELTA_PM],  # eNS
    [DELTA_NM, DELTA_NS, DELTA_Z,  DELTA_Z,  DELTA_Z,  DELTA_PS, DELTA_PM],  # eZ  — triple Z central
    [DELTA_NM, DELTA_NS, DELTA_Z,  DELTA_Z,  DELTA_PS, DELTA_PS, DELTA_PM],  # ePS
    [DELTA_NM, DELTA_NS, DELTA_Z,  DELTA_PS, DELTA_PS, DELTA_PM, DELTA_PB],  # ePM
    [DELTA_NS, DELTA_Z,  DELTA_Z,  DELTA_PS, DELTA_PM, DELTA_PB, DELTA_PV],  # ePB
    [DELTA_Z,  DELTA_Z,  DELTA_PS, DELTA_PM, DELTA_PB, DELTA_PV, DELTA_PV],  # ePV: muy abajo
]


def fuzzificar_error(e):
    """Convierte error (cm) en 9 grados de pertenencia."""
    return [
        trapmf(e, -50, -50, -12, -6),   # NV: muy arriba
        trimf(e,  -10,  -6, -3),         # NB
        trimf(e,   -5,  -3, -1),         # NM
        trimf(e,   -2,  -1,  0),         # NS
        trimf(e,   -1.5, 0,  1.5),       # Z : en el punto
        trimf(e,    0,   1,  2),         # PS
        trimf(e,    1,   3,  5),         # PM
        trimf(e,    3,   6, 10),         # PB
        trapmf(e,   6,  12, 50, 50),     # PV: muy abajo
    ]

def fuzzificar_derivada(de):
    """Convierte derivada (cm/s) en 7 grados de pertenencia."""
    return [
        trapmf(de, -80, -80, -20,  -8),  # NV: subiendo muy rápido
        trimf(de,  -15,  -8,  -3),        # NB: subiendo rápido
        trimf(de,   -5,  -2,   0),        # NS: subiendo lento
        trimf(de,   -1.5, 0,   1.5),      # Z : quieta
        trimf(de,    0,   2,   5),        # PS: cayendo lento
        trimf(de,    3,   8,  15),        # PB: cayendo rápido
        trapmf(de,   8,  20,  80,  80),   # PV: cayendo en picada
    ]

def inferencia_fuzzy(error, derivada):
    """Evalúa la FAM y retorna delta_pwm por defuzzificación Sugeno."""
    e_niv = fuzzificar_error(error)
    d_niv = fuzzificar_derivada(derivada)

    num = 0.0
    den = 0.0
    for i in range(9):
        ei = e_niv[i]
        if ei <= 0:
            continue
        for j in range(7):
            w = min(ei, d_niv[j])
            if w > 0:
                num += w * FAM[i][j]
                den += w

    return (num / den) if den > 0 else 0.0


# ==================================================================
#  INICIO
# ==================================================================
print("=" * 65)
print("  FUZZY PID — Levitación de pelota (icopor 0.5 g)")
print("  Fuzzy PD + Integral con anti-windup")
print("=" * 65)

try:
    setpoint = float(input("Setpoint (cm, ej 20): ").strip())
except:
    setpoint = 20.0
print("Setpoint: {} cm".format(setpoint))

# Estado del controlador
pwm_actual   = 250.0
error_ant    = 0.0
deriv_f      = 0.0
integral     = 0.0           # ← NUEVO: acumulador integral
lecturas_inv = 0
data_log     = []

# Arranque progresivo
print("Arranque progresivo del ventilador...")
p = 80
while p < int(pwm_actual):
    fan.duty(p)
    time.sleep(0.04)
    p += 15
fan.duty(int(pwm_actual))
time.sleep(1)

t_inicio  = time.ticks_ms()
t_anterior = time.ticks_ms()
ciclos    = 0
gc.collect()

print("{:>7s} | {:>6s} | {:>7s} | {:>7s} | {:>7s} | {:>7s} | {:>7s}".format(
    "t(s)", "Dist", "Error", "dErr", "Integ", "dPWM", "PWM"))
print("-" * 70)

try:
    while True:
        # ---- Medir tiempo real transcurrido ----
        t_ahora  = time.ticks_ms()
        dt_real  = time.ticks_diff(t_ahora, t_anterior) / 1000.0
        t_anterior = t_ahora

        # Protección: evitar dt absurdos
        if dt_real < 0.01:
            dt_real = 0.01
        elif dt_real > 0.5:
            dt_real = 0.5

        # ---- Lectura del sensor ----
        dist = medir_cm()
        if dist < 0:
            lecturas_inv += 1
            if lecturas_inv > MAX_LECTURAS_INV:
                print("ERROR: Demasiadas lecturas inválidas. Deteniendo.")
                break
            time.sleep(DT_TARGET)
            continue

        lecturas_inv = 0
        tiempo_seg = time.ticks_diff(t_ahora, t_inicio) / 1000.0

        # ---- 1. ERROR ----
        error = dist - setpoint

        # ---- 2. DERIVADA (filtrada con EMA) ----
        deriv_raw = (error - error_ant) / dt_real
        deriv_f   = ALFA_DERIV * deriv_raw + (1.0 - ALFA_DERIV) * deriv_f
        error_ant = error

        # ---- 3. INTEGRAL (con anti-windup y decay) ----
        # Solo acumular cuando el error no es enorme (evita windup
        # durante transitorios o cuando la pelota está fuera de rango).
        if abs(error) < 10.0:
            integral += error * dt_real
            integral *= INTEGRAL_DECAY
            # Anti-windup: limitar acumulación
            if integral > INTEGRAL_MAX:
                integral = INTEGRAL_MAX
            elif integral < -INTEGRAL_MAX:
                integral = -INTEGRAL_MAX

        # ---- 4. INFERENCIA DIFUSA (PD) ----
        delta_fuzzy = inferencia_fuzzy(error, deriv_f)

        # ---- 5. CONTRIBUCIÓN INTEGRAL ----
        delta_integral = KI * integral

        # ---- 6. DELTA TOTAL ----
        delta_pwm = delta_fuzzy + delta_integral

        # ---- 7. LIMITADOR DE CAMBIO ADAPTATIVO ----
        if abs(error) > UMBRAL_CERCA:
            max_cambio = MAX_CAMBIO_LEJOS
        else:
            max_cambio = MAX_CAMBIO_CERCA
        delta_pwm = max(-max_cambio, min(max_cambio, delta_pwm))

        # ---- 8. APLICAR CON SUAVIZADO EMA ----
        pwm_objetivo = pwm_actual + delta_pwm
        pwm_objetivo = max(float(PWM_MIN), min(float(PWM_MAX), pwm_objetivo))
        pwm_actual   = ALFA_PWM * pwm_objetivo + (1.0 - ALFA_PWM) * pwm_actual
        pwm_actual   = max(float(PWM_MIN), min(float(PWM_MAX), pwm_actual))

        fan.duty(int(pwm_actual))

        # ---- Logging protegido contra MemoryError ----
        data_log.append((
            tiempo_seg, dist, setpoint, error,
            deriv_f, integral, delta_pwm, pwm_actual
        ))
        if len(data_log) > MAX_LOGS:
            data_log.pop(0)

        # Limpieza de RAM cada 100 ciclos (~5 s)
        ciclos += 1
        if ciclos % 100 == 0:
            gc.collect()

        print("{:7.2f} | {:6.2f} | {:+7.2f} | {:+7.2f} | {:+7.2f} | {:+7.2f} | {:7.2f}".format(
            tiempo_seg, dist, error, deriv_f, integral, delta_pwm, pwm_actual))

        # ---- Compensar tiempo de loop ----
        transcurrido = time.ticks_diff(time.ticks_ms(), t_ahora) / 1000.0
        espera = DT_TARGET - transcurrido
        if espera > 0:
            time.sleep(espera)

except KeyboardInterrupt:
    # Aterrizaje suave: bajar PWM gradualmente
    print("\nAterrizaje suave...")
    p = int(pwm_actual)
    while p > 150:
        fan.duty(p)
        time.sleep(0.04)
        p -= 20
    fan.duty(0)
    fan.deinit()
    print("Motor detenido. Aterrizaje seguro.")

    # ---- Estadísticas ----
    if data_log:
        errores = [abs(d[3]) for d in data_log]
        n = len(errores)
        err_prom = sum(errores) / n
        err_max  = max(errores)
        en_1cm   = sum(1 for e in errores if e <= 1.0) * 100 // n
        en_2cm   = sum(1 for e in errores if e <= 2.0) * 100 // n
        en_3cm   = sum(1 for e in errores if e <= 3.0) * 100 // n

        print("\n--- ESTADISTICAS ({} muestras) ---".format(n))
        print("Error promedio: {:.2f} cm".format(err_prom))
        print("Error maximo:   {:.2f} cm".format(err_max))
        print("Dentro de +-1 cm: {}%".format(en_1cm))
        print("Dentro de +-2 cm: {}%".format(en_2cm))
        print("Dentro de +-3 cm: {}%".format(en_3cm))

    # ---- Exportar CSV ----
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
