from machine import Pin, PWM, time_pulse_us
import time

# Pines
TRIG_PIN = 27
ECHO_PIN = 26
FAN_PIN  = 14
VOLTAJE_FUENTE = 12.0

trig = Pin(TRIG_PIN, Pin.OUT)
echo = Pin(ECHO_PIN, Pin.IN)
fan = PWM(Pin(FAN_PIN), freq=25000, duty=0)

# Variables de control
DT = 0.05
PWM_MAX = 900
PWM_MIN = 170
buf = []
_rechazos = 0           # Contador de lecturas rechazadas consecutivas
MAX_RECHAZOS = 5        # Tras este número, limpiar búfer para re-adaptar

# --- FUNCIONES DE PERTENENCIA DIFUSA (Fuzzification) ---
# Trapecio: x, a (inicio), b (subida), c (bajada), d (fin)
def trapmf(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    if a < x <= b: return (x - a) / (b - a) if b != a else 1.0
    if b < x <= c: return 1.0
    if c < x < d: return (d - x) / (d - c) if d != c else 1.0
    return 0.0

# Triángulo: x, a (inicio), b (pico), c (fin)
def trimf(x, a, b, c):
    return trapmf(x, a, b, b, c)

def medir_cm():
    global buf, _rechazos
    trig.off()
    time.sleep_us(2)
    trig.on()
    time.sleep_us(10)
    trig.off()
    dur = time_pulse_us(echo, 1, 30000)
    if dur < 0:
        _rechazos += 1
        if _rechazos >= MAX_RECHAZOS:
            buf = []
            _rechazos = 0
        return -1.0 if not buf else round(sorted(buf)[len(buf)//2], 1)
    d = dur * 0.034 / 2
    if d < 3 or d > 40:
        _rechazos += 1
        if _rechazos >= MAX_RECHAZOS:
            buf = []
            _rechazos = 0
        return -1.0 if not buf else round(sorted(buf)[len(buf)//2], 1)
    _rechazos = 0
    buf.append(d)
    if len(buf) > 5: buf.pop(0)
    return round(sorted(buf)[len(buf)//2], 1)

print("="*50)
print("CONTROL DIFUSO PD (Fuzzy Logic) - Levitación")
print("="*50)

try:
    setpoint = float(input("Setpoint (cm, ej 20): ").strip())
except:
    setpoint = 20.0

pwm_actual = 450 # PWM de arranque
error_ant = 0.0
deriv_f = 0.0
fan.duty(pwm_actual)
time.sleep(1)

while True:
    try:
        dist = medir_cm()
        if dist < 0:
            time.sleep(DT)
            continue

        # 1. Calcular Entradas (Crisp Inputs)
        error = dist - setpoint
        deriv = (error - error_ant) / DT
        deriv_f = 0.4 * deriv + 0.6 * deriv_f # Filtro suave para la derivada
        error_ant = error

        # 2. Fuzzificación del Error
        # e_N: Error Negativo (Pelota arriba)
        # e_Z: Error Cero (Pelota en objetivo)
        # e_P: Error Positivo (Pelota abajo)
        e_N = trapmf(error, -40, -40, -5, -1)
        e_Z = trapmf(error, -3, -1, 1, 3)
        e_P = trapmf(error, 1, 5, 40, 40)

        # 3. Fuzzificación de la Derivada
        # de_N: Subiendo rápido
        # de_Z: Estable
        # de_P: Cayendo rápido
        de_N = trapmf(deriv_f, -50, -50, -10, -2)
        de_Z = trapmf(deriv_f, -5, -1, 1, 5)
        de_P = trapmf(deriv_f, 2, 10, 50, 50)

        # 4. Evaluacion de Reglas Difusas (AND = min) y Salidas (Singletons)
        NB = -35 # Delta Muy Negativo (Frenar mucho)
        NS = -10 # Delta Poco Negativo
        Z  = 0   # Mantener
        PS = 10  # Delta Poco Positivo
        PB = 35  # Delta Muy Positivo (Acelerar mucho)

        reglas = [
            # Si pelota está ARRIBA...
            (min(e_N, de_N), NB), # y subiendo -> Frena MUCHO
            (min(e_N, de_Z), NS), # y quieta -> Frena poco
            (min(e_N, de_P), Z),  # pero cayendo -> Deja que caiga (Z)
            
            # Si pelota está en CENTRO...
            (min(e_Z, de_N), NS), # pero subiendo -> Frena poco para no pasarse
            (min(e_Z, de_Z), Z),  # y quieta -> Perfecto, mantener (Z)
            (min(e_Z, de_P), PS), # pero cayendo -> Acelera poco para sostener
            
            # Si pelota está ABAJO...
            (min(e_P, de_N), Z),  # pero subiendo -> Deja que suba (Z)
            (min(e_P, de_Z), PS), # y quieta -> Acelera poco
            (min(e_P, de_P), PB)  # y cayendo -> Acelera MUCHO (Rescate)
        ]

        # 5. Defuzzificación (Media Ponderada - Sugeno)
        numerador = sum(peso * salida for peso, salida in reglas)
        denominador = sum(peso for peso, salida in reglas)
        
        if denominador > 0:
            delta_pwm = numerador / denominador
        else:
            delta_pwm = 0

        # Aplicar el cambio al PWM
        pwm_actual += delta_pwm
        pwm_actual = int(max(PWM_MIN, min(PWM_MAX, pwm_actual)))
        
        fan.duty(pwm_actual)

        print(f"Dist: {dist:04.1f} | Err: {error:+05.1f} | Deriv: {deriv_f:+05.1f} | dPWM: {delta_pwm:+05.1f} | PWM: {pwm_actual}")
        time.sleep(DT)

    except KeyboardInterrupt:
        fan.duty(0)
        fan.deinit()
        print("\nMotor difuso detenido.")
        break