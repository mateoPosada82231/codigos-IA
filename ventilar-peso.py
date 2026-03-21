from machine import Pin, PWM, time_pulse_us
import time
import gc  # Recolector de basura para liberar memoria

# Pines
TRIG_PIN = 27
ECHO_PIN = 26
FAN_PIN  = 14

trig = Pin(TRIG_PIN, Pin.OUT)
echo = Pin(ECHO_PIN, Pin.IN)
fan = PWM(Pin(FAN_PIN), freq=25000, duty=0)

# Variables de control
DT = 0.05  # 20 Hz (Súper rápido para atrapar caídas)
PWM_MAX = 900
PWM_MIN = 170  # Límite seguro calibrado
ALFA_PWM = 0.4      # Suavizado exponencial de PWM (40% nuevo, 60% anterior)
ALFA_DERIV = 0.35   # Filtro de derivada más responsivo
ZONA_MUERTA = 1.0   # Zona de tolerancia (cm) alrededor del setpoint
UMBRAL_DERIV = 3.0   # Umbral de derivada para zona muerta (cm/s)
MAX_LECTURAS_INV = 10  # Máximo de lecturas inválidas consecutivas antes de parar
buf = []

# --- GESTIÓN DE MEMORIA PARA EL CSV ---
data_log = []
MAX_LOGS = 1200  # Aprox 60 segundos de grabación. Evita el MemoryError.

# --- FUNCIONES DE PERTENENCIA ---
def trapmf(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    if a < x <= b: return (x - a) / (b - a) if b != a else 1.0
    if b < x <= c: return 1.0
    if c < x < d: return (d - x) / (d - c) if d != c else 1.0
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
    if dur < 0: return -1.0 if not buf else round(sorted(buf)[len(buf)//2], 2)
    d = dur * 0.034 / 2
    if d < 3 or d > 40: return -1.0 if not buf else round(sorted(buf)[len(buf)//2], 2)
    
    # Búfer de 5 lecturas para mejor filtrado por mediana
    buf.append(d)
    if len(buf) > 5: buf.pop(0)
    return round(sorted(buf)[len(buf)//2], 2)

print("="*65)
print("CONTROL DIFUSO - AJUSTE 'PESO PLUMA' (ICOPOR 0.5g)")
print("="*65)

try:
    setpoint = float(input("Setpoint (cm, ej 20): ").strip())
except:
    setpoint = 20.0

pwm_actual = 350.0  # Iniciamos un poco más bajo por el poco peso
error_ant = 0.0
deriv_f = 0.0
lecturas_invalidas = 0

# Arranque progresivo del ventilador
print("Arranque progresivo...")
p = 150
while p < int(pwm_actual):
    fan.duty(p)
    time.sleep(0.05)
    p += 20
fan.duty(int(pwm_actual))
time.sleep(1)

# =================================================================
# DELTAS DE PWM: AJUSTADOS PARA CONTROL EFECTIVO DE 0.5 GRAMOS
# Los deltas anteriores eran demasiado pequeños (-4 a +8), haciendo
# que el sistema no pudiera corregir la posición a tiempo.
# =================================================================
NV_out = -15.0  # Freno de emergencia fuerte
NB_out =  -8.0  # Bajar rápido
NM_out =  -3.0  # Bajar moderado
NS_out =  -1.0  # Ajuste fino hacia abajo
Z_out  =   0.0  # MANTENER POTENCIA EXACTA
PS_out =   1.5  # Ajuste fino hacia arriba
PM_out =   4.0  # Empujón moderado
PB_out =  10.0  # Subida rápida
PV_out =  25.0  # Rescate fuerte desde el fondo

# Matriz FAM Asimétrica (Anti Yo-Yo) con amortiguación fuerte en zona cero
FAM = [
    [NV_out, NV_out, NB_out, NM_out, NS_out, Z_out,  Z_out ],  # NV: Muy arriba
    [NV_out, NB_out, NM_out, NS_out, Z_out,  PS_out, PS_out],  # NB
    [NB_out, NM_out, NS_out, NS_out, Z_out,  PS_out, PM_out],  # NM
    [NM_out, NS_out, NS_out, NS_out, PS_out, PM_out, PM_out],  # NS
    [NB_out, NM_out, NS_out, Z_out,  PS_out, PM_out, PB_out],  # Z: Amortiguar movimiento activamente
    [NM_out, NS_out, Z_out,  PS_out, PS_out, PM_out, PB_out],  # PS
    [NS_out, Z_out,  Z_out,  PS_out, PM_out, PB_out, PB_out],  # PM
    [Z_out,  Z_out,  PS_out, PM_out, PB_out, PB_out, PV_out],  # PB
    [Z_out,  PS_out, PM_out, PB_out, PB_out, PV_out, PV_out]   # PV: Muy abajo
]

t_inicio = time.ticks_ms()
ciclos = 0 # Contador para limpieza de memoria

gc.collect() # Limpieza inicial de RAM

try:
    while True:
        dist = medir_cm()
        if dist < 0:
            lecturas_invalidas += 1
            if lecturas_invalidas > MAX_LECTURAS_INV:
                print("Demasiadas lecturas invalidas. Deteniendo por seguridad.")
                break
            time.sleep(DT)
            continue

        lecturas_invalidas = 0
        tiempo_actual = time.ticks_diff(time.ticks_ms(), t_inicio) / 1000.0

        # 1. Error y Derivada con filtro mejorado
        error = dist - setpoint
        deriv = (error - error_ant) / DT
        deriv_f = ALFA_DERIV * deriv + (1 - ALFA_DERIV) * deriv_f
        error_ant = error

        # 2. Zona muerta: si error pequeño Y pelota casi quieta, no corregir
        if abs(error) < ZONA_MUERTA and abs(deriv_f) < UMBRAL_DERIV:
            delta_pwm = 0.0
        else:
            # 3. Funciones de Error (Zona "Cero" ampliada a ±2 cm)
            e_niveles = [
                trapmf(error, -50, -50, -15, -8),  # NV: Muy arriba
                trimf(error, -12, -8, -4),         # NB
                trimf(error, -6, -4, -1.5),        # NM
                trimf(error, -3.0, -1.0, 0),       # NS
                trimf(error, -2.0, 0.0, 2.0),      # Z: Zona de confort (±2 cm)
                trimf(error, 0, 1.0, 3.0),         # PS
                trimf(error, 1.5, 4, 6),           # PM
                trimf(error, 4, 8, 12),            # PB
                trapmf(error, 8, 15, 50, 50)       # PV: Muy abajo
            ]

            # 4. Funciones de Derivada (Velocidad de la pelota, banda quieta más ancha)
            de_niveles = [
                trapmf(deriv_f, -80, -80, -25, -10), # Subiendo rapidísimo
                trimf(deriv_f, -20, -10, -3),        # Subiendo rápido
                trimf(deriv_f, -6, -3, 0),           # Subiendo lento
                trimf(deriv_f, -2.0, 0, 2.0),        # Quieta (±2 cm/s)
                trimf(deriv_f, 0, 3, 6),             # Cayendo lento
                trimf(deriv_f, 3, 10, 20),           # Cayendo rápido
                trapmf(deriv_f, 10, 25, 80, 80)      # Cayendo en picada
            ]

            # 5. Evaluación de Inferencia
            numerador = 0.0
            denominador = 0.0
            for i in range(9):
                for j in range(7):
                    peso = min(e_niveles[i], de_niveles[j])
                    if peso > 0:
                        numerador += peso * FAM[i][j]
                        denominador += peso

            delta_pwm = (numerador / denominador) if denominador > 0 else 0.0

        # 6. Aplicar con suavizado exponencial (evita saltos bruscos)
        pwm_objetivo = pwm_actual + delta_pwm
        pwm_objetivo = max(float(PWM_MIN), min(float(PWM_MAX), pwm_objetivo))
        pwm_actual = ALFA_PWM * pwm_objetivo + (1 - ALFA_PWM) * pwm_actual
        pwm_actual = max(float(PWM_MIN), min(float(PWM_MAX), pwm_actual))

        fan.duty(int(pwm_actual))

        # --- GUARDADO PROTEGIDO CONTRA MEMORY ERROR ---
        data_log.append((tiempo_actual, dist, setpoint, error, deriv_f, delta_pwm, pwm_actual))
        
        if len(data_log) > MAX_LOGS:
            data_log.pop(0)

        # Limpiar la memoria RAM cada 100 ciclos (5 segundos)
        ciclos += 1
        if ciclos % 100 == 0:
            gc.collect()

        print(f"Dist: {dist:05.2f} | Err: {error:+06.2f} | dE: {deriv_f:+06.2f} | dPWM: {delta_pwm:+05.2f} | PWM: {pwm_actual:06.2f}")
        time.sleep(DT)

except KeyboardInterrupt:
    # Aterrizaje suave: reducir PWM gradualmente
    print("\nAterrizaje suave...")
    p = int(pwm_actual)
    while p > 150:
        fan.duty(p)
        time.sleep(0.04)
        p -= 20
    fan.duty(0)
    fan.deinit()
    print("Motor difuso detenido. Aterrizaje seguro.")
    
    resp = input(f"¿Guardar los últimos {len(data_log)} datos en CSV? (s/n): ").strip().lower()
    if resp == 's':
        try:
            with open("datos_levitacion.csv", "w") as f:
                f.write("tiempo,distancia,setpoint,error,derivada,delta_pwm,pwm\n")
                for d in data_log:
                    f.write(f"{d[0]:.3f},{d[1]:.2f},{d[2]:.2f},{d[3]:.2f},{d[4]:.2f},{d[5]:.2f},{d[6]:.2f}\n")
            print("Guardado con éxito en el ESP32.")
        except Exception as e:
            print("Error al escribir el archivo:", e)