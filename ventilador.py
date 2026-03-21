from machine import Pin, PWM
import sys
import select

pwm = PWM(Pin(14), freq=25000)

VOLTAJE_FUENTE = 12.0

def set_speed(percent):
    if percent < 0: percent = 0
    if percent > 100: percent = 100
    
    duty = int((percent / 100) * 1023)
    pwm.duty(duty)
    
    voltios = (percent / 100) * VOLTAJE_FUENTE
    print(f"✓ Velocidad: {percent}% | Voltaje: {voltios:.2f}V / {VOLTAJE_FUENTE}V | duty: {duty}/1023")

print("=" * 50)
print("  Control de ventilador 12V")
print("  Escribe la velocidad en % (0-100)")
print("=" * 50)

# Apagar ventilador al inicio
set_speed(0)

while True:
    try:
        print("Velocidad %: ", end="")
        entrada = sys.stdin.readline().strip()  # ← reemplaza input()
        
        if entrada == "":
            continue
            
        speed = int(entrada)
        set_speed(speed)
        
    except KeyboardInterrupt:
        print("\n🛑 Programa detenido")
        set_speed(0)   # Apaga el ventilador al salir
        pwm.deinit()   # Libera el pin PWM
        break
    except:
        print("⚠️ Escribe un número entre 0 y 100")