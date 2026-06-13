"""
maestro.py
========================
Orquestador interactivo para los controladores de Lógica Difusa:

  1. Menú para elegir qué controlador ejecutar en el ESP32:
       1. Fuzzy Centroide  (controller_centroid.py)
       2. Fuzzy Bisector   (controller_bisector.py)
       3. Fuzzy MOM        (controller_mom.py)

  2. Pide setpoint en el PC, sube el archivo sin input() al ESP32.
  3. Abre mini-terminal serie (pyserial): muestra output y envía Ctrl+C.
  4. Al terminar pregunta si descargar el CSV generado.
  5. Pregunta si ejecutar de nuevo (mismo método o volver al menú).

Uso:
    python maestro.py

Requisitos:
    pip install adafruit-ampy pyserial
"""

import os
import re
import subprocess
import time

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONTROLADORES = {
    "centroide": os.path.join(SCRIPT_DIR, "controller_centroid.py"),
    "bisector":  os.path.join(SCRIPT_DIR, "controller_bisector.py"),
    "mom":       os.path.join(SCRIPT_DIR, "controller_mom.py"),
}

CSV_NOMBRES = {
    "centroide": "datos_fuzzy_centroide.csv",
    "bisector":  "datos_fuzzy_bisector.csv",
    "mom":       "datos_fuzzy_mom.csv",
}

# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN DE PUERTO COM
# ─────────────────────────────────────────────────────────────────────────────

_PUERTO_CACHE = [None]


def _detectar_puerto():
    try:
        from serial.tools import list_ports
        candidatos = []
        for p in list_ports.comports():
            desc = (p.description or "").lower()
            if any(k in desc for k in ("cp210", "ch340", "ch341", "ftdi", "uart",
                                        "silabs", "usb serial", "esp")):
                candidatos.append(p.device)
        if len(candidatos) == 1:
            print(f"  Puerto detectado automáticamente: {candidatos[0]}")
            return candidatos[0]
        if len(candidatos) > 1:
            print("  Se encontraron varios puertos compatibles:")
            for i, p in enumerate(candidatos, 1):
                print(f"    {i}. {p}")
            while True:
                sel = input("  Elige el número del puerto: ").strip()
                if sel.isdigit() and 1 <= int(sel) <= len(candidatos):
                    return candidatos[int(sel) - 1]
                print("  Número inválido.")
        return None
    except ImportError:
        return None


def _pedir_puerto():
    if _PUERTO_CACHE[0]:
        return _PUERTO_CACHE[0]
    puerto = _detectar_puerto()
    if not puerto:
        print("  No se detectó el puerto automáticamente.")
        puerto = input("  Introduce el puerto COM manualmente (ej. COM3): ").strip()
    if puerto:
        _PUERTO_CACHE[0] = puerto
    return puerto or None


# ─────────────────────────────────────────────────────────────────────────────
# PREPARAR ARCHIVO SIN input() PARA EL ESP32
# ─────────────────────────────────────────────────────────────────────────────

def _preparar_archivo(metodo: str, setpoint: float) -> str:
    """
    Crea copia temporal del controlador con:
    - setpoint hardcodeado (elimina input() interactivo)
    - guardado CSV automático (resp = 's')
    """
    archivo_orig = CONTROLADORES[metodo]
    archivo_tmp  = os.path.join(SCRIPT_DIR, f"_tmp_fuzzy_{metodo}.py")

    with open(archivo_orig, 'r') as f:
        lineas = f.readlines()

    salida = []
    i = 0
    while i < len(lineas):
        linea = lineas[i]

        # Reemplazar bloque try/except del setpoint
        if linea.strip() == "try:" and i + 1 < len(lineas) and \
                "setpoint" in lineas[i+1].lower() and "input(" in lineas[i+1].lower():
            fin = i + 1
            while fin < len(lineas) and (
                "setpoint" in lineas[fin].lower() or
                lineas[fin].strip().startswith("except") or
                lineas[fin].strip() == ""
            ):
                fin += 1
            salida.append(f"setpoint = {setpoint:.2f}  # fijado desde maestro.py\n")
            i = fin
            continue

        # Reemplazar asignación directa de setpoint si no viene de input
        if re.match(r'\s*setpoint\s*=\s*[\d.]+', linea) and "input" not in linea and \
                "control_maestro" not in linea:
            salida.append(f"setpoint = {setpoint:.2f}  # fijado desde maestro.py\n")
            i += 1
            continue

        # Guardado automático del CSV
        if "resp" in linea and "input(" in linea and "strip().lower()" in linea:
            salida.append("    resp = 's'  # guardado automático desde maestro.py\n")
            i += 1
            continue

        salida.append(linea)
        i += 1

    with open(archivo_tmp, 'w') as f:
        f.writelines(salida)

    return archivo_tmp


# ─────────────────────────────────────────────────────────────────────────────
# MINI-TERMINAL SERIE
# ─────────────────────────────────────────────────────────────────────────────

def _terminal_serie(puerto: str, baud: int = 115200):
    import serial as _serial
    print("  Conectado al ESP32. Presiona Ctrl+C para detener el controlador.\n")
    print('─' * 60)
    with _serial.Serial(puerto, baud, timeout=0.05) as ser:
        try:
            while True:
                data = ser.read(256)
                if data:
                    print(data.decode('utf-8', errors='replace'), end='', flush=True)
        except KeyboardInterrupt:
            ser.write(b'\x03')
            time.sleep(1.5)
            ser.timeout = 0.2
            while True:
                data = ser.read(256)
                if not data:
                    break
                print(data.decode('utf-8', errors='replace'), end='', flush=True)
    print('\n' + '─' * 60)
    print("  Controlador detenido.")


# ─────────────────────────────────────────────────────────────────────────────
# EJECUTAR EN ESP32
# ─────────────────────────────────────────────────────────────────────────────

def ejecutar_en_esp32(metodo: str, setpoint: float):
    print(f"\n{'='*60}")
    print(f"  Preparando: {os.path.basename(CONTROLADORES[metodo])}")
    print('=' * 60)
    print("\n  IMPORTANTE: desconecta el MicroPico vREPL de VS Code antes de continuar.")
    input("  Presiona Enter cuando el puerto esté libre...")

    puerto = _pedir_puerto()
    if not puerto:
        print("[ERROR] No se especificó un puerto COM.")
        return -1

    archivo_tmp = _preparar_archivo(metodo, setpoint)

    print(f"\n  Subiendo controlador (setpoint={setpoint:.1f} cm) → main.py en {puerto}...")
    cmd_put = ["ampy", "--port", puerto, "put", archivo_tmp, "main.py"]
    try:
        res = subprocess.run(cmd_put, check=False, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[ERROR] ampy put falló:\n{res.stderr.strip()}")
            _PUERTO_CACHE[0] = None
            os.remove(archivo_tmp)
            return res.returncode
        print("  Archivo subido correctamente.")
    except FileNotFoundError:
        print("\n[ERROR] 'ampy' no está instalado. Instálalo con:  pip install adafruit-ampy")
        os.remove(archivo_tmp)
        return -1
    finally:
        try:
            os.remove(archivo_tmp)
        except OSError:
            pass

    # Soft-reset del ESP32
    try:
        import serial as _serial
        print(f"\n  Reiniciando ESP32 en {puerto}...")
        with _serial.Serial(puerto, 115200, timeout=1) as ser:
            ser.write(b'\x03\x03')
            time.sleep(0.3)
            ser.read_all()
            ser.write(b'\x04')
            time.sleep(2.0)
        print("  ESP32 reiniciado. Iniciando controlador...")
    except ImportError:
        print("\n[ERROR] pyserial no instalado. Instálalo con:  pip install pyserial")
        return -1
    except Exception as e:
        print(f"\n[ERROR] No se pudo reiniciar el ESP32: {e}")
        return -1

    _terminal_serie(puerto)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# DESCARGAR CSV DEL ESP32
# ─────────────────────────────────────────────────────────────────────────────

def descargar_csv(metodo: str):
    csv_remoto = CSV_NOMBRES[metodo]
    csv_local  = os.path.join(SCRIPT_DIR, f"datos_esp32_{metodo}.csv")
    puerto = _pedir_puerto()
    if not puerto:
        print("[AVISO] No hay puerto COM configurado.")
        return None
    print(f"\n  Descargando /{csv_remoto} desde el ESP32...")
    cmd = ["ampy", "--port", puerto, "get", csv_remoto, csv_local]
    try:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if res.returncode == 0 and os.path.isfile(csv_local):
            print(f"  CSV descargado en: {csv_local}")
            return csv_local
        else:
            print("  [AVISO] No se pudo descargar el CSV.")
            if res.stderr.strip():
                print(f"  Detalle: {res.stderr.strip()}")
            return None
    except FileNotFoundError:
        print("[AVISO] 'ampy' no instalado.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def _preguntar_si_no(pregunta: str) -> bool:
    while True:
        resp = input(f"{pregunta} [s/n]: ").strip().lower()
        if resp in ('s', 'si', 'sí', 'yes', 'y'):
            return True
        if resp in ('n', 'no'):
            return False
        print("  Responde 's' o 'n'.")


def _pedir_setpoint(default: float = 15.0) -> float:
    while True:
        sp = input(f"  Setpoint (cm, ej. {default}): ").strip()
        try:
            return float(sp) if sp else default
        except ValueError:
            print("  Valor inválido.")


def _mostrar_menu():
    print("\n" + "═" * 60)
    print("  CONTROL MAESTRO — Lógica Difusa  (Levitador de Pelota)")
    print("═" * 60)
    print("  Selecciona el método de desfusificación:\n")
    print("    1. Centroide   (controller_centroid.py)")
    print("    2. Bisector    (controller_bisector.py)")
    print("    3. MOM         (controller_mom.py)")
    print("    0. Salir")
    print("─" * 60)


OPCION_A_METODO = {
    "1": "centroide",
    "2": "bisector",
    "3": "mom",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n  Bienvenido al Control Maestro — Lógica Difusa")
    print("  Requisitos: pip install adafruit-ampy pyserial\n")

    while True:
        _mostrar_menu()
        opcion = input("  Opción: ").strip()

        if opcion == "0":
            print("\nSaliendo. ¡Hasta luego!\n")
            break

        if opcion not in OPCION_A_METODO:
            print("  Opción inválida.")
            continue

        metodo = OPCION_A_METODO[opcion]
        setpoint = _pedir_setpoint()

        # Ciclo ejecutar → descarga CSV → ejecutar de nuevo
        while True:
            ejecutar_en_esp32(metodo, setpoint)

            if _preguntar_si_no("\n¿Deseas descargar el CSV de datos?"):
                descargar_csv(metodo)

            if not _preguntar_si_no("\n¿Deseas ejecutar de nuevo?"):
                if _preguntar_si_no("¿Volver al menú principal?"):
                    break
                else:
                    print("\nSaliendo. ¡Hasta luego!\n")
                    return

            # Preguntar si cambia el setpoint antes de repetir
            if _preguntar_si_no("¿Cambiar el setpoint?"):
                setpoint = _pedir_setpoint(setpoint)


if __name__ == "__main__":
    main()
