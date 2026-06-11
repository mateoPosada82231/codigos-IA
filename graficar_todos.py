import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Crear directorio para guardar gráficas
output_dir = r"c:\codigos-IA\graficas_compiladas"
os.makedirs(output_dir, exist_ok=True)

# Lista de archivos CSV con sus rutas
csv_files = {
    'Fuzzy Bisector': r'c:\codigos-IA\logica_difusa\datos_esp32_bisector.csv',
    'Fuzzy Centroide': r'c:\codigos-IA\logica_difusa\datos_esp32_centroide.csv',
    'Fuzzy MOM': r'c:\codigos-IA\logica_difusa\datos_esp32_mom.csv',
    'Red Neuronal ReLU': r'c:\codigos-IA\redes_neuronales\datos_esp32_relu.csv',
    'Red Neuronal Sigmoid': r'c:\codigos-IA\redes_neuronales\datos_esp32_sigmoid.csv',
    'Red Neuronal Tanh': r'c:\codigos-IA\redes_neuronales\datos_esp32_tanh.csv',
    'RL DQN': r'c:\codigos-IA\aprendizaje_refuerzo\datos_esp32_dqn.csv',
    'RL Q-Learning': r'c:\codigos-IA\aprendizaje_refuerzo\datos_esp32_qlearning.csv',
}

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')

for nombre, ruta in csv_files.items():
    try:
        print(f"Procesando: {nombre}")
        df = pd.read_csv(ruta)
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Análisis de Datos - {nombre}', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Tiempo vs Distancia
        axes[0, 0].plot(df['tiempo'], df['distancia'], 'b-', linewidth=2, label='Distancia')
        axes[0, 0].axhline(y=df['setpoint'].iloc[0], color='r', linestyle='--', linewidth=2, label='Setpoint')
        axes[0, 0].set_xlabel('Tiempo (ms)', fontsize=10)
        axes[0, 0].set_ylabel('Distancia (cm)', fontsize=10)
        axes[0, 0].set_title('Distancia vs Tiempo', fontsize=11)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gráfico 2: Tiempo vs Error
        axes[0, 1].plot(df['tiempo'], df['error'], 'g-', linewidth=2)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axes[0, 1].set_xlabel('Tiempo (ms)', fontsize=10)
        axes[0, 1].set_ylabel('Error (cm)', fontsize=10)
        axes[0, 1].set_title('Error vs Tiempo', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Tiempo vs PWM
        axes[1, 0].plot(df['tiempo'], df['pwm'], 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Tiempo (ms)', fontsize=10)
        axes[1, 0].set_ylabel('PWM', fontsize=10)
        axes[1, 0].set_title('PWM vs Tiempo', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gráfico 4: Tiempo vs Derivada (o velocidad si existe)
        if 'derivada' in df.columns:
            axes[1, 1].plot(df['tiempo'], df['derivada'], 'orange', linewidth=2, label='Derivada')
        if 'velocidad' in df.columns:
            axes[1, 1].plot(df['tiempo'], df['velocidad'], 'cyan', linewidth=2, label='Velocidad')
        axes[1, 1].set_xlabel('Tiempo (ms)', fontsize=10)
        axes[1, 1].set_ylabel('Valor', fontsize=10)
        axes[1, 1].set_title('Derivada / Velocidad vs Tiempo', fontsize=11)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar figura
        filename = f"{nombre.replace(' ', '_').replace('/', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Guardado: {filename}")
        plt.close()
        
    except Exception as e:
        print(f"✗ Error procesando {nombre}: {e}")

# Crear una figura comparativa de distancias
print("\nCreando gráfico comparativo de distancias...")
fig, ax = plt.subplots(figsize=(14, 8))

for nombre, ruta in csv_files.items():
    try:
        df = pd.read_csv(ruta)
        ax.plot(df['tiempo'], df['distancia'], linewidth=2, label=nombre, alpha=0.8)
    except:
        pass

ax.set_xlabel('Tiempo (ms)', fontsize=12)
ax.set_ylabel('Distancia (cm)', fontsize=12)
ax.set_title('Comparativa de Distancia - Todos los Métodos', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Comparativa_Distancias.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: Comparativa_Distancias.png")

# Crear una figura comparativa de errores
print("\nCreando gráfico comparativo de errores...")
fig, ax = plt.subplots(figsize=(14, 8))

for nombre, ruta in csv_files.items():
    try:
        df = pd.read_csv(ruta)
        ax.plot(df['tiempo'], df['error'], linewidth=2, label=nombre, alpha=0.8)
    except:
        pass

ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Tiempo (ms)', fontsize=12)
ax.set_ylabel('Error (cm)', fontsize=12)
ax.set_title('Comparativa de Error - Todos los Métodos', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Comparativa_Errores.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: Comparativa_Errores.png")

print(f"\n✓ ¡Todas las gráficas han sido guardadas en: {output_dir}\n")
