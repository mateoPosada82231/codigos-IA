import pandas as pd
import matplotlib.pyplot as plt
import os

# Crear directorio para guardar gráficas
output_dir = r"figures"
os.makedirs(output_dir, exist_ok=True)

# Organizar métodos por categoría
fuzzy_methods = {
    'Bisector': r'fuzzy\data_bisector.csv',
    'Centroide': r'fuzzy\data_centroid.csv',
    'MOM': r'fuzzy\data_mom.csv',
}

neural_methods = {
    'ReLU': r'neural_networks\data_relu.csv',
    'Sigmoid': r'neural_networks\data_sigmoid.csv',
    'Tanh': r'neural_networks\data_tanh.csv',
}

rl_methods = {
    'DQN': r'reinforcement_learning\data_dqn.csv',
    'Q-Learning': r'reinforcement_learning\data_qlearning.csv',
}

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')

# ===== CREAR GRÁFICAS POR CATEGORÍA - TODOS SUPERPUESTOS EN 2 PLOTS =====
print("Creating category comparison graphs (all methods overlaid)...")

# 1. FUZZY LOGIC
print("\n1. Creating Fuzzy Logic comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Fuzzy Logic Methods - Comparison', fontsize=16, fontweight='bold')

for metodo_nombre, ruta in fuzzy_methods.items():
    df = pd.read_csv(ruta)
    # Normalizar tiempo (restar el tiempo inicial)
    tiempo_norm = df['tiempo'] - df['tiempo'].iloc[0]
    ax1.plot(tiempo_norm, df['distancia'], linewidth=2.5, label=metodo_nombre, marker='', alpha=0.8)
    ax2.plot(tiempo_norm, df['error'], linewidth=2.5, label=metodo_nombre, marker='', alpha=0.8)

# Setpoint en distancia
setpoint = pd.read_csv(list(fuzzy_methods.values())[0])['setpoint'].iloc[0]
ax1.axhline(y=setpoint, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Setpoint')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

ax1.set_xlabel('Time (ms)', fontsize=12)
ax1.set_ylabel('Distance (cm)', fontsize=12)
ax1.set_title('Distance vs Time', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Time (ms)', fontsize=12)
ax2.set_ylabel('Error (cm)', fontsize=12)
ax2.set_title('Error vs Time', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Category_Fuzzy_Logic.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: Category_Fuzzy_Logic.png")

# 2. NEURAL NETWORKS
print("2. Creating Neural Networks comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Neural Network Methods - Comparison', fontsize=16, fontweight='bold')

for metodo_nombre, ruta in neural_methods.items():
    df = pd.read_csv(ruta)
    # Normalizar tiempo (restar el tiempo inicial)
    tiempo_norm = df['tiempo'] - df['tiempo'].iloc[0]
    ax1.plot(tiempo_norm, df['distancia'], linewidth=2.5, label=metodo_nombre, marker='', alpha=0.8)
    ax2.plot(tiempo_norm, df['error'], linewidth=2.5, label=metodo_nombre, marker='', alpha=0.8)

setpoint = pd.read_csv(list(neural_methods.values())[0])['setpoint'].iloc[0]
ax1.axhline(y=setpoint, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Setpoint')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

ax1.set_xlabel('Time (ms)', fontsize=12)
ax1.set_ylabel('Distance (cm)', fontsize=12)
ax1.set_title('Distance vs Time', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Time (ms)', fontsize=12)
ax2.set_ylabel('Error (cm)', fontsize=12)
ax2.set_title('Error vs Time', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Category_Neural_Networks.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: Category_Neural_Networks.png")

# 3. REINFORCEMENT LEARNING
print("3. Creating Reinforcement Learning comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Reinforcement Learning Methods - Comparison', fontsize=16, fontweight='bold')

for metodo_nombre, ruta in rl_methods.items():
    df = pd.read_csv(ruta)
    # Normalizar tiempo (restar el tiempo inicial)
    tiempo_norm = df['tiempo'] - df['tiempo'].iloc[0]
    ax1.plot(tiempo_norm, df['distancia'], linewidth=2.5, label=metodo_nombre, marker='', alpha=0.8)
    ax2.plot(tiempo_norm, df['error'], linewidth=2.5, label=metodo_nombre, marker='', alpha=0.8)

setpoint = pd.read_csv(list(rl_methods.values())[0])['setpoint'].iloc[0]
ax1.axhline(y=setpoint, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Setpoint')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

ax1.set_xlabel('Time (ms)', fontsize=12)
ax1.set_ylabel('Distance (cm)', fontsize=12)
ax1.set_title('Distance vs Time', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Time (ms)', fontsize=12)
ax2.set_ylabel('Error (cm)', fontsize=12)
ax2.set_title('Error vs Time', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Category_Reinforcement_Learning.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: Category_Reinforcement_Learning.png")

# 4. COMPARATIVA GENERAL
print("4. Creating overall comparison...")
all_methods = {
    'Fuzzy Bisector': r'fuzzy\data_bisector.csv',
    'Fuzzy Centroide': r'fuzzy\data_centroid.csv',
    'Fuzzy MOM': r'fuzzy\data_mom.csv',
    'NN ReLU': r'neural_networks\data_relu.csv',
    'NN Sigmoid': r'neural_networks\data_sigmoid.csv',
    'NN Tanh': r'neural_networks\data_tanh.csv',
    'RL DQN': r'reinforcement_learning\data_dqn.csv',
    'RL Q-Learning': r'reinforcement_learning\data_qlearning.csv',
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('All Methods - Overall Comparison', fontsize=16, fontweight='bold')

for metodo_nombre, ruta in all_methods.items():
    df = pd.read_csv(ruta)
    # Normalizar tiempo (restar el tiempo inicial)
    tiempo_norm = df['tiempo'] - df['tiempo'].iloc[0]
    ax1.plot(tiempo_norm, df['distancia'], linewidth=2, label=metodo_nombre, alpha=0.75)
    ax2.plot(tiempo_norm, df['error'], linewidth=2, label=metodo_nombre, alpha=0.75)

ax1.axhline(y=15, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Setpoint')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

ax1.set_xlabel('Time (ms)', fontsize=12)
ax1.set_ylabel('Distance (cm)', fontsize=12)
ax1.set_title('Distance vs Time', fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Time (ms)', fontsize=12)
ax2.set_ylabel('Error (cm)', fontsize=12)
ax2.set_title('Error vs Time', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Comparison_All_Methods.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: Comparison_All_Methods.png")

# 5. GRÁFICAS INDIVIDUALES DETALLADAS (4 subplots por método)
print("\n5. Creating individual detailed graphs (4 subplots each)...")

all_csv_files = {
    'Fuzzy Bisector': r'fuzzy\data_bisector.csv',
    'Fuzzy Centroide': r'fuzzy\data_centroid.csv',
    'Fuzzy MOM': r'fuzzy\data_mom.csv',
    'Neural Network ReLU': r'neural_networks\data_relu.csv',
    'Neural Network Sigmoid': r'neural_networks\data_sigmoid.csv',
    'Neural Network Tanh': r'neural_networks\data_tanh.csv',
    'Reinforcement Learning DQN': r'reinforcement_learning\data_dqn.csv',
    'Reinforcement Learning Q-Learning': r'reinforcement_learning\data_qlearning.csv',
}

for nombre, ruta in all_csv_files.items():
    df = pd.read_csv(ruta)
    # Normalizar tiempo
    tiempo_norm = df['tiempo'] - df['tiempo'].iloc[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Performance Analysis - {nombre}', fontsize=16, fontweight='bold')
    
    # Distancia
    axes[0, 0].plot(tiempo_norm, df['distancia'], 'b-', linewidth=2)
    axes[0, 0].axhline(y=df['setpoint'].iloc[0], color='r', linestyle='--', linewidth=2, label='Setpoint')
    axes[0, 0].set_xlabel('Time (ms)', fontsize=10)
    axes[0, 0].set_ylabel('Distance (cm)', fontsize=10)
    axes[0, 0].set_title('Distance vs Time', fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error
    axes[0, 1].plot(tiempo_norm, df['error'], 'g-', linewidth=2)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel('Time (ms)', fontsize=10)
    axes[0, 1].set_ylabel('Error (cm)', fontsize=10)
    axes[0, 1].set_title('Error vs Time', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # PWM
    axes[1, 0].plot(tiempo_norm, df['pwm'], 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Time (ms)', fontsize=10)
    axes[1, 0].set_ylabel('PWM', fontsize=10)
    axes[1, 0].set_title('PWM vs Time', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Derivada o Velocidad
    if 'derivada' in df.columns:
        axes[1, 1].plot(tiempo_norm, df['derivada'], 'orange', linewidth=2, label='Derivative')
        axes[1, 1].set_title('Derivative vs Time', fontsize=11)
    if 'velocidad' in df.columns:
        axes[1, 1].plot(tiempo_norm, df['velocidad'], 'cyan', linewidth=2, label='Velocity')
        axes[1, 1].set_title('Velocity vs Time', fontsize=11)
    
    axes[1, 1].set_xlabel('Time (ms)', fontsize=10)
    axes[1, 1].set_ylabel('Value', fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f"{nombre.replace(' ', '_').replace('/', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")

print("\n" + "=" * 60)
print(f"✓ All graphs successfully saved to: {output_dir}")
print("=" * 60)
