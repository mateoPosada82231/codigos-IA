import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Cargar los 3 archivos CSV ──────────────────────────────────────────────────
archivos = {
    "Setpoint 10 cm": "datos_levitacion_10cm.csv",
    "Setpoint 15 cm": "datos_levitacion_15cm.csv",
    "Setpoint 20 cm": "datos_levitacion_20cm.csv",
}

colores = {
    "Setpoint 10 cm": "#1f77b4",
    "Setpoint 15 cm": "#ff7f0e",
    "Setpoint 20 cm": "#2ca02c",
}

datos = {nombre: pd.read_csv(archivo) for nombre, archivo in archivos.items()}

# ── Figura 1: Distancia vs Setpoint (comparación de los 3 ensayos) ─────────────
fig1, ax1 = plt.subplots(figsize=(12, 5))
for nombre, df in datos.items():
    ax1.plot(df["tiempo"], df["distancia"], label=f"Distancia – {nombre}",
             color=colores[nombre], linewidth=1.2)
    ax1.axhline(df["setpoint"].iloc[0], linestyle="--", color=colores[nombre],
                linewidth=0.9, alpha=0.6, label=f"Setpoint – {nombre}")
ax1.set_xlabel("Tiempo (s)")
ax1.set_ylabel("Distancia (cm)")
ax1.set_title("Comparación de distancia medida vs setpoint")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()

# ── Figura 2: Un panel por ensayo con las variables principales ────────────────
fig2 = plt.figure(figsize=(16, 12))
fig2.suptitle("Variables de control – Levitación magnética", fontsize=13, fontweight="bold")

variables = [
    ("distancia",  "Distancia (cm)"),
    ("error",      "Error (cm)"),
    ("derivada",   "Derivada (cm/s)"),
    ("integral",   "Integral (cm·s)"),
    ("delta_pwm",  "Δ PWM"),
    ("pwm",        "PWM"),
]
n_vars = len(variables)
n_ensayos = len(datos)

gs = gridspec.GridSpec(n_vars, n_ensayos, figure=fig2, hspace=0.55, wspace=0.3)

for col, (nombre, df) in enumerate(datos.items()):
    for row, (col_name, y_label) in enumerate(variables):
        ax = fig2.add_subplot(gs[row, col])
        ax.plot(df["tiempo"], df[col_name],
                color=colores[nombre], linewidth=0.9)
        if col_name == "distancia":
            ax.axhline(df["setpoint"].iloc[0], linestyle="--",
                       color="gray", linewidth=0.8, alpha=0.7)
        if row == 0:
            ax.set_title(nombre, fontsize=10, fontweight="bold")
        ax.set_xlabel("Tiempo (s)", fontsize=7)
        ax.set_ylabel(y_label, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)

fig2.tight_layout(rect=[0, 0, 1, 0.97])

# ── Figura 3: PWM y distancia superpuestos para cada ensayo ───────────────────
fig3, axes = plt.subplots(n_ensayos, 1, figsize=(12, 10), sharex=False)
fig3.suptitle("PWM y Distancia por ensayo", fontsize=13, fontweight="bold")

for ax, (nombre, df) in zip(axes, datos.items()):
    color = colores[nombre]
    ax2 = ax.twinx()
    l1, = ax.plot(df["tiempo"], df["distancia"], color=color,
                  linewidth=1.2, label="Distancia (cm)")
    ax.axhline(df["setpoint"].iloc[0], linestyle="--", color=color,
               linewidth=0.8, alpha=0.6, label="Setpoint")
    l2, = ax2.plot(df["tiempo"], df["pwm"], color="crimson",
                   linewidth=0.9, alpha=0.75, label="PWM")
    ax.set_title(nombre, fontsize=10)
    ax.set_xlabel("Tiempo (s)", fontsize=8)
    ax.set_ylabel("Distancia (cm)", fontsize=8, color=color)
    ax2.set_ylabel("PWM", fontsize=8, color="crimson")
    ax.tick_params(axis="y", labelcolor=color)
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax.grid(True, alpha=0.25)
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=7, loc="upper right")

fig3.tight_layout(rect=[0, 0, 1, 0.97])

plt.show()
