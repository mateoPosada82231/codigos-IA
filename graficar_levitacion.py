import pandas as pd
import matplotlib.pyplot as plt

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

# Requested plots only: error, error derivative, and PWM
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Magnetic Levitation Control Signals", fontsize=13, fontweight="bold")

plot_config = [
    ("error", "Error", "Error (cm)"),
    ("derivada", "Error Derivative", "Error derivative (cm/s)"),
    ("pwm", "PWM Used to Reach Setpoints", "PWM"),
]

for ax, (col_name, title, y_label) in zip(axes, plot_config):
    for nombre, df in datos.items():
        ax.plot(
            df["tiempo"],
            df[col_name],
            label=nombre,
            color=colores[nombre],
            linewidth=1.1,
        )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

axes[-1].set_xlabel("Time (s)")
fig.tight_layout(rect=(0, 0, 1, 0.97))

plt.show()
