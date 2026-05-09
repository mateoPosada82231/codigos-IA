import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KI = 0.10

FILES = {
    "Setpoint 10 cm": "datos_levitacion_10cm.csv",
    "Setpoint 15 cm": "datos_levitacion_15cm.csv",
    "Setpoint 20 cm": "datos_levitacion_20cm.csv",
}

COLORS = {
    "Setpoint 10 cm": "#1f77b4",
    "Setpoint 15 cm": "#ff7f0e",
    "Setpoint 20 cm": "#2ca02c",
}

ERROR_LEVELS = [
    ("NV", (-50, -50, -15, -8)),
    ("NB", (-12, -9, -7, -4)),
    ("NM", (-6, -4.5, -3.5, -1.5)),
    ("NS", (-2.5, -1.5, -0.5, 0)),
    ("Z", (-1.0, -0.3, 0.3, 1.0)),
    ("PS", (0, 0.5, 1.5, 2.5)),
    ("PM", (1.5, 3, 5, 6)),
    ("PB", (4, 6, 10, 12)),
    ("PV", (8, 15, 50, 50)),
]

DERIV_LEVELS = [
    ("NV", (-80, -80, -25, -10)),
    ("NB", (-20, -12, -8, -3)),
    ("NS", (-6, -4, -2, 0)),
    ("Z", (-1.5, -0.5, 0.5, 1.5)),
    ("PS", (0, 2, 4, 6)),
    ("PB", (3, 7, 13, 20)),
    ("PV", (10, 25, 80, 80)),
]

FAM = np.array([
    [-6.0, -6.0, -3.0, -1.5, -0.5, 0.0, 0.0],
    [-6.0, -3.0, -1.5, -0.5, 0.0, 0.0, 0.8],
    [-3.0, -1.5, -0.5, -0.5, 0.0, 0.8, 2.5],
    [-1.5, -0.5, -0.5, 0.0, 0.0, 0.8, 2.5],
    [-1.5, -0.5, 0.0, 0.0, 0.0, 0.8, 2.5],
    [-1.5, -0.5, 0.0, 0.0, 0.8, 0.8, 2.5],
    [-1.5, -0.5, 0.0, 0.8, 0.8, 2.5, 6.0],
    [-0.5, 0.0, 0.0, 0.8, 2.5, 6.0, 18.0],
    [0.0, 0.0, 0.8, 2.5, 6.0, 18.0, 18.0],
])


def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    left = (a < x) & (x <= b)
    if b != a:
        y[left] = (x[left] - a) / (b - a)
    else:
        y[left] = 1.0

    middle = (b < x) & (x <= c)
    y[middle] = 1.0

    right = (c < x) & (x < d)
    if d != c:
        y[right] = (d - x[right]) / (d - c)
    else:
        y[right] = 1.0

    y[(x == b) | (x == c)] = 1.0
    return y


def load_data():
    return {name: pd.read_csv(path) for name, path in FILES.items()}


def plot_memberships(ax, x_values, levels, title, xlabel):
    for label, params in levels:
        ax.plot(x_values, trapmf(x_values, *params), linewidth=1.6, label=label)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Membership degree")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(5, len(levels)), fontsize=8, loc="upper right")


def plot_fuzzy_output(ax, data):
    for name, df in data.items():
        delta_fuzzy = df["delta_pwm"] - (KI * df["integral"])
        ax.plot(df["tiempo"], delta_fuzzy, color=COLORS[name], linewidth=1.1, label=name)
    ax.set_title("Reconstructed Fuzzy Output (delta_fuzzy)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Delta PWM")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def plot_fam_heatmap(ax):
    im = ax.imshow(FAM, cmap="coolwarm", aspect="auto", origin="upper")
    ax.set_title("Fuzzy Rule Matrix (FAM)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Derivative linguistic level")
    ax.set_ylabel("Error linguistic level")
    ax.set_xticks(range(len(DERIV_LEVELS)))
    ax.set_xticklabels([name for name, _ in DERIV_LEVELS])
    ax.set_yticks(range(len(ERROR_LEVELS)))
    ax.set_yticklabels([name for name, _ in ERROR_LEVELS])

    for i in range(FAM.shape[0]):
        for j in range(FAM.shape[1]):
            ax.text(j, i, f"{FAM[i, j]:.1f}", ha="center", va="center", fontsize=7, color="black")

    return im


def main():
    data = load_data()

    error_x = np.linspace(-20, 20, 1000)
    deriv_x = np.linspace(-40, 40, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Fuzzy Values - Levitation Controller (7 derivative levels)", fontsize=13, fontweight="bold")

    plot_memberships(
        axes[0, 0],
        error_x,
        ERROR_LEVELS,
        "Error Membership Functions (9 levels)",
        "Error (cm)",
    )
    plot_memberships(
        axes[0, 1],
        deriv_x,
        DERIV_LEVELS,
        "Error Derivative Membership Functions (7 levels)",
        "Error derivative (cm/s)",
    )
    plot_fuzzy_output(axes[1, 0], data)
    im = plot_fam_heatmap(axes[1, 1])

    cbar = fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar.set_label("Rule output (delta PWM)")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


if __name__ == "__main__":
    main()

