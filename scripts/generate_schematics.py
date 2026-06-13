"""
esquemas_informe.py
Generates the three informative schematics required by informe.tex:

    1. esquema_montaje.png   -> fig:montaje         (physical assembly)
    2. estructura_repo.png   -> fig:estructura      (repository layout)
    3. arquitectura_rn.png   -> fig:arquitectura-rn (neural network)

Style: matches the existing figures of informe.tex (white background, bold
titles, restrained palette, English labels, IEEEtran-friendly aspect).

Run from any working directory:

    python esquemas_informe.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

NAVY    = "#1f77b4"
ORANGE  = "#ff7f0e"
GREEN   = "#2ca02c"
RED     = "#d62728"
PURPLE  = "#8e44ad"
GRAY    = "#2c3e50"
LIGHT   = "#ecf0f1"
BG_BLUE = "#f0f8ff"

DPI = 220


# ────────────────────────────────────────────────────────────────────────
# 1. Physical Assembly Diagram  (fig:montaje)
# ────────────────────────────────────────────────────────────────────────
def draw_montaje():
    fig, ax = plt.subplots(figsize=(8.0, 10.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(5.0, 13.6, "Physical Assembly Diagram",
            ha="center", fontsize=15, fontweight="bold", color=GRAY)
    ax.text(5.0, 13.05, "Air levitation system: sensor \u2192 ESP32 \u2192 PWM fan",
            ha="center", fontsize=10, color=GRAY, style="italic")

    # acrylic tube  (from y=2.6 to y=11.4)
    tube = Rectangle((4.0, 2.6), 2.0, 8.8, linewidth=2,
                     edgecolor=GRAY, facecolor=BG_BLUE, alpha=0.45, zorder=1)
    ax.add_patch(tube)
    ax.text(3.85, 5.4, "Acrylic\ntube\n(40 cm)",
            ha="right", va="center", fontsize=9.5, fontweight="bold", color=GRAY)

    # fan at the base
    fan = FancyBboxPatch((3.3, 1.05), 3.4, 1.25,
                         boxstyle="round,pad=0.05,rounding_size=0.12",
                         facecolor="#34495e", edgecolor=GRAY, lw=1.5, zorder=2)
    ax.add_patch(fan)
    ax.text(5.0, 1.68, "DC Fan (PWM 25 kHz)",
            ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    # airflow arrow
    ax.annotate("", xy=(5.0, 5.5), xytext=(5.0, 2.35),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.2, zorder=3))
    ax.text(5.55, 3.9, "Air flow", color=NAVY, fontsize=9.5,
            fontweight="bold")

    # styrofoam ball
    ball = Circle((5.0, 6.9), 0.4, color=LIGHT, ec=GRAY, lw=1.2, zorder=5)
    ax.add_patch(ball)
    ax.annotate("Styrofoam ball\n(0.5 g)",
                xy=(5.0, 6.9), xytext=(7.7, 7.9),
                fontsize=9.5, ha="left", color=GRAY,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))

    # HC-SR04 at the top
    sensor = FancyBboxPatch((3.9, 11.55), 2.2, 0.85,
                            boxstyle="round,pad=0.05,rounding_size=0.10",
                            facecolor="#16a085", edgecolor=GRAY, lw=1.5, zorder=2)
    ax.add_patch(sensor)
    ax.text(5.0, 11.97, "HC-SR04 Ultrasonic",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")
    # distance beam (dashed red, double headed) - label on the right
    ax.annotate("", xy=(5.0, 7.35), xytext=(5.0, 11.55),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.4,
                                linestyle="--", zorder=3))
    ax.text(6.20, 9.5, "Distance\nmeasure", color=RED, fontsize=9.5,
            fontweight="bold", ha="left")

    # ESP32 box
    esp = FancyBboxPatch((0.6, 5.4), 2.5, 1.6,
                         boxstyle="round,pad=0.08,rounding_size=0.15",
                         facecolor=NAVY, edgecolor=GRAY, lw=1.5, zorder=2)
    ax.add_patch(esp)
    ax.text(1.85, 6.35, "ESP32", ha="center", va="center",
            fontsize=13, fontweight="bold", color="white")
    ax.text(1.85, 5.75, "(MicroPython, 20 Hz loop)",
            ha="center", va="center", fontsize=8.5, color="white", style="italic")

    # TRIG / ECHO link (sensor -> ESP32) - label on the far left
    ax.annotate("", xy=(3.0, 6.3), xytext=(3.9, 11.8),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0,
                                connectionstyle="arc3,rad=-0.30"))
    ax.text(1.5, 9.7, "TRIG / ECHO\n(GPIO 27 / 26)",
            fontsize=8.5, color=GRAY, fontweight="bold", ha="center")

    # PWM link (ESP32 -> fan)
    ax.annotate("", xy=(3.4, 1.7), xytext=(2.95, 5.5),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6,
                                connectionstyle="arc3,rad=0.25"))
    ax.text(2.4, 3.4, "PWM\n(GPIO 14)", fontsize=8.5,
            color=GREEN, fontweight="bold", ha="center")

    # legend
    legend_items = [
        ("Air flow",         NAVY,  "-"),
        ("Distance measure", RED,   "--"),
        ("PWM control",      GREEN, "-"),
        ("TRIG / ECHO",      GRAY,  "-"),
    ]
    lx, ly = 0.4, 0.55
    for i, (label, color, _style) in enumerate(legend_items):
        y = ly + (len(legend_items) - 1 - i) * 0.25
        ax.plot([lx, lx + 0.4], [y, y], color=color, lw=2.0)
        ax.text(lx + 0.5, y, label, fontsize=8.5, va="center", color=GRAY)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "esquema_montaje.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out)


# ────────────────────────────────────────────────────────────────────────
# 2. Repository Structure  (fig:estructura)
# ────────────────────────────────────────────────────────────────────────
def draw_estructura():
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(6.0, 11.55, "Repository Structure",
            ha="center", fontsize=15, fontweight="bold", color=GRAY)
    ax.text(6.0, 11.10, "Three control modules with their configuration variants",
            ha="center", fontsize=10, color=GRAY, style="italic")

    # root
    root = FancyBboxPatch((0.5, 9.55), 11.0, 0.75,
                          boxstyle="round,pad=0.05,rounding_size=0.10",
                          facecolor=NAVY, edgecolor=GRAY, lw=1.5)
    ax.add_patch(root)
    ax.text(6.0, 9.92, "codigos-IA/",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="white", family="monospace")

    modules = [
        {
            "name": "fuzzy/",
            "x": 0.55, "color": "#e67e22",
            "tag": "Fuzzy PD+I  (FAM 9\u00d77)",
            "files": [
                "controller_centroid.py",
                "controller_bisector.py",
                "controller_mom.py",
            ],
        },
        {
            "name": "neural_networks/",
            "x": 4.45, "color": "#27ae60",
            "tag": "FCLayer(3 \u2192 16 \u2192 12 \u2192 8 \u2192 1)",
            "files": [
                "train.py",
                "export_weights.py",
                "controller_sigmoid.py",
                "controller_tanh.py",
                "controller_relu.py",
            ],
        },
        {
            "name": "reinforcement_learning/",
            "x": 8.35, "color": PURPLE,
            "tag": "Q-Learning  +  DQN",
            "files": [
                "qlearning_esp32.py  (Q-Learning)",
                "dqn_train.py  (PC training)",
                "dqn_esp32.py  (inference)",
                "export_dqn_weights.py",
            ],
        },
    ]

    box_w = 3.30
    for m in modules:
        # connector root -> module (from bottom-center of root to top-center of module)
        ax.plot([6.0, m["x"] + box_w / 2], [9.55, 8.45],
                color=GRAY, lw=1.0)

        # module box
        box = FancyBboxPatch((m["x"], 7.80), box_w, 0.65,
                             boxstyle="round,pad=0.05,rounding_size=0.10",
                             facecolor=m["color"], edgecolor=GRAY, lw=1.4)
        ax.add_patch(box)
        ax.text(m["x"] + box_w / 2, 8.27, m["name"],
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="white", family="monospace")
        ax.text(m["x"] + box_w / 2, 8.00, m["tag"],
                ha="center", va="center", fontsize=7.8,
                color="white", style="italic")

        # files (more vertical space per row)
        for i, fname in enumerate(m["files"]):
            y = 7.40 - i * 0.42
            ax.text(m["x"] + 0.10, y, "\u251c\u2500\u2500",
                    fontsize=9, color=GRAY, family="monospace")
            ax.text(m["x"] + 0.55, y, fname,
                    fontsize=8.4, family="monospace", color=GRAY)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "estructura_repo.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out)


# ────────────────────────────────────────────────────────────────────────
# 3. Neural Network Architecture  (fig:arquitectura-rn)
# ────────────────────────────────────────────────────────────────────────
def draw_arquitectura_rn():
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7.0, 8.5, "Neural Network Architecture",
            ha="center", fontsize=15, fontweight="bold", color=GRAY)
    ax.text(7.0, 8.05, "FCLayer(3 \u2192 16 \u2192 12 \u2192 8 \u2192 1)",
            ha="center", fontsize=11, color=GRAY, family="monospace")

    layer_sizes  = [3, 16, 12, 8, 1]
    layer_labels = ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"]
    layer_x      = [1.7, 4.5, 7.4, 10.1, 13.0]
    layer_colors = [NAVY, ORANGE, ORANGE, ORANGE, GREEN]

    cy = 4.2
    spacing = 0.30
    neuron_ys = []
    for n in layer_sizes:
        if n == 1:
            ys = np.array([cy])
        else:
            ys = np.linspace(cy - (n - 1) * spacing / 2,
                             cy + (n - 1) * spacing / 2, n)
        neuron_ys.append(ys)

    # connections
    for i in range(len(layer_sizes) - 1):
        x1, x2 = layer_x[i] + 0.20, layer_x[i + 1] - 0.20
        ys1, ys2 = neuron_ys[i], neuron_ys[i + 1]
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1, x2], [y1, y2],
                        color="#95a5a6", lw=0.45, alpha=0.55, zorder=1)

    # neurons + labels
    for n, x, ys, color, label in zip(layer_sizes, layer_x, neuron_ys,
                                      layer_colors, layer_labels):
        for y in ys:
            ax.add_patch(Circle((x, y), 0.20, facecolor=color,
                                edgecolor=GRAY, lw=1.0, zorder=3))
        bottom = ys.min() - 0.45
        top    = ys.max() + 0.45
        ax.text(x, bottom, label, ha="center", va="top",
                fontsize=10.5, fontweight="bold", color=GRAY)
        ax.text(x, top, f"{n} neuron{'s' if n != 1 else ''}",
                ha="center", va="bottom", fontsize=9, color=GRAY)

    # input feature names
    input_labels = ["error", "\u0394error", "\u222berror"]
    for j, lbl in enumerate(input_labels):
        ax.text(layer_x[0] - 0.55, neuron_ys[0][j], lbl,
                ha="right", va="center", fontsize=9.5,
                family="monospace", color=GRAY)

    # output feature name
    ax.text(layer_x[-1] + 0.55, neuron_ys[-1][0], "\u0394PWM",
            ha="left", va="center", fontsize=10.5, family="monospace",
            fontweight="bold", color=GRAY)

    # activation annotations (placed below network)
    ax.text((layer_x[1] + layer_x[3]) / 2, 0.7,
            "Hidden activation:  \u03c3  /  tanh  /  ReLU   (configurable)",
            ha="center", fontsize=9.5, color=GRAY,
            bbox=dict(facecolor=LIGHT, edgecolor=GRAY,
                      boxstyle="round,pad=0.4", lw=0.8))
    ax.text(layer_x[-1], 0.7, "Linear output",
            ha="center", fontsize=9.5, color=GRAY, style="italic",
            bbox=dict(facecolor=LIGHT, edgecolor=GRAY,
                      boxstyle="round,pad=0.4", lw=0.8))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "arquitectura_rn.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out)


# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    draw_montaje()
    draw_estructura()
    draw_arquitectura_rn()
    print("All schematics generated.")
