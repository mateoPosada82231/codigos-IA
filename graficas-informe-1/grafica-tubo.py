import matplotlib.pyplot as plt
import matplotlib.patches as patches

# System Parameters
setpoint = 15.0
h_max = 40.0  # Max visible tube height
r_ball = 0.8  # Visual radius of the ball

# Fuzzy Zone Definitions (Error = Actual Distance - Setpoint)
# Height = Error + Setpoint
# Format: (Label, Error_min, Error_max, Color, Label_Alignment)
zones = [
    ('NV', -40.0, -8.0, '#d73027', 'left'),  # Negative Very Big (Danger Top)
    ('NB', -12.0, -4.0, '#f46d43', 'right'),  # Negative Big
    ('NM', -6.0, -1.5, '#fdae61', 'left'),  # Negative Medium
    ('NS', -3.0, 0.0, '#fee08b', 'right'),  # Negative Small
    ('Z', -1.5, 1.5, '#ffffbf', 'center'),  # Zero (Comfort Zone)
    ('PS', 0.0, 3.0, '#d9ef8b', 'right'),  # Positive Small
    ('PM', 1.5, 6.0, '#a6d96a', 'left'),  # Positive Medium
    ('PB', 4.0, 12.0, '#66bd63', 'right'),  # Positive Big
    ('PV', 8.0, 40.0, '#1a9850', 'left'),  # Positive Very Big (Rescue/Lift)
]

fig, ax = plt.subplots(figsize=(8, 10))

# 1. Draw the Acrylic Tube
tube = patches.Rectangle((2, 0), 2, h_max, linewidth=2, edgecolor='#333333',
                         facecolor='#f0f8ff', alpha=0.3, zorder=1)
ax.add_patch(tube)

# 2. Draw Fuzzy Membership Zones
for name, e_min, e_max, color, align in zones:
    h_min = max(0, setpoint + e_min)
    h_top = min(h_max, setpoint + e_max)

    # Draw semi-transparent overlay
    rect = patches.Rectangle((2, h_min), 2, h_top - h_min, facecolor=color,
                             alpha=0.4, zorder=2)
    ax.add_patch(rect)

    # Zone Labels
    x_pos = 1.8 if align == 'left' else 4.2
    ax.text(x_pos, (h_min + h_top) / 2, name, fontweight='bold', fontsize=10,
            ha=align, va='center', color='#2c3e50')

# 3. Draw Setpoint Goal Line
ax.axhline(y=setpoint, xmin=0.25, xmax=0.75, color='blue', linestyle='--',
           linewidth=1.5, label='Setpoint (15cm)', zorder=5)

# 4. Draw the Styrofoam Ball
ball = plt.Circle((3, setpoint), r_ball, color='#ecf0f1', ec='#7f8c8d', lw=1, zorder=10)
ax.add_patch(ball)
ax.text(3, setpoint - 2, 'Ball (0.5g)', ha='center', fontsize=9,
        color='#34495e', fontweight='bold')

# Axis and Styling
ax.set_xlim(0, 6)
ax.set_ylim(0, h_max)
ax.set_ylabel('Tube Height (cm)', fontsize=12, fontweight='bold')
ax.set_title('Fuzzy Logic Control Infographic: Air Levitator',
             fontsize=14, pad=20, fontweight='bold')

# Hide X axis and clean up spines
ax.get_xaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Add measurement ruler
ax.yaxis.set_major_locator(plt.MultipleLocator(5))
ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
ax.grid(axis='y', which='both', linestyle=':', alpha=0.5)

# Sidebar Notes (Control Logic)
ax.text(5, 38, "ERROR ZONES\n$E = H_{actual} - H_{set}$", fontsize=10,
        bbox=dict(facecolor='white', alpha=0.5))
ax.text(5, 5, "RED: Brake (Reduce PWM)\nGREEN: Accelerate (Increase PWM)",
        fontsize=9, color='#c0392b', ha='center', transform=ax.get_yaxis_transform())

plt.tight_layout()
plt.show()