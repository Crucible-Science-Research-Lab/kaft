"""
/maxwell_flat_vs_curved.py

kaft · Maxwell Wave · Geometry Comparison Demo

Usage:
  python /maxwell_flat_vs_curved.py            # saves GIF
  python /maxwell_flat_vs_curved.py --show     # live view
  python /maxwell_flat_vs_curved.py --panels 2 # 2 panels only
  python /maxwell_flat_vs_curved.py --panels 3 # all three
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kaft.geometry.divergences import DivergenceRegistry

# ── Configuration ─────────────────────────────────────────────
GRID     = 120
STEPS    = 300
DT       = 0.35
C_BASE   = 0.5
SOURCE_R = 2
VMIN, VMAX = -0.15, 0.15

# Parse args
SHOW   = "--show" in sys.argv
PANELS = 3
for i, arg in enumerate(sys.argv):
    if arg == "--panels" and i + 1 < len(sys.argv):
        PANELS = int(sys.argv[i + 1])

# ── Physics ────────────────────────────────────────────────────
registry = DivergenceRegistry()


def make_pulse(grid, radius=SOURCE_R):
    cx, cy = grid // 2, grid // 2
    x, y = np.arange(grid), np.arange(grid)
    xx, yy = np.meshgrid(x, y)
    return np.exp(-((xx - cx)**2 + (yy - cy)**2) / radius**2)


def laplacian(u):
    return (np.roll(u, 1, 0) + np.roll(u, -1, 0) +
            np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u)


def simulate(speed_field):
    u, u_old = make_pulse(GRID), make_pulse(GRID)
    frames = [u.copy()]
    for _ in range(STEPS):
        u_new = 2*u - u_old + (DT**2) * (speed_field**2) * laplacian(u)
        u_old, u = u, u_new
        frames.append(u.copy())
    return frames


# ── Build speed fields ─────────────────────────────────────────
flat_speed    = np.full((GRID, GRID), C_BASE)
curved_metric = registry.get("gaussian_curved")
mink_metric   = registry.get("minkowski")
curved_speed  = C_BASE * curved_metric.speed_field(GRID)
mink_speed    = C_BASE * mink_metric.speed_field(GRID)

configs = [
    ("Euclidean  ·  flat space",       "Blues",  flat_speed,   None),
    ("Gaussian Curved  ·  Non uniform space", "RdPu",   curved_speed, curved_metric),
    ("Minkowski  ·  spacetime",        "YlOrRd", mink_speed,   mink_metric),
][:PANELS]

print(f"Simulating {PANELS} geometries...")
all_frames = [simulate(cfg[2]) for cfg in configs]
print("Rendering...")

# ── Figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, PANELS, figsize=(5.5 * PANELS, 5.5))
if PANELS == 1:
    axes = [axes]
fig.patch.set_facecolor('#0d0d11')

ims = []
for i, (ax, (title, cmap, speed, metric)) in enumerate(zip(axes, configs)):
    ax.set_facecolor('#0d0d11')
    ax.set_title(title, color='#c8c8d0', fontsize=10, pad=8,
                 fontfamily='monospace')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a35')

    # Overlay curvature contour where applicable
    if metric is not None and hasattr(metric, 'speed_field'):
        sf = metric.speed_field(GRID)
        ax.contour(sf, levels=[0.3, 0.6], colors=['#ffffff'],
                   alpha=0.15, linewidths=0.7)

    im = ax.imshow(all_frames[i][0], vmin=VMIN, vmax=VMAX,
                   cmap=cmap, interpolation='bilinear', origin='lower')
    ims.append(im)

# Bottom label strip
labels = {
    3: [" uniform  metric ", "position-dependent metric", "Minkowski metric"],
    2: [" uniform metric", "position-dependent metric"],
    1: [" uniform metric"],
}
for ax, lbl in zip(axes, labels[PANELS]):
    ax.set_xlabel(lbl, color='#666', fontsize=8, fontfamily='monospace')

step_label = fig.text(0.5, 0.01, "t = 0", ha='center',
                      color='#555', fontsize=8, fontfamily='monospace')

fig.suptitle("kaft  ·  Same Wave Equation  ·  Different Geometries",
             color='#aaa', fontsize=12, y=0.97, fontfamily='monospace')

plt.tight_layout(rect=[0, 0.04, 1, 0.95])


def update(frame):
    for im, frames in zip(ims, all_frames):
        im.set_data(frames[frame])
    step_label.set_text(f"t = {frame * DT:.1f}")
    return ims + [step_label]


ani = animation.FuncAnimation(fig, update, frames=STEPS,
                               interval=40, blit=False)

if SHOW:
    plt.show()
else:
    out = f"maxwell_{PANELS}panel.gif"
    ani.save(out, writer=animation.PillowWriter(fps=25))
    print(f"✓ Saved → {out}")

plt.close(fig)

