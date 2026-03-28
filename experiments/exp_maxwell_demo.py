import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from kaft.simulate.maxwell import MaxwellDynamics

nx, ny = 100, 100
manifold = {
    'Ex': np.zeros((ny, nx)),
    'Ey': np.zeros((ny, nx)),
    'Bz': np.zeros((ny, nx)),
    'dx': 1.0, 'dy': 1.0
}

# Point source at center
manifold['Bz'][ny // 2, nx // 2] = 1.0

sim = MaxwellDynamics()
frames = []
for _ in range(200):
    manifold = sim.run(manifold, dt=0.5)
    frames.append(manifold['Bz'].copy())

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(frames[0], cmap='RdBu', vmin=-0.05, vmax=0.05)
plt.colorbar(im)
ax.set_title('Maxwell B-field — kaft dynamics engine')

def update(i):
    im.set_data(frames[i])
    return [im]

ani = FuncAnimation(fig, update, frames=len(frames), interval=40, blit=True)
plt.show()