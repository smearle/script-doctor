import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json
import jax
import os
from globals import JAX_PROFILING_RESULTS_DIR

LEVEL=0
GAME="Multi-word_Dictionary_Game"

devices = jax.devices()
device_name = devices[0].device_kind
device_name = device_name.replace(' ', '_')
device_dir = os.path.join(JAX_PROFILING_RESULTS_DIR, device_name)
os.makedirs(device_dir, exist_ok=True)
level_str = f'level-{LEVEL}'
results_path = os.path.join(device_dir, GAME, level_str + '.json')
data = json.load(open(results_path, 'r'))

# Parse envs, steps, and metrics
envs = []
steps = []
wins = []
winrates = []
c_times = []
sim_times = []

for k, v in data.items():
    parts = k.split()
    env = int(parts[1])   # after "env"
    step = int(parts[3])  # after "step"
    envs.append(env)
    steps.append(step)
    wins.append(v["wins"])
    winrates.append(v["winrate"])
    c_times.append(v["c_time"])
    sim_times.append(v["sim_time"])

# Convert to numpy arrays for plotting
envs = np.array(envs)
steps = np.array(steps)
winrates = np.array(winrates)
c_times = np.array(c_times)
sim_times = np.array(sim_times)

fig = plt.figure(figsize=(15, 5))
fig.suptitle(f"{GAME} - Level {LEVEL}")

pos = 311
# Function to plot a 3D scatter
def plot_3d(fig : plt.Figure, x, y, z, zlabel, color):
    global pos
    ax = fig.add_subplot(pos, projection='3d')
    ax.scatter(x, y, z, c=color, s=50)

    ax.set_xlabel("Envs")
    ax.set_ylabel("Steps")
    ax.set_zlabel(zlabel)
    ax.set_title(f"{zlabel} vs Envs and Steps")
    pos += 1


# Plot all three metrics
plot_3d(fig, envs, steps, winrates, "Winrate", "blue")
plot_3d(fig, envs, steps, c_times, "Compile Time", "green")
plot_3d(fig, envs, steps, sim_times, "Simulation Time", "red")
plt.tight_layout()
plt.show()