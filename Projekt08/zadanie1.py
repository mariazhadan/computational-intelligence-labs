import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).with_name(".deps")))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".mplconfig")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history


def endurance(args):
    x, y, z, u, v, w = args
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)


def f(swarm):
    return np.array([-endurance(particle) for particle in swarm])


np.random.seed(0)

options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
bounds = (np.zeros(6), np.ones(6))

optimizer = ps.single.GlobalBestPSO(
    n_particles=10,
    dimensions=6,
    options=options,
    bounds=bounds,
)

cost, pos = optimizer.optimize(f, iters=1000, verbose=False)

print("best cost:", cost)
print("best pos:", pos)
print("endurance:", endurance(pos))

plot_cost_history(optimizer.cost_history)
plt.savefig("zadanie1_koszt.png")
