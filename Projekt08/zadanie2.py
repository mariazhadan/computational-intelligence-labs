import contextlib
import io
import math
import random
import matplotlib.pyplot as plt
from aco import AntColony

# COORDS = tuple((random.randint(0, 100), random.randint(0, 100)) for _ in range(15))

COORDS = (
 (0, 0),(10, 0),(20, 0),(30, 0),(40, 0),
 (0, 10),(10, 10),(20, 10),(30, 10),(40, 10),
 (0, 20),(10, 20),(20, 20),(30, 20),(40, 20),(0, 30),
 (10, 30),(20, 30),(30, 30),(40, 30),(0, 40),
 (10, 40),(20, 40),(30, 40),(40,40),
)

def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()

colony = AntColony(
            COORDS,
            ant_count=300,
            alpha=0.5,
            beta=1.2,
            pheromone_evaporation_rate=0.40,
            pheromone_constant=1000.0,
            iterations=300,
        )

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )

plt.show()

# Version 1 
# Dystans: 358.76434561891057
# colony = AntColony(
#             COORDS,
#             ant_count=300,
#             alpha=0.5,
#             beta=1.2,
#             pheromone_evaporation_rate=0.40,
#             pheromone_constant=1000.0,
#             iterations=300,
#         )
# Version 2 
# Dystans: 325.09859145739256
# colony = AntColony(
#             COORDS,
#             ant_count=300,
#             alpha=0.2,
#             beta=1.0,
#             pheromone_evaporation_rate=0.20,
#             pheromone_constant=1000.0,
#             iterations=100,
#         )
# Version 3 
# Dystans: 378.47750219487807
# colony = AntColony(
#             COORDS,
#             ant_count=300,
#             alpha=0.8,
#             beta=1.4,
#             pheromone_evaporation_rate=0.60,
#             pheromone_constant=1000.0,
#             iterations=300,
#         )

# GRID
# Dystans: 284.7213595499958
