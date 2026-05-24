import contextlib
import io
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).with_name(".deps")))

from aco import AntColony


def reset_aco():
    AntColony.antArray = []
    AntColony.pheromoneMap = {}
    AntColony.tmpPheromoneMap = {}


def run_aco(coords, alpha=0.5, beta=1.2, pheromone_evaporation_rate=0.40, pheromone_constant=1000.0):
    reset_aco()
    random.seed(1)
    with contextlib.redirect_stdout(io.StringIO()):
        colony = AntColony(
            coords,
            ant_count=100,
            alpha=alpha,
            beta=beta,
            pheromone_evaporation_rate=pheromone_evaporation_rate,
            pheromone_constant=pheromone_constant,
            iterations=100,
        )
    return colony.bestDistance, colony.bestSeenPath


def path_length(path):
    return sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))


COORDS_7 = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
)

random.seed(8)
COORDS_15 = tuple((random.randint(0, 100), random.randint(0, 100)) for _ in range(15))

GRID_5X5 = tuple((10 * x, 10 * y) for y in range(5) for x in range(5))

HUMAN_GRID_PATH = (
    (30, 10),
    (30, 20),
    (30, 30),
    (20, 30),
    (10, 30),
    (10, 20),
    (10, 10),
    (10, 0),
    (0, 0),
    (0, 10),
    (0, 20),
    (0, 30),
    (0, 40),
    (10, 40),
    (20, 40),
    (30, 40),
    (40, 40),
    (40, 30),
    (40, 20),
    (40, 10),
    (40, 0),
    (30, 0),
    (20, 0),
    (20, 10),
    (20, 20),
    (30, 10),
)

print("7 punktow:", run_aco(COORDS_7)[0])
print("15 losowych punktow:", run_aco(COORDS_15)[0])

experiments = (
    (0.5, 1.2, 0.40, 1000.0),
    (1.0, 2.0, 0.30, 1000.0),
    (0.2, 3.0, 0.50, 500.0),
)

for alpha, beta, evaporation, pheromone in experiments:
    distance, _ = run_aco(COORDS_15, alpha, beta, evaporation, pheromone)
    print("parametry:", alpha, beta, evaporation, pheromone, "dlugosc:", distance)

# Wniosek: dla tej paczki zmiany alpha, beta, parowania i stalej feromonu nie poprawily wyniku
# w moich testach na tych samych 15 punktach. Implementacja wybiera najbardziej prawdopodobna
# krawedz, a w praktyce sprowadza sie to prawie do wyboru najblizszego nastepnego punktu.

human_grid_distance = path_length(HUMAN_GRID_PATH)
aco_grid_distance, _ = run_aco(GRID_5X5)

print("grid 5x5 po ludzku:", human_grid_distance)
print("grid 5x5 ACO:", aco_grid_distance)

# Dla gridu 5x5 da sie ulozyc cykl z 24 odcinkow po 10 i jedna przekatna sqrt(200),
# czyli 240 + 14.1421356237 = 254.1421356237. ACO znalazl w tym uruchomieniu trase dluzsza.
