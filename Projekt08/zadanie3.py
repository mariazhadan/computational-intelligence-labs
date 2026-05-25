import numpy as np
from collections import defaultdict

MAZE = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,1,0,0,0,0,0,1],
    [1,1,1,0,0,0,0,1,0,1,0,1],
    [1,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,1,0,1,1,0,0,1,1,0,1],
    [1,0,0,1,1,0,0,0,1,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,1,1],
    [1,0,1,0,0,1,1,0,1,0,0,1],
    [1,0,1,1,1,0,0,0,1,1,0,1],
    [1,0,1,0,1,1,0,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1]
])
START = (1, 1)
GOAL = (10, 10)

MOVES = [
    (-1, 0), # gora
    (0, 1), # prawo
    (1, 0), # dol
    (0, -1), # lewo
]

def get_neighbors(pos):
    x, y = pos
    neighbors = []

    for dx, dy in MOVES:
        nx, ny = x + dx, y + dy
        if MAZE[nx, ny] == 0:
            neighbors.append((nx, ny))

    return neighbors

graph = {}
for i in range(MAZE.shape[0]):
    for j in range(MAZE.shape[1]):
        if MAZE[i, j] == 0:
            graph[(i, j)] = get_neighbors((i, j))


class MazeACO:
    def __init__(self, graph, START, GOAL, n_ants=40, n_iters=100,
                 alpha=1.0, beta=2.0, evaporation=0.3, Q=100):

        self.graph = graph
        self.START = START
        self.GOAL = GOAL
        self.n_ants = n_ants
        self.n_iters = n_iters
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q

        self.pheromone = defaultdict(lambda: 1.0)
        self.best_path = None
        self.best_length = float("inf")

    def heuristic(self, node):
        return 1 / (abs(node[0] - self.GOAL[0]) + abs(node[1] - self.GOAL[1]) + 1)

    def choose_next(self, current, visited):
        neighbors = [n for n in self.graph[current] if n not in visited]

        if not neighbors:
            return None

        probs = []
        for n in neighbors:
            edge = (current, n)
            tau = self.pheromone[edge] ** self.alpha # moc feromonu w krawedzie
            eta = self.heuristic(n) ** self.beta # dystans do celu
            probs.append(tau * eta)

        probs = np.array(probs)
        probs = probs / probs.sum()

        return neighbors[np.random.choice(len(neighbors), p=probs)] # wybiera w zalenosci od probs

    def construct_path(self):
        current = self.START
        visited = set([current])
        path = [current]

        steps_limit = len(self.graph) * 2

        for _ in range(steps_limit):
            if current == self.GOAL:
                return path

            nxt = self.choose_next(current, visited)

            if nxt is None:
                return None

            path.append(nxt)
            visited.add(nxt)
            current = nxt

        return None

    def update_pheromones(self, paths):
        for edge in list(self.pheromone.keys()):
            self.pheromone[edge] *= (1 - self.evaporation)

        for path in paths:
            if path is None:
                continue
            deposit = self.Q / len(path)

            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                self.pheromone[edge] += deposit

    def run(self):
        for _ in range(self.n_iters):
            paths = []

            for _ in range(self.n_ants):
                path = self.construct_path()
                paths.append(path)

                if path is not None and path[-1] == self.GOAL:
                    if len(path) < self.best_length:
                        self.best_length = len(path)
                        self.best_path = path

            self.update_pheromones(paths)

        return self.best_path, self.best_length



aco = MazeACO(graph, START, GOAL, n_ants=50, n_iters=150,
              alpha=1.0, beta=2.0, evaporation=0.25, Q=200)

best_path, best_length = aco.run()

print("Długość:", best_length)
print("Najlepsza ścieżka:")
print(best_path)
