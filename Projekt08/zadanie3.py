import random
from collections import deque


MAZE = (
    "S.....#",
    ".###..#",
    "...#..#",
    "##.#.##",
    "...#...",
    ".#.###.",
    ".#....E",
)


def maze_graph():
    free = set()
    start = None
    end = None

    for y, row in enumerate(MAZE):
        for x, value in enumerate(row):
            if value != "#":
                free.add((x, y))
            if value == "S":
                start = (x, y)
            if value == "E":
                end = (x, y)

    return free, start, end


def neighbors(cell, free):
    x, y = cell
    possible = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
    return [cell for cell in possible if cell in free]


def edge(a, b):
    return tuple(sorted((a, b)))


def bfs_shortest_path(free, start, end):
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        cell, path = queue.popleft()
        if cell == end:
            return path
        for next_cell in neighbors(cell, free):
            if next_cell not in visited:
                visited.add(next_cell)
                queue.append((next_cell, path + [next_cell]))


def choose_next(current, options, pheromone, end):
    weights = []
    for option in options:
        heuristic = 1 / (abs(option[0] - end[0]) + abs(option[1] - end[1]) + 1)
        weights.append((pheromone[edge(current, option)] ** 1.0) * (heuristic**2.0))

    value = random.random() * sum(weights)
    total = 0
    for option, weight in zip(options, weights):
        total += weight
        if total >= value:
            return option


def aco_maze(free, start, end):
    pheromone = {edge(cell, next_cell): 1.0 for cell in free for next_cell in neighbors(cell, free)}
    best_path = None

    random.seed(3)

    for _ in range(100):
        for _ in range(50):
            path = [start]
            visited = {start}
            current = start

            for _ in range(100):
                if current == end:
                    break

                options = [cell for cell in neighbors(current, free) if cell not in visited]
                if not options:
                    break

                current = choose_next(current, options, pheromone, end)
                path.append(current)
                visited.add(current)

            if path[-1] == end:
                if best_path is None or len(path) < len(best_path):
                    best_path = path

                deposit = 10 / (len(path) - 1)
                for a, b in zip(path, path[1:]):
                    pheromone[edge(a, b)] += deposit

        for pheromone_edge in pheromone:
            pheromone[pheromone_edge] *= 0.9

    return best_path


free, start, end = maze_graph()
aco_path = aco_maze(free, start, end)
shortest_path = bfs_shortest_path(free, start, end)

print("ACO dlugosc:", len(aco_path) - 1)
print("ACO sciezka:", aco_path)
print("BFS najkrotsza dlugosc:", len(shortest_path) - 1)

# Wybrane podejscie: ACO. Labirynt jest grafem, gdzie komorki to wierzcholki,
# a przejscia gora/dol/prawo/lewo to krawedzie. W tym przypadku ACO znalazl
# sciezke o takiej samej dlugosci jak BFS, wiec znalazl najkrotsza sciezke.
