import time
import pygad
import numpy as np


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
EXIT = (10, 10)

MOVES = {
    0: (-1, 0), # gora
    1: (0, 1), # prawo
    2: (1, 0), # dol
    3: (0, -1), # lewo
    
}

MAX_STEPS = 30

def fitness_func(model, solution, solution_idx):
    x, y = START
    penalty = 0
    best_distance = abs(x - EXIT[0]) + abs(y - EXIT[1])

    for step, move in enumerate(solution):
        dx, dy = MOVES[int(move)]
        nx, ny = x + dx, y + dy

        if MAZE[nx, ny] == 0:
            x, y = nx, ny
        else:
            penalty += 5

        distance = abs(x - EXIT[0]) + abs(y - EXIT[1])
        best_distance = min(best_distance, distance)

        if (x, y) == EXIT:
            return 1000 + (MAX_STEPS - step) * 10 - penalty

    return 1000 - best_distance * 60 - distance * 20 - penalty

def create_ga(seed=1):
    ga_instance = pygad.GA(
        num_generations=300,
        sol_per_pop=100,
        num_parents_mating=30,
        num_genes=MAX_STEPS,
        gene_space=[0, 1, 2, 3],
        gene_type=int,
        fitness_func=fitness_func,
        parent_selection_type="sss",
        keep_elitism=8,
        crossover_type="single_point",
        mutation_type="random",
        mutation_by_replacement=True,
        mutation_percent_genes=8,
        stop_criteria=["reach_950"],
    )

    start_t = time.time()
    ga_instance.run()
    end_t = time.time()

    solution, fitness, _ = ga_instance.best_solution()
    return ga_instance, solution, fitness, end_t - start_t

times = []
solutions = []
fitnesses = []

for _ in range(10):
    ga_instance, solution, fitness, t = create_ga()

    times.append(t)
    solutions.append(solution)
    fitnesses.append(fitness)

avg_time = sum(times) / len(times)

best_idx = int(np.argmax(fitnesses))

print("Sredni czas:", avg_time)
print("Best fitness:", fitnesses[best_idx])
print("Best solution:", solutions[best_idx])
