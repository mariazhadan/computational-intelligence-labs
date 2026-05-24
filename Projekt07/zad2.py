import math
import pygad


# Funkcja oblicza wytrzymalosc stopu dla 6 skladnikow.
def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)


# Fitness jest rowny wytrzymalosci, bo chcemy ja maksymalizowac.
def fitness_func(ga_instance, solution, solution_idx):
    return endurance(*solution)


# Jeden chromosom ma 6 genow z przedzialu [0, 1).
def create_ga(seed):
    return pygad.GA(
        num_generations=80,
        num_parents_mating=30,
        fitness_func=fitness_func,
        sol_per_pop=40,
        num_genes=6,
        gene_space={"low": 0.0, "high": 1.0},
        parent_selection_type="rank",
        keep_elitism=15,
        crossover_type="uniform",
        mutation_type="random",
        mutation_percent_genes=15,
        random_seed=seed,
    )



best_solution = None
best_fitness = float("-inf")
best_ga_instance = None

for seed in range(1, 5):
    ga_instance = create_ga(seed)
    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()
    print(f"Uruchomienie {seed}: fitness={solution_fitness:.12f}, rozwiazanie={solution}")

    if solution_fitness > best_fitness:
        best_solution = solution
        best_fitness = solution_fitness
        best_ga_instance = ga_instance


print("Parametry najlepszego rozwiazania:", best_solution)
print("Wartosc fitness najlepszego rozwiazania =", best_fitness)
best_ga_instance.plot_fitness(title="Przebieg optymalizacji")
