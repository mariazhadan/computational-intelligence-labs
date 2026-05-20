import time

import numpy
import pygad


LIMIT_WAGI = 25
CEL = 1630

przedmioty = [
    ("zegar",100,7),
    ("obraz-pejzaz",300,7),
    ("obraz-portret",200,6),
    ("radio",40,2),
    ("laptop",500,5),
    ("lampka nocna",70,6),
    ("srebrne sztucce",100,1),
    ("porcelana",250,3),
    ("figura z brazu",300,10),
    ("skorzana torebka",280,3),
    ("odkurzacz",300,15),
]

wartosci = numpy.array([p[1] for p in przedmioty])
wagi = numpy.array([p[2] for p in przedmioty])


def fitness_func(ga_instance, solution, solution_idx):
    wartosc = numpy.sum(solution * wartosci)
    waga = numpy.sum(solution * wagi)

    if waga > LIMIT_WAGI:
        return 0

    return wartosc


def nowy_model():
    return pygad.GA(
        gene_space=[0, 1],
        gene_type=int,
        num_genes=len(przedmioty),
        fitness_func=fitness_func,
        sol_per_pop=50,
        num_generations=150,
        num_parents_mating=30,
        parent_selection_type="rank",
        keep_parents=6,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=15,
        stop_criteria=["reach_1630"],
    )


def podsumuj(solution):
    solution = solution.astype(int)
    wartosc = int(numpy.sum(solution * wartosci))
    waga = int(numpy.sum(solution * wagi))
    wybrane = [p for gen, p in zip(solution, przedmioty) if gen == 1]
    return solution, wybrane, wartosc, waga


ga_instance = nowy_model()
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
solution, wybrane, wartosc, waga = podsumuj(solution)

print("Najlepszy chromosom:", solution.tolist())
print("Wybrane przedmioty:")
for nazwa, wartosc_przedmiotu, waga_przedmiotu in wybrane:
    print(f"- {nazwa}: wartosc przedmiotu={wartosc_przedmiotu}, waga={waga_przedmiotu}")
print("Suma wartosci przedmiotow:", wartosc)
print("Suma wag:", waga)

sukcesy = 0
czasy_udanych_prob = []

for i in range(10):
    ga_instance = nowy_model()

    start = time.time()
    ga_instance.run()
    end = time.time()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    solution, wybrane, wartosc, waga = podsumuj(solution)

    if wartosc == CEL and waga <= LIMIT_WAGI:
        sukcesy += 1
        czasy_udanych_prob.append(end - start)

    print(f"Proba {i + 1}: wartosc={wartosc}, waga={waga}, czas={end - start:.6f}s")


skutecznosc = sukcesy / 10 * 100
print("Skutecznosc:", skutecznosc, "%")

if czasy_udanych_prob:
    sredni_czas = sum(czasy_udanych_prob) / len(czasy_udanych_prob)
    print("Sredni czas udanych prob:", sredni_czas, "s")

ga_instance.plot_fitness(title="Przebieg optymalizacji")
