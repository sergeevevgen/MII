from collections import namedtuple
from functools import partial
from typing import List, Callable, Tuple
from random import choices, randint, randrange, random

Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
Locality = namedtuple('Locality', ['name', 'supplies', 'distance'])
y = 3000

# # 2 производства, 3 города
# producers = [[1, 2, 3],
#              [12, 14, 15]]
#
# norms = [5, 6, 7]
# road_prices = [80, 100]
# limit = 100

localities = [
    Locality('Ульяновск', 617, 10),
    Locality('Инза', 113, 91),
    Locality('Новоульяновск', 165, 140),
    Locality('Чердаклы', 684, 197),
    Locality('Ишеевка', 1257, 206),
    Locality('Старая Майна', 314, 232),
    Locality('Мирный', 508, 240),
    Locality('Октябрьский', 1173, 248),
    Locality('Кузоватово', 522, 318),
    Locality('Белый ключ', 1253, 475)
]


# Функция создания генома заданном длины
def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)


# Функция создания популяции заданного размера с геномами заданной длины
def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


# Фитнесс-функция - определяет приспособленность генов
def fitness(genome: Genome, localities1: [Locality], value_limit: int, distance_limit: int) -> int:
    if len(genome) != len(localities1):
        raise ValueError("genome и localities должны быть одинаковой длины!")
    value = 0
    distance = 0
    for i, local in enumerate(localities1):
        if genome[i] == 1:
            value += local.supplies
            distance += local.distance
            if distance > distance_limit or value * 1.1 > value_limit:
                return 0
    return value


# Выбор пары для кроссовера (рождения потомка)
def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(population=population, weights=[fitness_func(genome) for genome in population], k=2)


# Кроссовер (рождение потомка)
def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Геномы должны быть одинаковой длины!")
    length = len(a)
    if length < 2:
        return a, b
    p = randint(2, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


# Мутация
def mutation(genome: Genome, num: int = 1, probability: float = 0.1) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if randint(0, 10) != (probability * 10) else abs(genome[index] - 1)
    return genome


# Запуск эволиционирования (запуск алгоритма)
def run_evolution(populate_func: PopulateFunc, fitness_func: FitnessFunc,
                  fitness_limit: int, selection_func: SelectionFunc = selection_pair,
                  crossover_func: CrossoverFunc = single_point_crossover, mutation_func: MutationFunc = mutation,
                  generation_limit: int = 100) -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)
        if fitness_func(population[0]) >= fitness_limit:
            break
        next_generation = population[0:2]
        for j in range(int(len(population) / 2) - 1):
            # выбор родителей
            parents = selection_func(population, fitness_func)
            # создание двух потомков
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            # возможная мутация
            offspring_a = mutation_func(offspring_a)
            # возможная мутация
            offspring_b = mutation_func(offspring_b)
            # создание новой генерации
            next_generation += [offspring_a, offspring_b]
        population = next_generation

    population = sorted(
        population,
        key=lambda genome: fitness_func(genome),
        reverse=True
    )
    return population, i


# Выводит города и значения приспособленности
def genome_to_things(genome: Genome, localities1: [Locality]) -> [Locality]:
    result = []
    value = 0
    for i, local in enumerate(localities1):
        if genome[i] == 1:
            result += [local.name]
            value += local.supplies
    return result, value


# Вывод
def final_route(genome: Genome, localities1: [Locality]) -> int:
    route, value = genome_to_things(population[0], localities1)
    print(f"1 пункт: {route} = {value}")
    route = []
    value = 0
    id = 2
    for i, local in enumerate(localities1):
        if genome[i] == 0:
            if value + local.supplies >= y:
                print(f"{id} пункт: {route} = {value}")
                id += 1
                value = 0
                route = []
            route += [local.name]
            value += local.supplies
    print(f"{id} пункт: {route} = {value}")
    return 1


# Запуск эволиции (генетического алгоритма)
population, generations = run_evolution(
    populate_func=partial(
        generate_population, size=8, genome_length=len(localities)
    ),
    fitness_func=partial(
        fitness, localities=localities, value_limit=y, distance_limit=1000
    ),
    fitness_limit=4000,
    generation_limit=101
)

print(f"Количество поколений: {generations}")
# print(f"best solution: {genome_to_things(population[0], cities)}")
final_route(population[0], localities)
print()
print("========== ПОПУЛЯЦИЯ ==========")
for i in range(len(population)):
    print(f"[{i + 1}]: {genome_to_things(population[i], localities)}")
