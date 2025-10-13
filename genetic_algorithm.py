

import random
import math
import copy 
from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans
import numbers

default_problems = {
5: [(733, 251), (706, 87), (546, 97), (562, 49), (576, 253)],
10:[(470, 169), (602, 202), (754, 239), (476, 233), (468, 301), (522, 29), (597, 171), (487, 325), (746, 232), (558, 136)],
12:[(728, 67), (560, 160), (602, 312), (712, 148), (535, 340), (720, 354), (568, 300), (629, 260), (539, 46), (634, 343), (491, 135), (768, 161)],
15:[(512, 317), (741, 72), (552, 50), (772, 346), (637, 12), (589, 131), (732, 165), (605, 15), (730, 38), (576, 216), (589, 381), (711, 387), (563, 228), (494, 22), (787, 288)]
}
# ...existing code...

# Depósitos estrategicamente espaçados (farthest-point sampling)
# ...existing code...
import math
# ...existing code...

def _sqdist_xy(a, b):
    ax, ay = a; bx, by = b
    return (ax - bx) ** 2 + (ay - by) ** 2

def pick_spread_depots(cities_locations, k):
    """
    Escolhe k depósitos bem espaçados (farthest-point sampling).
    Retorna lista de índices.
    """
    n = len(cities_locations)
    if n == 0:
        return []
    k = max(1, min(k, n))
    cx = sum(x for x, _ in cities_locations) / n
    cy = sum(y for _, y in cities_locations) / n
    first = max(range(n), key=lambda i: _sqdist_xy(cities_locations[i], (cx, cy)))
    depots = [first]
    while len(depots) < k:
        candidates = [i for i in range(n) if i not in depots]
        def min_dist(i):
            return min(_sqdist_xy(cities_locations[i], cities_locations[d]) for d in depots)
        next_idx = max(candidates, key=min_dist)
        depots.append(next_idx)
    return depots

def pick_next_farthest_depot(cities_locations, current_depots, candidates=None):
    """
    Escolhe 1 novo depósito maximizando a distância mínima aos depósitos atuais.
    Retorna índice ou None se não houver candidato.
    """
    n = len(cities_locations)
    if n == 0:
        return None
    if candidates is None:
        candidates = [i for i in range(n) if i not in current_depots]
    if not candidates:
        return None
    def min_dist(i):
        return min(_sqdist_xy(cities_locations[i], cities_locations[d]) for d in current_depots)
    return max(candidates, key=min_dist)
# ...existing code...

def prioritize_priority_cities(individual, priority_city_indices):
    """
    Em cada rota (lista de índices), mantém o depósito na posição 0 e
    move as cidades priorizadas para o início da rota (após o depósito),
    preservando a ordem relativa dentro de cada grupo.
    """
    fixed = []
    for route in individual:
        if not route:
            fixed.append(route)
            continue
        depot = route[0]
        tail = route[1:]
        pri = [c for c in tail if isinstance(c, int) and c in priority_city_indices]
        non = [c for c in tail if isinstance(c, int) and c not in priority_city_indices]
        fixed.append([depot] + pri + non)
    return fixed
# ...existing code...

def _route_to_coords_for_fitness(route, cities_locations=None):
    """
    Converte uma rota (índices ou coordenadas) para uma lista de coordenadas (x, y).
    Se houver inteiros, exige cities_locations.
    """
    if not route:
        return []
    coords = []
    for p in route:
        if isinstance(p, int):
            if cities_locations is None:
                raise ValueError("cities_locations must be provided when route contains indices")
            coords.append(cities_locations[p])
        else:
            x, y = p  # tuple/list
            coords.append((float(x), float(y)))
    return coords

def generate_random_population(cities_location: List[Tuple[float, float]], population_size: int) -> List[List[Tuple[float, float]]]:
    """
    Generate a random population of routes for a given set of cities.

    Parameters:
    - cities_location (List[Tuple[float, float]]): A list of tuples representing the locations of cities,
      where each tuple contains the latitude and longitude.
    - population_size (int): The size of the population, i.e., the number of routes to generate.

    Returns:
    List[List[Tuple[float, float]]]: A list of routes, where each route is represented as a list of city locations.
    """
    return [random.sample(cities_location, len(cities_location)) for _ in range(population_size)]


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - point1 (Tuple[float, float]): The coordinates of the first point.
    - point2 (Tuple[float, float]): The coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def fix_individual(individual, all_cities, n_vehicles):
    """
    Garante que cada cidade aparece exatamente uma vez nas rotas do indivíduo.
    """
    # Junta todas as cidades do indivíduo
    flat = [city for route in individual for city in route]
    # Adiciona cidades faltantes
    missing = [city for city in all_cities if city not in flat]
    flat += missing
    # Remove cidades duplicadas mantendo só a primeira ocorrência
    seen = set()
    flat_unique = []
    for city in flat:
        if city not in seen:
            flat_unique.append(city)
            seen.add(city)
    # Divide igualmente entre os veículos
    split = [flat_unique[i::n_vehicles] for i in range(n_vehicles)]
    return split

def calculate_fitness_multi_vehicle(individual: List[List[Tuple[float, float]]]) -> float:
    """
    Calcula o fitness de uma solução multi-veículo (lista de rotas).
    O fitness é a soma das distâncias de todas as rotas.
    """
    return sum(calculate_fitness(route) for route in individual)

def calculate_fitness(route, cities_locations=None):
    """
    Aceita rota como lista de índices (0..n-1) ou de coordenadas [(x,y),...].
    Retorna a distância do ciclo fechando no início.
    """
    path = _route_to_coords_for_fitness(route, cities_locations)
    n = len(path)
    if n < 2:
        return 0.0
    distance = 0.0
    for i in range(n):
        a = path[i]
        b = path[(i + 1) % n]
        distance += calculate_distance(a, b)
    return distance

def calculate_fitness_multi_vehicle_balanced(individual, cities_locations):
    """
    Fitness de indivíduo multi-veículos = max distância entre as rotas (balanceamento).
    Suporta rotas em índices ou coordenadas.
    """
    if not individual:
        return 0.0
    values = []
    for route in individual:
        if not route:
            values.append(0.0)
        else:
            values.append(calculate_fitness(route, cities_locations))
    return max(values) if values else 0.0



def order_crossover(parent1: List[Tuple[float, float]], parent2: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Perform order crossover (OX) between two parent sequences to create a child sequence.

    Parameters:
    - parent1 (List[Tuple[float, float]]): The first parent sequence.
    - parent2 (List[Tuple[float, float]]): The second parent sequence.

    Returns:
    List[Tuple[float, float]]: The child sequence resulting from the order crossover.
    """
    length = len(parent1)

    # Choose two random indices for the crossover
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    # Initialize the child with a copy of the substring from parent1
    child = parent1[start_index:end_index]

    # Fill in the remaining positions with genes from parent2
    remaining_positions = [i for i in range(length) if i < start_index or i >= end_index]
    remaining_genes = [gene for gene in parent2 if gene not in child]

    for position, gene in zip(remaining_positions, remaining_genes):
        child.insert(position, gene)

    return child



def heuristic_multi_vehicle_solution(cities, n_vehicles):
    coords = np.array(cities)
    kmeans = KMeans(n_clusters=n_vehicles, n_init=10, random_state=42)
    labels = kmeans.fit_predict(coords)
    solution = []
    for v in range(n_vehicles):
        cluster_cities = [cities[i] for i in range(len(cities)) if labels[i] == v]
        if not cluster_cities:
            # Se o cluster ficou vazio, adicione uma cidade aleatória
            cluster_cities = [random.choice(cities)]
        route = nearest_neighbour_route(cluster_cities)
        solution.append(route)
    return solution

### demonstration: crossover test code
# Example usage:
# parent1 = [(1, 1), (2, 2), (3, 3), (4,4), (5,5), (6, 6)]
# parent2 = [(6, 6), (5, 5), (4, 4), (3, 3),  (2, 2), (1, 1)]

# # parent1 = [1, 2, 3, 4, 5, 6]
# # parent2 = [6, 5, 4, 3, 2, 1]


# child = order_crossover(parent1, parent2)
# print("Parent 1:", [0, 1, 2, 3, 4, 5, 6, 7, 8])
# print("Parent 1:", parent1)
# print("Parent 2:", parent2)
# print("Child   :", child)


# # Example usage:
# population = generate_random_population(5, 10)

# print(calculate_fitness(population[0]))


# population = [(random.randint(0, 100), random.randint(0, 100))
#           for _ in range(3)]
def generate_random_population_multi_vehicle(cities, population_size, n_vehicles):
    population = []
    for _ in range(population_size):
        shuffled = random.sample(cities, len(cities))
        # Divide as cidades igualmente entre os veículos
        split = [shuffled[i::n_vehicles] for i in range(n_vehicles)]
        population.append(split)
    return population

def nearest_neighbour_route(cities_locations):
    unvisited = cities_locations[:]
    route = [unvisited.pop(0)]
    while unvisited:
        last = route[-1]
        next_city = min(unvisited, key=lambda city: calculate_distance(last, city))
        route.append(next_city)
        unvisited.remove(next_city)
    return route

# TODO: implement a mutation_intensity and invert pieces of code instead of just swamping two. 
def mutate(solution:  List[Tuple[float, float]], mutation_probability: float) ->  List[Tuple[float, float]]:
    """
    Mutate a solution by inverting a segment of the sequence with a given mutation probability.

    Parameters:
    - solution (List[int]): The solution sequence to be mutated.
    - mutation_probability (float): The probability of mutation for each individual in the solution.

    Returns:
    List[int]: The mutated solution sequence.
    """
    mutated_solution = copy.deepcopy(solution)

    # Check if mutation should occur    
    if random.random() < mutation_probability:
        
        # Ensure there are at least two cities to perform a swap
        if len(solution) < 2:
            return solution
    
        # Select a random index (excluding the last index) for swapping
        index = random.randint(0, len(solution) - 2)
        
        # Swap the cities at the selected index and the next index
        mutated_solution[index], mutated_solution[index + 1] = solution[index + 1], solution[index]   
        
    return mutated_solution

### Demonstration: mutation test code    
# # Example usage:
# original_solution = [(1, 1), (2, 2), (3, 3), (4, 4)]
# mutation_probability = 1

# mutated_solution = mutate(original_solution, mutation_probability)
# print("Original Solution:", original_solution)
# print("Mutated Solution:", mutated_solution)


def sort_population(population: List[List[Tuple[float, float]]], fitness: List[float]) -> Tuple[List[List[Tuple[float, float]]], List[float]]:
    """
    Sort a population based on fitness values.

    Parameters:
    - population (List[List[Tuple[float, float]]]): The population of solutions, where each solution is represented as a list.
    - fitness (List[float]): The corresponding fitness values for each solution in the population.

    Returns:
    Tuple[List[List[Tuple[float, float]]], List[float]]: A tuple containing the sorted population and corresponding sorted fitness values.
    """
    # Combine lists into pairs
    combined_lists = list(zip(population, fitness))

    # Sort based on the values of the fitness list
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1])

    # Separate the sorted pairs back into individual lists
    sorted_population, sorted_fitness = zip(*sorted_combined_lists)

    return sorted_population, sorted_fitness


if __name__ == '__main__':
    N_CITIES = 10
    
    POPULATION_SIZE = 100
    N_GENERATIONS = 100
    MUTATION_PROBABILITY = 0.3
    cities_locations = [(random.randint(0, 100), random.randint(0, 100))
              for _ in range(N_CITIES)]
    
    # CREATE INITIAL POPULATION
    population = generate_random_population(cities_locations, POPULATION_SIZE)

    # Lists to store best fitness and generation for plotting
    best_fitness_values = []
    best_solutions = []
    
    for generation in range(N_GENERATIONS):
  
        
        population_fitness = [calculate_fitness(individual) for individual in population]    
        
        population, population_fitness = sort_population(population,  population_fitness)
        
        best_fitness = calculate_fitness(population[0])
        best_solution = population[0]
           
        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)    

        print(f"Generation {generation}: Best fitness = {best_fitness}")

        new_population = [population[0]]  # Keep the best individual: ELITISM
        
        while len(new_population) < POPULATION_SIZE:
            
            # SELECTION
            parent1, parent2 = random.choices(population[:10], k=2)  # Select parents from the top 10 individuals
            


            # CROSSOVER
            child1 = order_crossover(parent1, parent2)
            
            ## MUTATION
            child1 = mutate(child1, MUTATION_PROBABILITY)
            
            new_population.append(child1)
            
    
        print('generation: ', generation)
        population = new_population
    


# ...existing code...
from typing import List, Tuple
import random

def _coord_to_index_map(cities_locations):
    return {tuple(c): i for i, c in enumerate(cities_locations)}

def to_index_route(route, cities_locations):
    """
    Aceita rota como lista de índices ou lista de coordenadas e devolve lista de índices.
    """
    if not route:
        return []
    if isinstance(route[0], int):
        return list(route)
    # rota em coordenadas
    m = _coord_to_index_map(cities_locations)
    idxs = []
    for p in route:
        # Se p já é int, só adiciona
        if isinstance(p, int):
            idxs.append(p)
            continue
        t = tuple(p)
        i = m.get(t)
        if i is None:
            i = m.get((int(p[0]), int(p[1])))
        if i is None:
            i = _find_nearest_index((float(p[0]), float(p[1])), cities_locations)
        idxs.append(i)
    return idxs
def to_coord_route(route_idx, cities_locations):
    return [cities_locations[i] for i in route_idx]

def normalize_route_preserving_depot(route_idx, depot_idx):
    if not route_idx:
        return [depot_idx]
    # remove todas as ocorrências do depot e recoloca no início
    body = [n for n in route_idx if n != depot_idx]
    # se o depot existia, também rotaciona se necessário (já removemos acima)
    return [depot_idx] + body

def normalize_individual(individual, start_city_indices, cities_locations):
    """
    Garante que cada rota começa no depot correto e que o depot aparece só no início.
    individual: List[List[int|tuple]] (índices ou coordenadas)
    retorna: List[List[int]] (índices)
    """
    fixed = []
    for ridx, route in enumerate(individual):
        depot_idx = start_city_indices[min(ridx, len(start_city_indices)-1)]
        route_idx = to_index_route(route, cities_locations)
        if depot_idx in route_idx:
            i = route_idx.index(depot_idx)
            route_idx = route_idx[i:] + route_idx[:i]
        route_idx = normalize_route_preserving_depot(route_idx, depot_idx)
        fixed.append(route_idx)
    return fixed

def mutate_route_preserving_depot(route_idx, mutation_prob):
    # nunca altera posição 0 (depot)
    if len(route_idx) <= 2:
        return route_idx
    head = route_idx[0]
    tail = route_idx[1:]
    if random.random() < mutation_prob:
        i, j = random.sample(range(len(tail)), 2)
        tail[i], tail[j] = tail[j], tail[i]
    return [head] + tail

def mutate_individual_preserving_depots(individual, mutation_prob):
    return [mutate_route_preserving_depot(r, mutation_prob) for r in individual]
# ...existing code...

def normalize_individual_coords(individual, start_city_indices, cities_locations):
    """
    Aplica a normalização que preserva os depots e devolve SEMPRE listas de coordenadas.
    """
    fixed = normalize_individual(individual, start_city_indices, cities_locations)
    return [route_to_coords(r, cities_locations) for r in fixed]



def repair_unique_clients(individual, start_city_indices, n_cities):
    """
    Remove cidades duplicadas entre rotas e garante que cada cliente apareça em uma única rota.
    Mantém depósitos apenas na posição 0 de sua respectiva rota.
    individual: List[List[int]] (rotas em índices)
    """
    depots = set(start_city_indices)
    seen = set()
    fixed = []

    # 1) limpa cada rota: mantém depot no início e remove duplicatas/depósitos indevidos
    for ridx, route in enumerate(individual):
        depot = start_city_indices[ridx]
        cleaned = [depot]
        for c in route:
            if not isinstance(c, int):
                continue  # ignorar qualquer coord perdida (rotas devem estar em índices)
            if c in depots:
                continue  # depósitos só ficam na cabeça da própria rota
            if c not in seen:
                cleaned.append(c)
                seen.add(c)
        fixed.append(cleaned)

    # 2) insere clientes faltantes (se algum sumiu) na rota mais curta
    all_clients = set(range(n_cities)) - depots
    missing = list(all_clients - seen)
    for c in missing:
        target = min(range(len(fixed)), key=lambda i: len(fixed[i]))
        fixed[target].append(c)

    return fixed


def route_to_coords(route, cities_locations):
    """
    Converte rota para coordenadas [(x,y), ...], aceitando:
    - lista de índices [0,2,5,...]
    - lista de coordenadas [(x,y), ...]
    - lista mista (índices e coords)
    """
    if not route:
        return []
    coords = []
    for p in route:
        if isinstance(p, numbers.Integral):
            idx = int(p)
            coords.append(tuple(map(int, cities_locations[idx])))
        elif isinstance(p, (list, tuple)) and len(p) == 2:
            x, y = p
            coords.append((int(x), int(y)))
        else:
            raise TypeError(f"Invalid route point (expected index or (x,y)): {p!r}")
    return coords


def remove_extra_depots(individual, start_city_indices):
    """
    Para cada rota, remove todas as cidades fixas exceto a do início.
    individual: lista de rotas (cada rota é lista de índices)
    start_city_indices: lista de índices das cidades fixas (depósitos)
    """
    for idx, route in enumerate(individual):
        depot = start_city_indices[idx]
        # Mantém o depósito só no início, remove se aparecer em outras posições
        individual[idx] = [depot] + [c for i, c in enumerate(route) if c != depot or i == 0]
    return individual

# ...existing code...
def crossover_multi(parent1, parent2):
    """
    Aplica order_crossover rota-a-rota, preservando a estrutura multi-veículos.
    parent1/parent2: List[List[int]] (rotas por veículo, em índices)
    """
    size = min(len(parent1), len(parent2))
    child = []
    for i in range(size):
        child.append(order_crossover(parent1[i], parent2[i]))
    # se tiverem comprimentos diferentes, completa com as rotas do parent1
    for i in range(size, len(parent1)):
        child.append(list(parent1[i]))
    return child
# ...existing code...