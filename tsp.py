import pygame
import random
import itertools
import sys
import numpy as np
import pandas as pd
import datetime
import math

from draw_functions import draw_paths, draw_plot, draw_cities, generate_random_colors
from demo_mutation import mutate_exchange_between_vehicles
from genetic_algorithm import (
    mutate,
    order_crossover,
    pick_next_farthest_depot,
    pick_spread_depots,
    generate_random_population,
    calculate_fitness,
    sort_population,
    nearest_neighbour_route,
    generate_random_population_multi_vehicle,
    fix_individual,
    calculate_fitness_multi_vehicle_balanced,
    heuristic_multi_vehicle_solution,
    crossover_multi,
    repair_unique_clients,
    normalize_individual,
    mutate_individual_preserving_depots,
    route_to_coords,
    remove_extra_depots,
    prioritize_priority_cities,
)
from cidades import chat_sobre_rotas, gerar_relatorio, df as cidades_df

# --- CONFIGURAÇÕES iniciais ---
WIDTH, HEIGHT = 1200, 600
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450

N_CITIES = 16
POPULATION_SIZE = 100
MUTATION_PROBABILITY = 0.5

BLACK = (125, 125, 125)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

N_VEHICLES = 1
VEHICLE_AUTONOMY = 500
MAX_STABLE_GENERATIONS = 300
geracoes_desde_incremento = 0
historico_best_fitness = []

# prioridades de entrega
PRIORITY_COUNT = 3                 # quantidade de cidades priorizadas
PRIORITY_RING_COLOR = (255, 0, 0)  # vermelho
PRIORITY_RING_OFFSET = 6           # raio extra do anel em relação ao NODE_RADIUS
PRIORITY_RING_WIDTH = 3            # espessura do anel


# --- INICIALIZAÇÃO DA POPULAÇÃO ---
random.seed(142)  # seed para reprodutibilidade
# --- GERAÇÃO DAS CIDADES ---  
cities_locations = [
    (random.randint(NODE_RADIUS + PLOT_X_OFFSET, WIDTH - NODE_RADIUS),
     random.randint(NODE_RADIUS, HEIGHT - NODE_RADIUS))
    for _ in range(N_CITIES)
]
city_names = list(cidades_df['Cidade'])

start_city_indices = pick_spread_depots(cities_locations, N_VEHICLES)
 # prioridades não podem ser depósitos
available_for_priority = list(set(range(len(cities_locations))) - set(start_city_indices))
priority_city_indices = set(random.sample(
     available_for_priority,
     k=min(PRIORITY_COUNT, len(available_for_priority))
 ))
# --- INICIALIZAÇÃO DA POPULAÇÃO E CORES ---
if N_VEHICLES == 1:
    VEHICLE_COLORS = [BLUE]
    # Heurística + aleatórios
    population = [nearest_neighbour_route(cities_locations)]
    population += generate_random_population(cities_locations, POPULATION_SIZE - 1)
else:

    VEHICLE_COLORS = generate_random_colors(N_VEHICLES)
    population = [heuristic_multi_vehicle_solution(cities_locations, N_VEHICLES)]
    population += generate_random_population_multi_vehicle(cities_locations, POPULATION_SIZE - 1, N_VEHICLES)
    population = [normalize_individual(ind, start_city_indices, cities_locations) for ind in population]
    population = [remove_extra_depots(ind, start_city_indices) for ind in population]
    population = [repair_unique_clients(ind, start_city_indices, len(cities_locations)) for ind in population]
    population = [prioritize_priority_cities(ind, priority_city_indices) for ind in population]



best_fitness_values = []
best_solutions = []
# --- INICIALIZAÇÃO DO PYGAME ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver using Pygame")
clock = pygame.time.Clock()
generation_counter = itertools.count(start=1)
overlay_font = pygame.font.SysFont(None, 24)

paused = False
running = True

show_city_names = False

df = pd.DataFrame(columns=[
    "generation",              # geração atual
    "N_VEHICLES",              # número de veículos
    "best_fitness",            # fitness global
    "fitness_veiculos",        # fitness individual por veículo
    "cities_locations",        # coordenadas
    "city_names",              # nomes das cidades
    "best_solution",           # rotas
    "vehicle_autonomy",        # autonomia limite usada
    "load_capacity",           # capacidade de carga (quando for inserida)
    "delivery_priority",       # prioridade de entregas
    "stable_generations",      # gerações sem melhora
    "mutation_probability",    # taxa de mutação
    "population_size",         # tamanho da população
    "avg_fitness",             # média de fitness da geração
    "timestamp",               # hora da geração
    "execution_time",          # tempo total de execução (ms/s)
    "heuristic_type",          # heurística usada (aleatória, kmeans etc.)
    "selection_method",        # método de seleção
    "crossover_method",        # tipo de crossover
    "mutation_method",         # tipo de mutação
    "LLM_summary",            # texto gerado pelo LLM (explicação de entregas)
    "PRIORITY_COUNT",
    "priority_city_indices",   # NOVO
    "priority_city_names",     # NOVO
    "start_city_indices",      # NOVO
    "start_city_names"  

])


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_p:
                paused = not paused
            elif event.key == pygame.K_i:
                show_city_names = not show_city_names
            elif event.key == pygame.K_r:
                gerar_relatorio(df)
            elif event.key == pygame.K_c:
                chat_sobre_rotas(df)
    if paused:
        pygame.time.wait(100)
        continue

    generation = next(generation_counter)
    geracoes_desde_incremento += 1
    screen.fill((0, 0, 0))

    # --- FITNESS E ORDENAÇÃO ---
    if N_VEHICLES == 1:
        population_fitness = [calculate_fitness(ind) for ind in population]
        population, population_fitness = sort_population(population, population_fitness)
        best_fitness = calculate_fitness(population[0])
        best_solution = population[0]
        fitness_veiculos = [round(best_fitness, 2)]
    else:
        population_fitness = [calculate_fitness_multi_vehicle_balanced(ind, cities_locations) for ind in population]
        population, population_fitness = sort_population(population, population_fitness)
        best_fitness = calculate_fitness_multi_vehicle_balanced(population[0], cities_locations)
        best_solution = population[0]
        fitness_veiculos = [round(calculate_fitness(route, cities_locations), 2) for route in best_solution]

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    # --- DESENHO DOS GRÁFICOS ---

    draw_plot(screen, list(range(len(best_fitness_values))), best_fitness_values, y_label="Fitness - Distance (pxls)")

    draw_cities(screen, cities_locations, RED, NODE_RADIUS)
    historico_best_fitness.append(best_fitness)
    
    for idx in priority_city_indices:
        x, y = cities_locations[idx]
        pygame.draw.circle(
            screen,
            PRIORITY_RING_COLOR,
            (x, y),
            NODE_RADIUS + PRIORITY_RING_OFFSET,
            PRIORITY_RING_WIDTH
        )

    # Overlay: número de veículos (no início da área do mapa)
    vehicles_text = overlay_font.render(f"Veículos e Rotas: {N_VEHICLES}", True, BLACK)
    screen.blit(vehicles_text, (PLOT_X_OFFSET + 10, 10))

    if show_city_names:
        font = pygame.font.SysFont(None, 20)
        for idx, (x, y) in enumerate(cities_locations):
            text = font.render(city_names[idx], True, (125, 125, 125))
            screen.blit(text, (x + 10, y - 10))


    # --- DESENHO DAS ROTAS ---
    if N_VEHICLES == 1:
        if len(best_solution) >= 2:
            route_coords = route_to_coords(best_solution, cities_locations)
            draw_paths(screen, route_coords, VEHICLE_COLORS[0], width=3)
            start_city = route_coords[0]
            pygame.draw.circle(screen, (0, 255, 0), start_city, NODE_RADIUS + 4, 2)
        if len(population) > 1 and len(population[1]) >= 2:
            pop1_coords = route_to_coords(population[1], cities_locations)
            draw_paths(screen, pop1_coords, rgb_color=(128, 128, 128), width=1)
    else:
        for idx, route in enumerate(best_solution):
            if len(route) >= 2:
                route_coords = route_to_coords(route, cities_locations)
                color = VEHICLE_COLORS[idx % len(VEHICLE_COLORS)]
                draw_paths(screen, route_coords, color, width=3)
                pygame.draw.circle(screen, (0, 255, 0), route_coords[0], NODE_RADIUS + 4, 2)
        for route in population[1]:
            if len(route) >= 2:
                pop_coords = route_to_coords(route, cities_locations)
                draw_paths(screen, pop_coords, rgb_color=(128, 128, 128), width=1)


    
    if N_VEHICLES > 1:
        print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)} | Fitness individual de cada veículo: {fitness_veiculos}")
    else:
        print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)}")

    # --- CHECAGEM DE AUTONOMIA: só incrementa veículos após MAX_STABLE_GENERATIONS ---
    if min(historico_best_fitness[-100:]) > VEHICLE_AUTONOMY and geracoes_desde_incremento >= MAX_STABLE_GENERATIONS:
        print(f"Maior rota ainda acima da autonomia ({VEHICLE_AUTONOMY}) após {geracoes_desde_incremento} gerações. Incrementando veículos para {N_VEHICLES + 1} e reiniciando população.")
        N_VEHICLES += 1
        VEHICLE_COLORS = generate_random_colors(N_VEHICLES)

        # novo depósito: maximiza distância mínima aos depósitos existentes
        available = [i for i in range(len(cities_locations)) if i not in start_city_indices]
        new_depot = pick_next_farthest_depot(cities_locations, start_city_indices, available)
        if new_depot is not None:
            start_city_indices.append(new_depot)
        else:
            start_city_indices = pick_spread_depots(cities_locations, N_VEHICLES)


        # garantir prioridades ≠ depósitos e repor se necessário
        priority_city_indices -= set(start_city_indices)
        reposicao_pool = list(set(range(len(cities_locations))) - set(start_city_indices) - set(priority_city_indices))
        faltam = max(0, PRIORITY_COUNT - len(priority_city_indices))
        if faltam > 0 and reposicao_pool:
            priority_city_indices |= set(random.sample(reposicao_pool, k=min(faltam, len(reposicao_pool))))

        population = [heuristic_multi_vehicle_solution(cities_locations, N_VEHICLES)]
        population += generate_random_population_multi_vehicle(cities_locations, POPULATION_SIZE - 1, N_VEHICLES)
        population = [normalize_individual(ind, start_city_indices, cities_locations) for ind in population]
        population = [remove_extra_depots(ind, start_city_indices) for ind in population]
        population = [repair_unique_clients(ind, start_city_indices, len(cities_locations)) for ind in population]
        population = [prioritize_priority_cities(ind, priority_city_indices) for ind in population]

        best_fitness_values = []
        best_solutions = []
        generation_counter = itertools.count(start=1)
        geracoes_desde_incremento = 0  # zera o contador
        historico_best_fitness = []    # zera o histórico para o novo ciclo
        #paused = True 
        continue
    elif min(historico_best_fitness) <= VEHICLE_AUTONOMY:
        geracoes_desde_incremento = 0  # zera o contador se já atingiu o objetivo
        
    # --- NOVA POPULAÇÃO ---
    new_population = [population[0]]  # ELITISM

    while len(new_population) < POPULATION_SIZE:

        if N_VEHICLES == 1:
            probability = 1 / np.array(population_fitness)
            parent1, parent2 = random.choices(population, weights=probability, k=2)
            child1 = order_crossover(parent1, parent2)
            child1 = mutate(child1, MUTATION_PROBABILITY)
            new_population.append(child1)
        else:
            fitness_array = np.array(population_fitness, dtype=float)
            fitness_array[fitness_array == 0] = 1e-8
            probability = 1 / fitness_array
            if not np.all(np.isfinite(probability)):
                probability = np.nan_to_num(probability, nan=0.0, posinf=0.0, neginf=0.0)
            parent1, parent2 = random.choices(population, weights=probability, k=2)

            # ANTES: child1 = order_crossover(parent1, parent2)  # errado para multi
            child1 = crossover_multi(parent1, parent2)  # correto: rota-a-rota

            # mutação preservando depósito (não mexe posição 0 de cada rota)
            child1 = mutate_individual_preserving_depots(child1, MUTATION_PROBABILITY)

            # mutação entre veículos (se houver)
            if len(new_population) > POPULATION_SIZE * 0.80:
                child1 = mutate_exchange_between_vehicles(child1, mutation_prob=0.1)

            # reparar e normalizar em ÍNDICES
            child1 = fix_individual(child1, cities_locations, N_VEHICLES)
            child1 = normalize_individual(child1, start_city_indices, cities_locations)
            child1 = remove_extra_depots(child1, start_city_indices)
            child1 = repair_unique_clients(child1, start_city_indices, len(cities_locations))
            child1 = prioritize_priority_cities(child1, priority_city_indices)

            new_population.append(child1)

    population = new_population
    # ...dentro do while running, após calcular as variáveis da geração...



    df.loc[len(df)] = {
        "generation": generation,
        "N_VEHICLES": N_VEHICLES,
        "best_fitness": best_fitness,
        "fitness_veiculos": fitness_veiculos.copy(),
        "cities_locations": cities_locations.copy(),
        "city_names": city_names.copy(),
        "best_solution": best_solution.copy(),
        "vehicle_autonomy": VEHICLE_AUTONOMY,
        "load_capacity": None,  # preencha se usar
        "delivery_priority": None,  # preencha se usar
    "stable_generations": geracoes_desde_incremento,
        "mutation_probability": MUTATION_PROBABILITY,
        "population_size": POPULATION_SIZE,
        "avg_fitness": float(np.mean(population_fitness)),
        "timestamp": datetime.datetime.now().isoformat(),
        "execution_time": None,  # preencha se medir
        "heuristic_type": "heurística usada",  # preencha se usar
        "selection_method": "seleção",         # preencha se usar
        "crossover_method": "crossover",       # preencha se usar
        "mutation_method": "mutação",          # preencha se usar
        "LLM_summary": None,  # preencha quando gerar resumo pela LLM
        "PRIORITY_COUNT": PRIORITY_COUNT,                                    # NOVO
        "priority_city_indices": list(priority_city_indices),                    # NOVO
        "priority_city_names": [city_names[i] for i in priority_city_indices],   # NOVO
        "start_city_indices": list(start_city_indices),                           # NOVO
        "start_city_names": [city_names[i] for i in start_city_indices]          # NOVO
    }

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()