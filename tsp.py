import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm import (
    mutate, order_crossover, generate_random_population, calculate_fitness, sort_population,
    default_problems, nearest_neighbour_route, generate_random_population_multi_vehicle,
    calculate_fitness_multi_vehicle, fix_individual, calculate_fitness_multi_vehicle_balanced,heuristic_multi_vehicle_solution
)
from draw_functions import draw_paths, draw_plot, draw_cities, generate_random_colors
import sys
from demo_mutation import mutate_exchange_between_vehicles
import numpy as np
from benchmark_att48 import *
from sklearn.cluster import KMeans
from cidades import df as cidades_df
from cidades import gerar_relatorio

# --- CONFIGURAÇÕES ---
WIDTH, HEIGHT = 800, 400
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450

N_CITIES = 3
POPULATION_SIZE = 100
MUTATION_PROBABILITY = 0.5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

N_VEHICLES = 1
VEHICLE_AUTONOMY = 500
MAX_STABLE_GENERATIONS = 90
geracoes_desde_incremento = 0
historico_best_fitness = []

random.seed(222)  # Escolha qualquer número inteiro para a seed
# --- GERAÇÃO DAS CIDADES ---
cities_locations = [
    (random.randint(NODE_RADIUS + PLOT_X_OFFSET, WIDTH - NODE_RADIUS),
     random.randint(NODE_RADIUS, HEIGHT - NODE_RADIUS))
    for _ in range(N_CITIES)
]
city_names = list(cidades_df['Cidade'])


# --- INICIALIZAÇÃO DA POPULAÇÃO E CORES ---
if N_VEHICLES == 1:
    VEHICLE_COLORS = [BLUE]
    # Heurística + aleatórios
    population = [nearest_neighbour_route(cities_locations)]
    population += generate_random_population(cities_locations, POPULATION_SIZE - 1)
else:

    VEHICLE_COLORS = generate_random_colors(N_VEHICLES)
    # Heurística + aleatórios
    population = [heuristic_multi_vehicle_solution(cities_locations, N_VEHICLES)]
    population += generate_random_population_multi_vehicle(cities_locations, POPULATION_SIZE - 1, N_VEHICLES)

best_fitness_values = []
best_solutions = []
random.seed()
# --- INICIALIZAÇÃO DO PYGAME ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver using Pygame")
clock = pygame.time.Clock()
generation_counter = itertools.count(start=1)

paused = False
running = True

show_city_names = False
stable_generations = 0
last_fitness_veiculos = None


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
            elif event.key == pygame.K_c:
                gerar_relatorio(generation, N_VEHICLES, best_fitness, fitness_veiculos, cities_locations, city_names, best_solution)

    if paused:
        pygame.time.wait(100)
        continue

    generation = next(generation_counter)
    geracoes_desde_incremento += 1
    screen.fill(WHITE)

    # --- FITNESS E ORDENAÇÃO ---
    if N_VEHICLES == 1:
        population_fitness = [calculate_fitness(ind) for ind in population]
        population, population_fitness = sort_population(population, population_fitness)
        best_fitness = calculate_fitness(population[0])
        best_solution = population[0]
        fitness_veiculos = [round(best_fitness, 2)]
    else:
        population_fitness = [calculate_fitness_multi_vehicle_balanced(ind) for ind in population]
        population, population_fitness = sort_population(population, population_fitness)
        best_fitness = calculate_fitness_multi_vehicle_balanced(population[0])
        best_solution = population[0]
        fitness_veiculos = [round(calculate_fitness(route), 2) for route in best_solution]

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    # --- DESENHO DOS GRÁFICOS ---

    draw_plot(screen, list(range(len(best_fitness_values))),
              best_fitness_values, y_label="Fitness - Distance (pxls)")

    draw_cities(screen, cities_locations, RED, NODE_RADIUS)
    historico_best_fitness.append(best_fitness)

    if show_city_names:
        font = pygame.font.SysFont(None, 20)
        for idx, (x, y) in enumerate(cities_locations):
            text = font.render(city_names[idx], True, (0, 0, 0))
            screen.blit(text, (x + 10, y - 10))

    # --- DESENHO DAS ROTAS ---
    if N_VEHICLES == 1:
        if len(best_solution) >= 2:
            draw_paths(screen, best_solution, VEHICLE_COLORS[0], width=3)
            start_city = best_solution[0]
            pygame.draw.circle(screen, (0, 255, 0), start_city, NODE_RADIUS + 4, 2)  # círculo verde maior
    if len(population[1]) >= 2:
        if len(population[1]) >= 2:
            draw_paths(screen, population[1], rgb_color=(128, 128, 128), width=1)
    else:
        for idx, route in enumerate(best_solution):
            if len(route) >= 2:
                color = VEHICLE_COLORS[idx % len(VEHICLE_COLORS)]
                draw_paths(screen, route, color, width=3)
                start_city = route[0]
                pygame.draw.circle(screen, (0, 255, 0), start_city, NODE_RADIUS + 4, 2)  # círculo verde maior
        for route in population[1]:
            if len(route) >= 2:
                draw_paths(screen, route, rgb_color=(128, 128, 128), width=1)


    
    if N_VEHICLES > 1:
        print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)} | Fitness individual de cada veículo: {fitness_veiculos}")
    else:
        print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)}")

    # --- CHECAGEM DE AUTONOMIA: só incrementa veículos após MAX_STABLE_GENERATIONS ---
    if min(historico_best_fitness[-100:]) > VEHICLE_AUTONOMY and geracoes_desde_incremento >= MAX_STABLE_GENERATIONS:
        print(f"Maior rota ainda acima da autonomia ({VEHICLE_AUTONOMY}) após {geracoes_desde_incremento} gerações. Incrementando veículos para {N_VEHICLES + 1} e reiniciando população.")
        N_VEHICLES += 1
        VEHICLE_COLORS = generate_random_colors(N_VEHICLES)
        population = [heuristic_multi_vehicle_solution(cities_locations, N_VEHICLES)]
        population += generate_random_population_multi_vehicle(cities_locations, POPULATION_SIZE - 1, N_VEHICLES)
        best_fitness_values = []
        best_solutions = []
        best_overall_fitness = None
        
        generation_counter = itertools.count(start=1)
        geracoes_desde_incremento = 0  # zera o contador
        historico_best_fitness = []    # zera o histórico para o novo ciclo
        paused = True 
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
            child1 = order_crossover(parent1, parent2)
            child1 = mutate(child1, MUTATION_PROBABILITY)
            # Adicione a mutação entre veículos apenas para os últimos 10%:
            if len(new_population) > POPULATION_SIZE * 0.99:
                child1 = mutate_exchange_between_vehicles(child1, mutation_prob=0.01)
            child1 = fix_individual(child1, cities_locations, N_VEHICLES)
            new_population.append(child1)

    population = new_population

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()