import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm import (
    mutate, order_crossover, generate_random_population, calculate_fitness, sort_population,
    default_problems, nearest_neighbour_route, generate_random_population_multi_vehicle,
    calculate_fitness_multi_vehicle, fix_individual
)
from draw_functions import draw_paths, draw_plot, draw_cities, generate_random_colors
import sys
import numpy as np
from benchmark_att48 import *

# --- CONFIGURAÇÕES ---
WIDTH, HEIGHT = 800, 400
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450

N_CITIES = 10
POPULATION_SIZE = 100
MUTATION_PROBABILITY = 0.5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# --- DEFINA O NÚMERO DE VEÍCULOS ---
N_VEHICLES = 1 # Altere para 1 para TSP clássico, ou mais para multi-veículo

# --- GERAÇÃO DAS CIDADES ---
cities_locations = [
    (random.randint(NODE_RADIUS + PLOT_X_OFFSET, WIDTH - NODE_RADIUS),
     random.randint(NODE_RADIUS, HEIGHT - NODE_RADIUS))
    for _ in range(N_CITIES)
]

# --- INICIALIZAÇÃO DA POPULAÇÃO E CORES ---
if N_VEHICLES == 1:
    VEHICLE_COLORS = [BLUE]
    # Heurística + aleatórios
    population = [nearest_neighbour_route(cities_locations)]
    population += generate_random_population(cities_locations, POPULATION_SIZE - 1)
else:
    VEHICLE_COLORS = generate_random_colors(N_VEHICLES)
    population = generate_random_population_multi_vehicle(cities_locations, POPULATION_SIZE, N_VEHICLES)

best_fitness_values = []
best_solutions = []

# --- INICIALIZAÇÃO DO PYGAME ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver using Pygame")
clock = pygame.time.Clock()
generation_counter = itertools.count(start=1)

paused = False
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_p:
                paused = not paused

    if paused:
        pygame.time.wait(100)
        continue

    generation = next(generation_counter)
    screen.fill(WHITE)

    # --- FITNESS E ORDENAÇÃO ---
    if N_VEHICLES == 1:
        population_fitness = [calculate_fitness(ind) for ind in population]
        population, population_fitness = sort_population(population, population_fitness)
        best_fitness = calculate_fitness(population[0])
        best_solution = population[0]
    else:
        population_fitness = [calculate_fitness_multi_vehicle(ind) for ind in population]
        population, population_fitness = sort_population(population, population_fitness)
        best_fitness = calculate_fitness_multi_vehicle(population[0])
        best_solution = population[0]

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    draw_plot(screen, list(range(len(best_fitness_values))),
              best_fitness_values, y_label="Fitness - Distance (pxls)")

    draw_cities(screen, cities_locations, RED, NODE_RADIUS)

    # --- DESENHO DAS ROTAS ---
    if N_VEHICLES == 1:
        if len(best_solution) >= 2:
            draw_paths(screen, best_solution, VEHICLE_COLORS[0], width=3)
        if len(population[1]) >= 2:
            draw_paths(screen, population[1], rgb_color=(128, 128, 128), width=1)
    else:
        for idx, route in enumerate(best_solution):
            if len(route) >= 2:
                color = VEHICLE_COLORS[idx % len(VEHICLE_COLORS)]
                draw_paths(screen, route, color, width=3)
        for route in population[1]:
            if len(route) >= 2:
                draw_paths(screen, route, rgb_color=(128, 128, 128), width=1)

    print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)}")

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
            child1 = fix_individual(child1, cities_locations, N_VEHICLES)
            new_population.append(child1)

    population = new_population

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()