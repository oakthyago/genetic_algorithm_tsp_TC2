# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:41:21 2023

@author: SérgioPolimante
"""

import random
import copy 
#### MUTATION ###
def mutate(solution, mutation_probability):

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
        
        
    
def mutate_exchange_between_vehicles(individual, mutation_prob=0.1):
    """Move uma cidade aleatória de um veículo para outro."""
    if random.random() < mutation_prob and len(individual) > 1:
        # Escolhe dois veículos diferentes
        v1, v2 = random.sample(range(len(individual)), 2)
        if individual[v1]:  # Só move se o veículo tiver cidades
            idx = random.randrange(len(individual[v1]))
            city = individual[v1].pop(idx)
            insert_pos = random.randrange(len(individual[v2]) + 1)
            individual[v2].insert(insert_pos, city)
    return individual

# Example usage:
original_solution =[(99, 100), (2, 50), (1, 71)]
mutation_probability = 1

mutated_solution = mutate(original_solution, mutation_probability)
print("Original Solution:", original_solution)
print("Mutated Solution:", mutated_solution)