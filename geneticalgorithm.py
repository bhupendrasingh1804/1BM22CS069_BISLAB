import numpy as np
import random

POPULATION_SIZE = 10  
GENES = 8             
GENERATIONS = 50      
MUTATION_RATE = 0.1   


def fitness_function(x):
    return x**2


def decode_chromosome(chromosome):
    return int("".join(map(str, chromosome)), 2)


def initialize_population():
    return [np.random.randint(0, 2, GENES).tolist() for _ in range(POPULATION_SIZE)]

def evaluate_fitness(population):
    return [fitness_function(decode_chromosome(individual)) for individual in population]


def select_parents(population, fitness):
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    selected = random.choices(population, weights=probabilities, k=2)
    return selected


def crossover(parent1, parent2):
    point = random.randint(1, GENES - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2


def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]  # Flip the bit
    return individual


def genetic_algorithm():
    # Step 1: Initialize population
    population = initialize_population()
    for generation in range(GENERATIONS):
        # Step 2: Evaluate fitness
        fitness = evaluate_fitness(population)
        
        # Logging the best solution of the generation
        best_individual = population[np.argmax(fitness)]
        best_fitness = max(fitness)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        # Step 3: Create the next generation
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            # Step 4: Select parents
            parent1, parent2 = select_parents(population, fitness)
            # Step 5: Crossover
            offspring1, offspring2 = crossover(parent1, parent2)
            # Step 6: Mutate
            new_population.append(mutate(offspring1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(offspring2))
        
       
        population = new_population
    

    fitness = evaluate_fitness(population)
    best_individual = population[np.argmax(fitness)]
    best_fitness = max(fitness)
    best_solution = decode_chromosome(best_individual)
    print("\nFinal Best Solution:")
    print(f"Chromosome: {best_individual}, Decoded: {best_solution}, Fitness: {best_fitness}")


if __name__ == "__main__":
    genetic_algorithm()
