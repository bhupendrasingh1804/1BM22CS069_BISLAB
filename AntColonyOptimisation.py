import numpy as np

# ACO Parameters
NUM_ANTS = 10          # Number of ants
NUM_ITERATIONS = 100   # Number of iterations
ALPHA = 1              # Pheromone importance
BETA = 2               # Heuristic importance
EVAPORATION_RATE = 0.5 # Pheromone evaporation rate
Q = 100                # Pheromone deposit factor

# Distance matrix for TSP
distance_matrix = np.array([
    [0, 2, 2, 5, 7],
    [2, 0, 4, 8, 2],
    [2, 4, 0, 1, 3],
    [5, 8, 1, 0, 2],
    [7, 2, 3, 2, 0],
])
NUM_CITIES = distance_matrix.shape[0]

#  pheromone matrix
pheromone_matrix = np.ones((NUM_CITIES, NUM_CITIES))

# Heuristic information (1/distance, avoid divide by zero)
heuristic_matrix = 1 / (distance_matrix + np.eye(NUM_CITIES))


def ant_colony_optimization():
    global pheromone_matrix
    best_path = None
    best_path_cost = float('inf')

    for iteration in range(NUM_ITERATIONS):
        all_paths = []
        all_costs = []

     
        for ant in range(NUM_ANTS):
            path = [np.random.randint(0, NUM_CITIES)]
            while len(path) < NUM_CITIES:
                current_city = path[-1]
                probabilities = calculate_transition_probabilities(path, current_city)
                next_city = np.random.choice(range(NUM_CITIES), p=probabilities)
                path.append(next_city)
            path.append(path[0])  # Return to the starting city
            cost = calculate_path_cost(path)
            all_paths.append(path)
            all_costs.append(cost)

      
        update_pheromones(all_paths, all_costs)

        
        min_cost_index = np.argmin(all_costs)
        if all_costs[min_cost_index] < best_path_cost:
            best_path = all_paths[min_cost_index]
            best_path_cost = all_costs[min_cost_index]

        
        print(f"Iteration {iteration + 1}: Best Cost = {best_path_cost}")

    print("\nFinal Best Solution:")
    print(f"Path: {best_path}, Cost: {best_path_cost}")


def calculate_transition_probabilities(visited, current_city):
    probabilities = []
    for city in range(NUM_CITIES):
        if city not in visited:
            pheromone = pheromone_matrix[current_city][city] ** ALPHA
            heuristic = heuristic_matrix[current_city][city] ** BETA
            probabilities.append(pheromone * heuristic)
        else:
            probabilities.append(0)
    probabilities = np.array(probabilities)
    return probabilities / probabilities.sum()


def calculate_path_cost(path):
    return sum(distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))


def update_pheromones(paths, costs):
    global pheromone_matrix
    pheromone_matrix *= (1 - EVAPORATION_RATE)
    for path, cost in zip(paths, costs):
        for i in range(len(path) - 1):
            pheromone_matrix[path[i]][path[i + 1]] += Q / cost

if __name__ == "__main__":
    ant_colony_optimization()
