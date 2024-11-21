import numpy as np


def signal_strength(position, ap_positions, max_distance=100):
    """
    Calculate the signal strength for an AP at the given position, using an inverse distance model.
    - `position`: The (x, y) position of the AP.
    - `ap_positions`: A list of positions of other APs.
    - `max_distance`: The maximum coverage distance.
    Returns a total signal strength value considering all other APs.
    """
    total_signal = 0
    for ap in ap_positions:
        distance = np.linalg.norm(np.array(position) - np.array(ap))
        if distance < max_distance:
            total_signal += (max_distance - distance)  # Inverse of distance model for signal
    return total_signal

# Interference Function: A simple inverse distance model for interference
def interference(ap1, ap2):
    """
    Calculate the interference between two APs based on their distance.
    - `ap1`, `ap2`: (x, y) positions of the two APs.
    Returns interference value.
    """
    distance = np.linalg.norm(np.array(ap1) - np.array(ap2))
    if distance == 0:
        return float('inf')  # If APs are in the same location, infinite interference
    else:
        return 1 / distance  # Inverse of distance model for interference

# Fitness Function
def fitness_function(solution, network_area, max_distance=100):
    """
    Calculate the fitness of a solution, which is a list of AP positions.
    Fitness is calculated as:
    - Maximizing the signal strength (coverage).
    - Minimizing interference between APs.
    """
    total_signal = 0
    total_interference = 0

    # Calculate total signal strength from all APs
    for i, ap in enumerate(solution):
        total_signal += signal_strength(ap, solution, max_distance)

    # Calculate total interference between APs
    for i in range(len(solution)):
        for j in range(i+1, len(solution)):
            total_interference += interference(solution[i], solution[j])

    # The fitness is the total signal minus interference (to be maximized)
    return total_signal - total_interference

# Lévy Flight function (used to generate new candidate solutions)
def levy_flight(dim, alpha=1.5):
    """
    Generate a step using Lévy flight.
    - `dim`: Dimensionality of the search space (e.g., 2D for (x, y)).
    - `alpha`: Lévy flight exponent.
    """
    sigma = (np.math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
             np.math.gamma((1 + alpha) / 2) * alpha * np.pi) ** (1 / alpha)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    step = u / np.abs(v) ** (1 / alpha)
    return step

# Cuckoo Search Algorithm
def cuckoo_search(network_area, num_nests, max_iter, discovery_prob, max_distance=100):
    """
    Implement the Cuckoo Search algorithm for optimizing AP placement.
    - `network_area`: Bounds for AP positions (e.g., [[0, 0], [100, 100]]).
    - `num_nests`: Number of nests (APs to place).
    - `max_iter`: Maximum number of iterations.
    - `discovery_prob`: Probability of replacing the worst nests.
    """
    dim = len(network_area[0])  # 2D for (x, y)

    # Initialize nests with random positions within the network area
    nests = [np.random.uniform(network_area[0], network_area[1], dim) for _ in range(num_nests)]
    fitness = [fitness_function(nest, network_area, max_distance) for nest in nests]

    best_nest = nests[np.argmax(fitness)]  # Best solution so far
    best_fitness = max(fitness)  # Maximum fitness value

    for iteration in range(max_iter):
        # Generate new nests via Lévy flights
        new_nests = []
        for i in range(num_nests):
            step = levy_flight(dim)
            new_nest = nests[i] + step
            new_nest = np.clip(new_nest, network_area[0], network_area[1])  # Clip to boundaries

            # Evaluate the fitness of the new nest
            new_fitness = fitness_function(new_nest, network_area, max_distance)

            # If the new solution is better, replace the old nest
            if new_fitness > fitness[i]:
                new_nests.append(new_nest)
                fitness[i] = new_fitness
            else:
                new_nests.append(nests[i])

        # Abandon the worst nests with probability `discovery_prob`
        for i in range(int(discovery_prob * num_nests)):
            random_nest = np.random.randint(num_nests)
            new_nests[random_nest] = np.random.uniform(network_area[0], network_area[1], dim)
            fitness[random_nest] = fitness_function(new_nests[random_nest], network_area, max_distance)

        nests = new_nests

        # Update the best solution
        max_fitness_idx = np.argmax(fitness)
        if fitness[max_fitness_idx] > best_fitness:
            best_fitness = fitness[max_fitness_idx]
            best_nest = nests[max_fitness_idx]

        # Output the current progress
        print(f"Iteration {iteration+1}/{max_iter} - Best Fitness: {best_fitness}")

    return best_nest, best_fitness

# Example usage of Cuckoo Search for Wireless Network Optimization
network_area = [[0, 0], [100, 100]]  # The network area (100x100 grid)
num_nests = 10  # Number of APs (nests)
max_iter = 1000  # Maximum iterations
discovery_prob = 0.25  # Discovery probability (fraction of worst nests to abandon)

best_solution, best_fitness = cuckoo_search(network_area, num_nests, max_iter, discovery_prob)

print("Best placement of access points:", best_solution)
print("Best fitness value (coverage - interference):", best_fitness)
