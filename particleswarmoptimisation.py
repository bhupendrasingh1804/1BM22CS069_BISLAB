import numpy as np


def objective_function(position):
    x, y = position
    return x**2 + y**2


NUM_PARTICLES = 30  # Number of particles
DIMENSIONS = 2      # Number of dimensions (e.g., x and y)
ITERATIONS = 100    # Number of iterations
W = 0.5             # Inertia weight
C1 = 2.0            # Cognitive (personal) acceleration coefficient
C2 = 2.0            # Social (global) acceleration coefficient
BOUNDS = (-10, 10)  # Bounds for the search space


particles = np.random.uniform(BOUNDS[0], BOUNDS[1], (NUM_PARTICLES, DIMENSIONS))
velocities = np.random.uniform(-1, 1, (NUM_PARTICLES, DIMENSIONS))


personal_best_positions = particles.copy()
personal_best_scores = np.array([objective_function(p) for p in particles])
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
global_best_score = min(personal_best_scores)


for iteration in range(ITERATIONS):
    for i in range(NUM_PARTICLES):
        # Update velocity
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive = C1 * r1 * (personal_best_positions[i] - particles[i])
        social = C2 * r2 * (global_best_position - particles[i])
        velocities[i] = W * velocities[i] + cognitive + social
        
     
        particles[i] += velocities[i]
        
        particles[i] = np.clip(particles[i], BOUNDS[0], BOUNDS[1])
        
      
        fitness = objective_function(particles[i])
        
     
        if fitness < personal_best_scores[i]:
            personal_best_positions[i] = particles[i]
            personal_best_scores[i] = fitness
    

    current_best_index = np.argmin(personal_best_scores)
    if personal_best_scores[current_best_index] < global_best_score:
        global_best_position = personal_best_positions[current_best_index]
        global_best_score = personal_best_scores[current_best_index]
    
    print(f"Iteration {iteration + 1}: Global Best Score = {global_best_score}")


print("\nFinal Best Solution:")
print(f"Position: {global_best_position}, Score: {global_best_score}")
