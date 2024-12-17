# Grey Wolf Optimiser
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def fitness_function(threshold, image):
    # Ensure threshold is in a valid range
    threshold = np.clip(threshold, 0, 255)
    _, segmented_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    foreground = image[segmented_image == 255]
    background = image[segmented_image == 0]
    ssd = np.sum((foreground - np.mean(foreground)) ** 2) + np.sum((background - np.mean(background)) ** 2)
    return ssd

class GreyWolfOptimizer:
    def __init__(self, num_wolves, max_iter, image):
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.image = image
        self.dim = 1
        self.positions = np.random.uniform(0, 255, size=(num_wolves, self.dim))
        self.fitness = np.zeros(num_wolves)
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float('inf')
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float('inf')
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float('inf')

        for i in range(self.num_wolves):
            self.fitness[i] = fitness_function(self.positions[i], self.image)
            if self.fitness[i] < self.alpha_score:
                self.alpha_score = self.fitness[i]
                self.alpha_pos = self.positions[i]
            elif self.fitness[i] < self.beta_score:
                self.beta_score = self.fitness[i]
                self.beta_pos = self.positions[i]
            elif self.fitness[i] < self.delta_score:
                self.delta_score = self.fitness[i]
                self.delta_pos = self.positions[i]

    def update_position(self, wolf_pos, alpha_pos, beta_pos, delta_pos, a):
        A1 = 2 * a * np.random.rand() - a
        C1 = 2 * np.random.rand()
        D1 = np.abs(C1 * alpha_pos - wolf_pos)
        X1 = alpha_pos - A1 * D1

        A2 = 2 * a * np.random.rand() - a
        C2 = 2 * np.random.rand()
        D2 = np.abs(C2 * beta_pos - wolf_pos)
        X2 = beta_pos - A2 * D2

        A3 = 2 * a * np.random.rand() - a
        C3 = 2 * np.random.rand()
        D3 = np.abs(C3 * delta_pos - wolf_pos)
        X3 = delta_pos - A3 * D3

        new_pos = (X1 + X2 + X3) / 3
        return np.clip(new_pos, 0, 255)

    def optimize(self):
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)
            for i in range(self.num_wolves):
                self.positions[i] = self.update_position(self.positions[i], self.alpha_pos, self.beta_pos, self.delta_pos, a)
                fitness = fitness_function(self.positions[i], self.image)
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i]
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i]
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i]

        # Return threshold as an integer
        return int(self.alpha_pos[0])  # Force as int

# Load the image
image = cv2.imread('/content/amazing-stone-path-in-forest-free-image.webp', cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not found. Make sure the file path is correct.")

# Optimize using Grey Wolf Optimizer
gwo = GreyWolfOptimizer(num_wolves=30, max_iter=100, image=image)
optimal_threshold = gwo.optimize()

# Ensure threshold is within range (0 to 255)
optimal_threshold = np.clip(optimal_threshold, 0, 255)

# Apply thresholding using OpenCV
_, segmented_image = cv2.threshold(image, optimal_threshold, 255, cv2.THRESH_BINARY)

# Plot the original and segmented images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='gray')
plt.title(f"Segmented Image (Threshold = {optimal_threshold})")

plt.show()
