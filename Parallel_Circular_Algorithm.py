# Parallel Circular Algorithm
import numpy as np
import cv2
import matplotlib.pyplot as plt


def sobel_operator(image):

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


    grad_x = cv2.filter2D(image, -1, sobel_x)
    grad_y = cv2.filter2D(image, -1, sobel_y)


    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_magnitude = np.uint8(np.clip(grad_magnitude, 0, 255))
    return grad_magnitude

def update_cell_state(image, i, j, W, H, sobel_magnitude):

    neighbors = []
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni, nj = (i + di) % W, (j + dj) % H
            neighbors.append(sobel_magnitude[ni, nj])


    return np.mean(neighbors)

def parallel_cellular_edge_detection(image, max_iterations=10):

    W, H = image.shape
    sobel_magnitude = sobel_operator(image)

    result_image = np.copy(image)


    for iteration in range(max_iterations):
        new_image = np.copy(result_image)


        for i in range(W):
            for j in range(H):
                new_image[i, j] = update_cell_state(image, i, j, W, H, sobel_magnitude)


        result_image = new_image

    return result_image


image = cv2.imread('/content/amazing-stone-path-in-forest-free-image.webp', cv2.IMREAD_GRAYSCALE)


if image is None:
    raise ValueError("Image not found. Make sure the file path is correct.")


output_image = parallel_cellular_edge_detection(image)


plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(output_image, cmap='gray')
plt.title("Processed Image (Edge Detection)")

plt.show()
