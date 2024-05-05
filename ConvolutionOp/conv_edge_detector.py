import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a 25x25 black image
image = np.zeros((25, 25))

# Draw a white ring in the middle
for i in range(25):
    for j in range(25):
        dist = (i - 12) ** 2 + (j - 12) ** 2
        if 45 <= dist <= 55:  # defining the ring thickness
            image[i, j] = 1

# Create a target feature map with the desired arc segment in the middle-center
target = np.zeros((25, 25))
for i in range(25):
    for j in range(25):
        dist = (i - 12) ** 2 + (j - 12) ** 2
        if 45 <= dist <= 55 and 11 <= j <= 13:  # defining the arc segment
            target[i, j] = 1

# Initialize a random 7x7 filter
filter = np.random.rand(7, 7) - 0.5  # centering around 0
filter /= np.linalg.norm(filter)  # Normalize the filter

learning_rate = 0.01
epochs = 200

fig, ax = plt.subplots(1, 3, figsize=(10, 6))

ax[0].imshow(image, cmap="gray")
ax[0].set_title("Original Image")


def update(num, filter, image, ax):
    # Convolving the image with the filter
    feature_map = np.zeros((19, 19))
    for i in range(19):
        for j in range(19):
            feature_map[i, j] = np.sum(image[i:i + 7, j:j + 7] * filter)

    # Calculate the gradient based on the target arc segment
    error = feature_map - target[3:-3, 3:-3]
    gradient = np.zeros_like(filter)
    for i in range(19):
        for j in range(19):
            gradient += error[i, j] * image[i:i + 7, j:j + 7]

    gradient /= np.linalg.norm(gradient)  # Normalize the gradient to avoid explosion
    # Update the filter
    filter -= learning_rate * gradient

    if num % 20 == 0:
        print(f"Epoch {num}, Loss: {np.sum(error ** 2)}")

    ax[1].imshow(filter, cmap="gray")
    ax[1].set_title("Filter")
    ax[2].imshow(feature_map, cmap="gray")
    ax[2].set_title("Feature Map")


ani = animation.FuncAnimation(fig, update, epochs, fargs=[filter, image, ax],
                              interval=50)
plt.tight_layout()
plt.show()
