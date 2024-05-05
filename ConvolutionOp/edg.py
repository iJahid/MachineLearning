import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

image = np.zeros((28, 28))
image[:, 7:9] = 10
image[:, 19:21] = 10
image[10:12, :] = 10
image[18:20, :] = 10

# plt.imshow(image, cmap='gray')
# plt.title('Orginal')
# plt.show()

sobol_filter = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

feature_map = convolve2d(image, sobol_filter, mode='same')
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Orginal')
plt.subplot(1, 2, 2)
plt.imshow(feature_map, cmap='gray')
plt.title('feature_map')


plt.show()
