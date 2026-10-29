import numpy as np

from src.convolution import convolution2d


image = np.array([
    [10, 20, 30],
    [20, 30, 40],
    [30, 40, 50]
], dtype=np.float32)


kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)


result = convolution2d(
    image,
    kernel
)


print("Ảnh gốc:")
print(image)

print("\nKernel:")
print(kernel)

print("\nKết quả:")
print(result)