from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np

cmap = colormaps["plasma"]


def generate(bits):
    A = np.array([[0]])
    n = 1
    for _ in range(bits):
        N = n*n
        B = np.empty((2*n, 2*n), dtype=int)
        B[:n, :n] = A.T
        B[n:, :n] = A + N
        B[n:, n:] = A + 2*N
        B[:n, n:] = A.T[::-1, ::-1] + 3*N
        n += n
        A = B
    return A


bits = 8
A = generate(bits)
im = cmap(A)
plt.imshow(A, cmap="plasma", origin="lower")
plt.show()
