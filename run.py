from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np

cmap = colormaps["plasma"]


def generate(bits):
    if not bits:
        return np.array([[0]])
    A = generate(bits - 1)
    n = 2**(bits - 1)
    N = n*n
    result = np.empty((2*n, 2*n), dtype=int)
    result[:n, :n] = A.T
    result[n:, :n] = A + N
    result[n:, n:] = A + 2*N
    result[:n, n:] = A.T[::-1, ::-1] + 3*N
    return result


bits = 8
A = generate(bits)
im = cmap(A)
plt.imshow(A, cmap="plasma", origin="lower")
plt.show()
