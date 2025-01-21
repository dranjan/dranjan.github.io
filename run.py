"""
Usage: python run.py

Generates a visualization of a Hilbert curve. The output is saved in
build/output.png.
"""
import os

from matplotlib import colormaps
import numpy as np
import PIL


def hilbert_curve_coordinate(bits):
    A = np.array([[0]])
    n = 1
    for _ in range(bits):
        N = n*n
        n += n
        A = np.array(
            [[A.T, A.T[::-1, ::-1] + 3*N], [A + N, A + 2*N]]
        ).transpose((0, 2, 1, 3)).reshape(n, n)
    return A / (n*n)


bits = 8
A = hilbert_curve_coordinate(bits)
image = PIL.Image.fromarray(colormaps["plasma"](A[::-1], bytes=True))
os.makedirs('build', exist_ok=True)
image.save('build/favicon.ico')
