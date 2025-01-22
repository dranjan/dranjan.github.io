"""
Usage: python run.py

Generates a visualization of a Hilbert curve.

There are two outputs:
1. ./build/favicon.ico
2. ./build/output-padded.png
"""
import os

from matplotlib import colormaps
import numpy as np
import PIL


def generate_image(bits):
    """
    Generate the single-channel image. Its shape will be 2**bits by 2**bits,
    and its values will be in the interval [0, 1).
    """
    A = np.array([[0]])
    n = 1
    for _ in range(bits):
        N = n*n
        n += n
        A = np.array(
            [[A.T, A.T[::-1, ::-1] + 3*N], [A + N, A + 2*N]]
        ).transpose((0, 2, 1, 3)).reshape(n, n)
    return A / (n*n)


def pad_image(image, vpad=0.5):
    """
    Add a transparent border to the given image. The result will be (1 + vpad)
    times as large as the input on the vertical axis, with equal padding on all
    four sides.
    """
    size0, size1 = image.shape[:2]
    offset = int(size0 * (vpad/2))
    image2 = np.zeros((size0 + 2*offset, size1 + 2*offset, 4), dtype=np.uint8)
    image2[offset:offset + size0, offset:offset + size1] = image
    return image2


bits = 8
A = generate_image(bits)
image = colormaps["plasma"](A[::-1], bytes=True)
os.makedirs('build', exist_ok=True)
PIL.Image.fromarray(image).save('build/favicon.ico')
PIL.Image.fromarray(pad_image(image)).save('build/output-padded.png')
