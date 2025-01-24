# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright 2025 Darsh Ranjan.
#
# /// script
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "Pillow",
#     "scipy",
# ]
# ///
"""
Usage: python run.py

Generates a visualization of a Hilbert curve.

There are three outputs:
1. ./build/favicon.ico
2. ./build/output.png
3. ./build/output-padded.png
"""
import os

from matplotlib import colormaps
from matplotlib import pyplot as plt
import numpy as np
import PIL
import scipy.signal
import scipy.ndimage


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


def edge_filter(n):
    """
    Return a simple edge detection filter based on a Gaussian kernel.
    The result will be square with 2*n + 1 elements on each side.
    """
    x = np.linspace(-10, 10, 2*n + 1)
    y = x[:, None]
    z = np.exp(-0.5*(x*x + y*y))
    z /= -z.sum()
    z[n, n] += 1
    return z


def circular_stencil(n):
    """
    Return a binary mask of shape (2*n + 1, 2*n + 1) approximating a
    filled circle.
    """
    x = np.linspace(-1, 1, 2*n+1)
    y = x[:, None]
    return x*x + y*y <= 1


def maxmin_filter(x, stencil):
    """
    Apply a minimum filter followed by a maximum filter to the input image.
    """
    y = scipy.ndimage.minimum_filter(x, footprint=stencil, mode='constant')
    z = scipy.ndimage.maximum_filter(y, footprint=stencil, mode='constant')
    return z


bits = 8
A = generate_image(bits)

e = scipy.signal.convolve(A + 4, edge_filter(5), mode='same')
e = np.abs(e)
v0 = e.min()
v1 = 25
v2 = 0.6
v3 = 4

mask = 1/(1 + ((e - v0)*v1)**v2)**v3

masks = [mask]
for n in [1, 2, 3, 4]:
    masks.append(maxmin_filter(mask, circular_stencil(n)))
mask = np.sum(masks, axis=0)/len(masks)

cmap = colormaps["plasma"]
B = cmap(A)
B[..., :3] *= mask[..., None]

image = (B[::-1]*255).astype(np.uint8)
os.makedirs('build', exist_ok=True)
PIL.Image.fromarray(image).save('build/favicon.ico')
PIL.Image.fromarray(image).save('build/output.png')
PIL.Image.fromarray(pad_image(image)).save('build/output-padded.png')
