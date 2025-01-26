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

from matplotlib import pyplot as plt
import numpy as np
import PIL
import scipy.signal
import scipy.ndimage


def hilbert_data(bits):
    """
    Compute Hilbert curve coordinates on a power-of-two square grid.
    The result has shape (2**bits, 2**bits) and values in the interval
    [0, 1).
    """
    A = np.array([[0]])
    for _ in range(bits):
        A = np.block([[A.T, A.T[::-1, ::-1] + 3], [A + 1, A + 2]])/4
    return A


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


# This is the "pure" 2D Hilbert curve data. It makes a fine image on its
# own, but we'll do some further processing to enhance edges and
# corners.
bits = 8
A = hilbert_data(bits)

# We do some edge enhancement here, specifically making them darker.
# First we apply a linear edge detection filter, with some nonlinear
# postprocessing.
e = scipy.signal.convolve(A + 4, edge_filter(5), mode='same')
e = np.abs(e)
v0 = e.min()
v1 = 25
v2 = 0.6
v3 = 4

mask = 1/(1 + ((e - v0)*v1)**v2)**v3

# Finally we do some morphology here to enhance corners.
masks = [mask]
for n in [1, 2, 3, 4]:
    masks.append(maxmin_filter(mask, circular_stencil(n)))
mask = np.sum(masks, axis=0)/len(masks)

# It's worth experimenting with different colormaps here, but nothing
# seems to beat plasma. We want something that's perceptually uniform
# and doesn't get too light or dark.
cmap = plt.get_cmap('plasma')
B = cmap(A)
B[..., :3] *= mask[..., None]

image = (B[::-1]*255).astype(np.uint8)
os.makedirs('build', exist_ok=True)
PIL.Image.fromarray(image).save('build/favicon.ico')
PIL.Image.fromarray(image).save('build/output.png')
PIL.Image.fromarray(pad_image(image)).save('build/output-padded.png')
