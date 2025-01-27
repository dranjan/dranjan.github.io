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


def color_image(x):
    """
    Get RGBA data from the Hilbert curve coordinates. Edges will be
    darkened, and corners will be rounded.
    """
    # First we compute a mask to darken edges.
    e = scipy.signal.convolve(x + 4, edge_filter(1), mode='same')
    mask = -np.log(np.abs(e))
    mask -= mask.min()

    # Now we do some morphology to enhance corners.
    masks = [mask]
    for n in [1, 2, 3, 4]:
        masks.append(maxmin_filter(mask, circular_stencil(n)))
    mask = np.sum(masks, axis=0)/len(masks)

    # We separate the mask into two halves, a darkening mask and a
    # lightening mask.
    mask /= np.quantile(mask, 0.5)
    v1 = mask.max() + 1.0
    mask0 = mask <= 1
    mask1 = mask0 ^ True

    # It's worth experimenting with different colormaps here.
    # We want something that's perceptually uniform and doesn't get too
    # light or dark.
    cmap = plt.get_cmap('plasma')
    y = cmap(x)
    y[mask0, :3] *= mask[mask0, None]
    y[mask1, :3] = 1 - (1 - y[mask1, :3])*(v1 - mask[mask1, None])/(v1 - 1)

    return y


def edge_filter(n):
    """
    Return a simple edge detection filter based on a Gaussian kernel.
    The result will be square with 2*n + 1 elements on each side.
    """
    x = np.r_[-n:n+1]
    y = np.maximum(0, n + 1 - np.hypot(x, x[:, None]))
    y /= -y.sum()
    y[n, n] += 1
    return y


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


def pad_image(image, vpad=1.0, r=15, s=(3, 3)):
    """
    Add a transparent border to the given image. The result will be (1 + vpad)
    times as large as the input on the vertical axis, with equal padding on all
    four sides.

    The image will have a black border that fades away from the image. A shift
    can also be applied to add the illusion of depth.
    """
    size0, size1 = image.shape[:2]
    offset = int(size0 * (vpad/2))
    image2 = np.zeros((size0 + 2*offset, size1 + 2*offset, 4), dtype=np.uint8)
    image2[offset:offset + size0, offset:offset + size1] = image

    # Here's the black border and shadow. distance_transforme_edt is a
    # little bit too much machinery for what we're doing here, but we
    # already have the dependency and it's a one-liner, so...
    bg = np.ones(image2.shape[:2])
    bg[offset:offset + size0, offset:offset + size1] = 0
    d = scipy.ndimage.distance_transform_edt(bg)
    alpha = np.maximum(0, r - d) / r
    alpha = (alpha*255).astype(np.uint8)
    alpha = np.roll(alpha, s[0], axis=0)
    alpha = np.roll(alpha, s[1], axis=1)

    image2[..., 3] = np.maximum(image2[..., 3], alpha)

    return image2


bits = 8
A = hilbert_data(bits)
B = color_image(A)
image = (B[::-1]*255).astype(np.uint8)
os.makedirs('build', exist_ok=True)
PIL.Image.fromarray(image).save('build/favicon.ico')
PIL.Image.fromarray(image).save('build/output.png')
pad = 460/2**bits - 1
PIL.Image.fromarray(pad_image(image, pad)).save('build/output-padded.png')
