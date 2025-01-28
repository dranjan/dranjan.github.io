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
2. ./build/hilbert.png
3. ./build/avatar.png
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


def color_image(A, roundness=11):
    """
    Get RGBA data from the Hilbert curve coordinates. Edges will be
    darkened, and corners will be rounded.

    The `roundness` parameter can be any nonnegative integer, and a
    value of 0 disables the corner-rounding entirely.
    """
    # First we compute a mask to darken edges.
    e = scipy.signal.convolve(A + 4, edge_filter(1), mode='same')
    mask = -np.log(np.abs(e))
    mask -= mask[mask.shape[0]//2, mask.shape[1]//2]
    mask[mask < 0] = 0

    # Now we do some morphology to enhance corners.
    masks = [mask]
    for n in range(1, roundness):
        masks.append(maxmin_filter(mask, circular_stencil(n)))
    mask = np.sum(masks, axis=0)/len(masks)

    # We separate the mask into two parts, a darkening mask and a
    # lightening mask.
    mask /= np.quantile(mask, 0.75)
    v1 = mask.max() + 1.0
    mask0 = mask <= 1
    mask1 = mask0 ^ True

    # It's worth experimenting with different colormaps here.
    # We want something that's perceptually uniform and doesn't get too
    # light or dark.
    cmap = plt.get_cmap('plasma')
    B = cmap(A)
    B[mask0, :3] *= mask[mask0, None]
    B[mask1, :3] = 1 - (1 - B[mask1, :3])*(v1 - mask[mask1, None])/(v1 - 1)

    return B


def edge_filter(n):
    """
    Return a simple edge detection filter.
    The result will be square with 2*n + 1 elements on each side.
    """
    x = np.r_[-n:n+1]
    A = np.maximum(0, n + 1 - np.hypot(x, x[:, None]))
    A /= -A.sum()
    A[n, n] += 1
    return A


def circular_stencil(n):
    """
    Return a binary mask of shape (2*n + 1, 2*n + 1) approximating a
    filled circle.
    """
    x = np.linspace(-1, 1, 2*n+1)
    y = x[:, None]
    return x*x + y*y <= 1


def maxmin_filter(A, stencil):
    """
    Apply a minimum filter followed by a maximum filter to the input image.
    """
    B = scipy.ndimage.minimum_filter(A, footprint=stencil, mode='constant')
    C = scipy.ndimage.maximum_filter(B, footprint=stencil, mode='constant')
    return C


def pad_image(A, dim=(460, 460), r=10, s=(3, 3)):
    """
    Add a transparent border to the given image, padding it to the specified
    dimensions. The input image will be centered in the result.

    The image will have a black border that fades away from the image. A shift
    can also be applied to add the illusion of depth.
    """
    size0, size1 = A.shape[:2]
    offset0 = (dim[0] - size0)//2
    offset1 = (dim[1] - size1)//2
    B = np.zeros((dim[0], dim[1], 4), dtype=np.uint8)
    B[offset0:offset0 + size0, offset1:offset1 + size1] = A

    # Here's the black border and shadow. distance_transforme_edt is a
    # little bit too much machinery for what we're doing here, but we
    # already have the dependency and it's a one-liner, so...
    bg = np.ones(B.shape[:2])
    bg[offset0:offset0 + size0, offset1:offset1 + size1] = 0
    d = scipy.ndimage.distance_transform_edt(bg)
    alpha = np.maximum(0, r - d) / r
    alpha = (alpha*255).astype(np.uint8)
    alpha = np.roll(alpha, s[0], axis=0)
    alpha = np.roll(alpha, s[1], axis=1)

    B[..., 3] = np.maximum(B[..., 3], alpha)

    return B


def shrink_image(A):
    """
    Shrink the input image by a factor of two. This is done carefully
    by first adding a one-pixel border, so that internal Hilbert
    curve borders get sharpened.
    """
    s0, s1 = A.shape[:2]
    B = np.zeros((s0 + 2, s1 + 2, 4), dtype=np.uint8)
    B[..., 3] = 255
    B[1:-1, 1:-1, :] = A
    B = B.reshape(s0//2 + 1, 2,  s1//2 + 1, 2, 4)
    return B.mean(axis=(1, 3)).astype(np.uint8)


bits = 9
A = hilbert_data(bits)
B = color_image(A)
image = (B[::-1]*255).astype(np.uint8)
image = shrink_image(image)
os.makedirs('build', exist_ok=True)
PIL.Image.fromarray(image).save('build/favicon.ico')
PIL.Image.fromarray(image).save('build/hilbert.png')
PIL.Image.fromarray(pad_image(image)).save('build/avatar.png')
