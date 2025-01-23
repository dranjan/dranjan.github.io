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

bits = 6
A = generate_image(bits)

Au = A[1:]
Ad = A[:-1]
Aud0 = np.abs(Au - Ad)/2
Aud1 = (Au + Ad)/2

Ar = A[:, 1:]
Al = A[:, :-1]
Arl0 = np.abs(Ar - Al)/2
Arl1 = (Ar + Al)/2

Aur = Au[:, 1:]
Aul = Au[:, :-1]
Adr = Ad[:, 1:]
Adl = Ad[:, :-1]
Audrl_min = np.stack((Aur, Aul, Adr, Adl), axis=2).min(axis=2)
Audrl_max = np.stack((Aur, Aul, Adr, Adl), axis=2).max(axis=2)
Audrl0 = (Audrl_max - Audrl_min)/2
Audrl1 = (Audrl_max + Audrl_min)/2

print(Aud0.min(), Aud0.max())
print(Arl0.min(), Arl0.max())
print(Audrl0.min(), Audrl0.max())

v0 = Aud0.min()
v1 = 100
v2 = 1


def get_scale(x):
    return 1/(1 + ((x - v0)*v1)**v2)


Aud0 = get_scale(Aud0)
Arl0 = get_scale(Arl0)
Audrl0 = get_scale(Audrl0)

Bsize = 2*A.shape[0] - 1

mask = np.ones((Bsize, Bsize))
mask[1:-1:2, 1:-1:2] = Audrl0
mask[::2, 1:-1:2] = Arl0
mask[1:-1:2, ::2] = Aud0


def make_stencil(n):
    x = np.linspace(-1, 1, 2*n+1)
    y = x[:, None]
    return x*x + y*y <= 1


def maxmin_filter(x, stencil):
    y = scipy.ndimage.minimum_filter(x, footprint=stencil, mode='constant')
    z = scipy.ndimage.maximum_filter(y, footprint=stencil, mode='constant')
    return z


#masks = [mask]
#for n in [1, 2, 3, 4]:
#    stencil = make_stencil(n)
#    masks.append(maxmin_filter(mask, stencil))
#mask = np.sum(masks, axis=0)/len(masks)

stencil = make_stencil(1)
mask1 = scipy.ndimage.minimum_filter(mask, footprint=stencil, mode='constant')
mask1 = scipy.ndimage.maximum_filter(mask1, footprint=stencil, mode='constant')
mask = 0.75*mask + 0.25*mask1

cmap = colormaps["plasma"]

B = np.empty((Bsize, Bsize, 4))
B[..., 3] = 1
B[::2, ::2, :] = cmap(A)
B[1:-1:2, 1:-1:2, :3] = cmap(Audrl1)[..., :3]
B[::2, 1:-1:2, :3] = cmap(Arl1)[..., :3]
B[1:-1:2, ::2, :3] = cmap(Aud1)[..., :3]
B[..., :3] *= mask[..., None]

#plt.imshow(B, origin='lower')
#plt.show()

#fil0 = np.array([[1, -1], [1, -1]])
#fil1 = np.array([[1, 1], [-1, -1]])
#
#A0 = np.abs(scipy.signal.convolve(A, fil0, mode='valid'))
#A1 = np.abs(scipy.signal.convolve(A, fil1, mode='valid'))
#Am = np.maximum(A0, A1)
#import matplotlib.pyplot as plt
#plt.imshow(np.log(Am), origin='lower', cmap='viridis')
#print(Am.min())
#plt.show()

#image = colormaps["plasma"](A[::-1], bytes=True)
image = (B[::-1]*255).astype(np.uint8)
print(B.shape)
print(image.shape)
os.makedirs('build', exist_ok=True)
PIL.Image.fromarray(image).save('build/favicon.ico')
PIL.Image.fromarray(image).save('build/output.png')
PIL.Image.fromarray(pad_image(image)).save('build/output-padded.png')
