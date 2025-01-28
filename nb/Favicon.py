# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # About the Favicon
#
# If you've spent some time on my website...firstly, thank you! Secondly, you may have noticed a colorful little square icon in the browser tab.
# When I first started this site, I was missing an icon, so naturally I set about making one for myself.
# What then began as some throwaway code to generate a tiny image soon became its own project and ultimately turned into something I'm kind of proud of, so I'm going to explain what it is and how I made it.
#
# ## The Hilbert Curve
#
# It's a Hilbert Curve.
#
# As you may have noticed, I'm kind of a fan of fractals.
# As far as fractals go, the Hilbert Curve is a true classic, dating back to 1891 when it was first discovered by the legendary mathematician David Hilbert.
# It belongs to a class of fractals called _space-filling curves_.
# Nowadays we know of countless examples of space-filling curves, but Hilbert's is still remarkable in its simplicity.
# It can be defined as the unique continuous function $H:[0,1]\to[0,1]\times[0,1]$ satisfying the functional equation
# $$
# \begin{align*}
# H(x) &= (f(x), g(x)) = \frac12\cdot
# \begin{cases}
# (g(4x), f(4x)) &\text{if } 0\leq x \leq\tfrac14, \\
# (f(4x-1), 1+g(4x-1)) &\text{if } \tfrac14 \leq x \leq \tfrac12, \\
# (1+f(4x-2), 1+g(4x-2)) &\text{if } \tfrac12 \leq x \leq \tfrac34, \\
# (2-g(4x-3), 1-f(4x-3)) &\text{if } \tfrac34 \leq x \leq 1,
# \end{cases} \\
# H(0) &= (0, 0), \\
# H(1) &= (1, 0).
# \end{align*}
# $$
# As for implementation, we can pretty much use exactly that, although technically what we compute is
# more like the inverse of that mapping.
#
# Here's the game plan: first we'll pick a nice colormap for the input interval $[0, 1]$, and then we'll
# apply the $H$ function above to transfer the colors onto the square.
# Technically this isn't a well defined result since $H$ isn't a one-to-one function, but
# it is approximated arbitrarily well by one-to-one functions, so it's fine in practice.
# We can construct these one-to-one approximations just by iterating the functional equation
# a finite number of times.
# By choosing a power-of-two grid for the square, we can additionally ensure that these
# approximations touch each pixel exactly once, so it's kind of perfect.
#
# This is the whole implementation:

# +
import matplotlib.pyplot as plt
import numpy as np

bits = 9
A = np.array([[0.5]])
for _ in range(bits):
    A = np.block([[A.T, A.T[::-1, ::-1] + 3], [A + 1, A + 2]])/4

plt.imshow(A, origin='lower', cmap='plasma')
plt.axis('off')
plt.show()
# -

# That's it! We can stop here, and just for the website favicon, there's arguably no reason to go any further.
# However, I'm not quite done yet.
# I think we can't do too much better than this with the simple color mapping approach,
# but in my opinion some of the most interesting fractal structure is difficult to get from this visualization,
# so we're going to look for a way to see it better in all its glory.
#
# ## Edge Enhancement
#
# We can try to get a better look at the fractal structure by enhancing color discontinuities.
# Here's the motivation: you may notice that the color of the image above changes abruptly
# across certain horizontal and vertical lines.
# Those lines have an interesting branching structure, but as they get smaller, the color
# changes also get smaller.
# So those discontinuities seem to contain a lot of the interesting structure, but they
# quickly become really difficult to see, and that's a fundamental limitation of the simple color mapping
# approach.
#
# To get around that, we can do additional processing to make these edge discontinuities stand out
# better.
# There are many ways to do this, but what we'll do is apply a _discrete Laplacian_ operator,
# which finds pixels whose values differ significantly from averages over small neighborhoods.
# That will give each pixel a "score" of how important of an edge it appears to be on,
# and we can then use those scores to effectively "draw" those edges in by darkening the
# corresponding pixels.
#
# (A radically different approach is not to _draw_ the edges in explicitly, but simply
# to fudge the color values around to magnify the smaller color jumps.
# _Unsharp masking_ is probably the best-known example of that technique.
# It's a completely valid method, but in my experience the results aren't as satisfying as
# the edge-darkening we're going to do, so we'll just leave it there.
# I will point out, though, that one of the significant drawbacks of the edge
# darkening is that we effectively have to sacrifice a bunch of pixels to turn into the
# edges, and the unsharp masking approach doesn't have that cost.)
#
# I didn't try too hard to optimize the filter coefficients defined below,
# but the most important thing is that it use a very small neighborhood to average over.
# We're interested in very small details, and larger neighborhoods would average
# those details out, which would be bad.

# +
import scipy

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

e = scipy.signal.convolve(A + 4, edge_filter(1), mode='same')
m = -np.log(np.abs(e))
m -= m[m.shape[0]//2, m.shape[1]//2]
m /= np.quantile(m, 0.75)

mask = np.clip(m, 0, 1)
# -

# Now we have a mask we can multiply over the image to darken the places where the color changes abruptly.
# We applied a log-scale transformation to further enhance the smallest details.
# We also normalized the values so that:
#
# - the obvious discontinuity right in the middle gets the maximum penalty, and
# - the smoothest 25% of pixels are deemed "non-edges" and won't be penalized.
#
# Here's the result. Also from now on, as a small technicality, we're going to look at the image in its natural resolution
# using the `IPython.display` module instead of `imshow` for maximum fidelity.

# +
cmap = plt.get_cmap('plasma')
B = cmap(A)
B[..., :3] *= mask[..., None]

import io
import IPython.display

def embed(data):
    stream = io.BytesIO()
    plt.imsave(stream, data, origin='lower', format='png')
    display(IPython.display.Image(stream.getvalue()))

embed(B)


# -

# Now we can really see the structure, but it's got the opposite problem:
# there's so much detail that it's hard for the eye to follow.
# We'll try to address that next.
#
# ## Rounding Corners
#
# It would be easier to see the path of the space-filling curve across the square if we could see where it was turning, right?
# To do that, we'll need to sacrifice some of the finest details in the masked image, but we'll try to make it worth the trade.
# We'll do it by applying a corner-rounding morphology operation.
# The technical name for it is _closing_, but I'll call it something else because in my opinion the terms _opening_ and _closing_ are too
# ambiguous and context-dependent.
# So I'll call it "maxmin" because it has the form $\operatorname{max}(\operatorname{min}(\ldots))$.
# To make the effect less abrupt, we'll apply several different corner radii and average over all the results.

# +
def maxmin_filter(A, stencil):
    """
    Apply a minimum filter followed by a maximum filter to the input image.
    """
    B = scipy.ndimage.minimum_filter(A, footprint=stencil, mode='constant')
    C = scipy.ndimage.maximum_filter(B, footprint=stencil, mode='constant')
    return C

def circular_stencil(n):
    """
    Return a binary mask of shape (2*n + 1, 2*n + 1) approximating a
    filled circle.
    """
    x = np.linspace(-1, 1, 2*n+1)
    y = x[:, None]
    return x*x + y*y <= 1

def round_corners(mask, roundness):
    masks = [mask]
    for n in range(1, roundness + 1):
        masks.append(maxmin_filter(mask, circular_stencil(n)))
    return np.sum(masks, axis=0)/len(masks)

roundness = 10
mask = round_corners(mask, roundness)
B = cmap(A)
B[..., :3] *= mask[..., None]
embed(B)


# -

# It seems to have done the right thing, but it's pretty clear that our normalization is no longer correct.
# To fix that, let's move some of the normalization to after the corner rounding, and let's also add the possibility of
# brightening the top few percent of mask values, to partially compensate for the fact that we're generally making everything darker.

# +
def get_edge_mask(A, roundness):
    e = scipy.signal.convolve(A + 4, edge_filter(1), mode='same')
    m = -np.log(np.abs(e))
    m -= m[m.shape[0]//2, m.shape[1]//2]
    m[m < 0] = 0
    mask = round_corners(m, roundness)
    mask /= np.quantile(mask, 0.75)
    return mask

def apply_edge_mask(A, mask):
    v1 = mask.max() + 1.0
    mask0 = mask <= 1
    mask1 = mask0 ^ True

    B = cmap(A)
    B[mask0, :3] *= mask[mask0, None]
    B[mask1, :3] = 1 - (1 - B[mask1, :3])*(v1 - mask[mask1, None])/(v1 - 1)
    return B

B = apply_edge_mask(A, get_edge_mask(A, roundness=10))
embed(B)
# -

# I think that strikes a pretty good compromise between our criteria.
# We can see a lot of small details, but it's still possible to follow the path from the blue parts through
# purple, pink, orange, and finally yellow.
#
# If it's still hard to see the larger-scale structure, simply increasing the `roundness` value works pretty well,
# but it does increase the computation time a bit, and of course we lose even more detail:

embed(apply_edge_mask(A, get_edge_mask(A, roundness=25)))


# ## Resharpening
#
# You might notice that the dark edges in the image are all two pixels wide, which seems really unnecessary in my opinion.
# The edge width is a feature of the edge detection filter we applied, and it's kind of impossible for it to do better than two pixels,
# since every edge will be detected from both sides.
# That's fine, though: since we can choose our resolution (as long as it's a power of two),
# we can just render the image at double the resolution we actually want, and then shrink it by a factor of two!
# Then our two-pixel edges should shrink down to just one pixel.
# However, we need to be careful here: the middle of every edge is always an even number of pixels from the image boundaries.
# That means if we average over two-by-two blocks to shrink the image, the edges will actually fall _between_ the blocks
# and won't be shrunk down at all!
# We can avoid that just by adding a one-pixel boundary to the image.
#
# I also owe you a small confession now: we've _already_ rendered at twice the resolution I actually want,
# so let's just continue on.

# +
def shrink_image(A):
    """
    Shrink the input image by a factor of two. This is done carefully
    by first adding a one-pixel border, so that internal Hilbert
    curve borders get sharpened.
    """
    s0, s1 = A.shape[:2]
    B = np.zeros((s0 + 2, s1 + 2, 4))
    B[..., 3] = 1
    B[1:-1, 1:-1, :] = A
    B = B.reshape(s0//2 + 1, 2,  s1//2 + 1, 2, 4)
    return B.mean(axis=(1, 3))

C = shrink_image(B)
embed(C)
# -

# This is it, for the most part. We'll go into some more minor technicalities below,
# but they won't change this final result of the computation.

# ## Padding
#
# If I wanted to...I don't know, create an avatar for my public GitHub profile, there are a few more considerations.
# First, the optimum size for GitHub avatars seems to be 460 pixels by 460 pixels.
# Second, the visible part of the avatar is, for reasons I still can't understand, a _circular_ region inside of the full avatar image.
# Third, the background the avatar is shown on doesn't have a fixed color and depends on the user's browser settings and selected theme.
#
# Now, the image we just created is 257 pixels by 257 pixels:

display(C.shape)


# Some quick math shows that a circle inscribed in a square of dimensions 460 pixels can contain the entire square of dimensions 257 pixels,
# with plenty of room to spare.
# Without going into a massive rant about GitHub's circular avatars,
# what made sense to me was to add a dark border that fades out into transparency.
# We can also offset the border to make it look like a shadow, which looks kind of cool.
#
# Here's the code for that, and the final result.

# +
def pad_image(A, dim=(460, 460), r=10, s=(-3, 3)):
    """
    Add a transparent border to the given image, padding it to the specified
    dimensions. The input image will be centered in the result.

    The image will have a black border that fades away from the image. A shift
    can also be applied to add the illusion of depth.
    """
    size0, size1 = A.shape[:2]
    offset0 = (dim[0] - size0)//2
    offset1 = (dim[1] - size1)//2
    B = np.zeros((dim[0], dim[1], 4))
    B[offset0:offset0 + size0, offset1:offset1 + size1] = A

    # Here's the black border and shadow. distance_transforme_edt is a
    # little bit too much machinery for what we're doing here, but we
    # already have the dependency and it's a one-liner, so...
    bg = np.ones(B.shape[:2])
    bg[offset0:offset0 + size0, offset1:offset1 + size1] = 0
    d = scipy.ndimage.distance_transform_edt(bg)
    alpha = np.maximum(0, r - d) / r
    alpha = np.roll(alpha, s[0], axis=0)
    alpha = np.roll(alpha, s[1], axis=1)

    B[..., 3] = np.maximum(B[..., 3], alpha)

    return B

D = pad_image(C)
embed(D)
# -

# ## Acknowledgements
#
# A lot of heavy lifting is being done by the colormap. The
# scientific programming community has done some great work in creating
# colormaps that are both visually appealing and avoid distortion as
# perceived visually by humans, and both of those qualities are valuable
# here. Matplotlib's excellent `plasma` colormap, which is the one I've selected here,
# was created by [St&eacute;fan van der Walt and Nathaniel
# Smith](https://bids.github.io/colormap/).
