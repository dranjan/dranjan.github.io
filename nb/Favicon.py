# ---
# jupyter:
#   jekyll:
#     katex: true
#     page_toc: true
#     title: "About the Favicon"
#     parent: "Mathematical Art"
#   jupytext:
#     formats: ipynb,py:light
#     notebook_metadata_filter: jekyll
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

# # About the Favicon <!--insert:no-toc-->
#
# ***Or: Visualizing Hilbert Curves in Python***
#
# <!--insert:toc-->
#
# If you've spent some time on my website...firstly, thank you! Secondly, you may have noticed a colorful little square icon in the browser tab.
# When I first started this site, I was missing an icon, so naturally I set about making one for myself.
# What then began as some throwaway code to generate a tiny image soon became its own project and ultimately turned into something pretty cool,
# so I'm going to explain what it is and how I made it.
#
# ## The Hilbert Curve
#
# It's a Hilbert curve.
#
# You may have noticed from the rest of my site that I'm kind of a fan of fractals.
# As far as fractals go, the Hilbert curve is a true classic, dating back to 1891 when it was first discovered by the legendary mathematician David Hilbert.
# It belongs to a class of fractals called _space-filling curves_.
# Nowadays we know of countless examples of space-filling curves, but Hilbert's is still remarkable in its simplicity.
# It can be defined as the unique continuous function <!--begin:mathinline-->$H:[0,1]\to[0,1]\times[0,1]$<!--end:mathinline-->
# satisfying the functional equation
#
# $$
# \begin{align*}
# H(v) &= (f(v), g(v)) = \frac12\cdot
# \begin{cases}
# (g(4v), f(4v)) &\text{if } 0\leq v \leq\tfrac14, \\
# (f(4v-1), 1+g(4v-1)) &\text{if } \tfrac14 \leq v \leq \tfrac12, \\
# (1+f(4v-2), 1+g(4v-2)) &\text{if } \tfrac12 \leq v \leq \tfrac34, \\
# (2-g(4v-3), 1-f(4v-3)) &\text{if } \tfrac34 \leq v \leq 1,
# \end{cases} \\
# H(0) &= (0, 0), \\
# H(1) &= (1, 0).
# \end{align*}
# $$
#
# Here's the game plan: first we'll pick a nice colormap for the input interval <!--begin:mathinline-->$[0, 1]$<!--end:mathinline-->, and then we'll
# apply this <!--begin:mathinline-->$H$<!--end:mathinline--> function to transfer the colors onto the square.
# Here's the mathematical representation of that:
#
# $$
# \operatorname{Color}(x, y) = \operatorname{Colormap}(H^{-1}(x, y)),
# $$
#
# where <!--begin:mathinline-->$\operatorname{Color}(x, y)$<!--end:mathinline--> is the per-pixel color
# we're trying to compute on the unit square, and <!--begin:mathinline-->$\operatorname{Colormap}$<!--end:mathinline-->
# is our chosen colormap for the unit interval.
# There's a slight complication here because <!--begin:mathinline-->$H$<!--end:mathinline--> isn't one-to-one,
# so <!--begin:mathinline-->$H^{-1}$<!--end:mathinline--> isn't technically well defined,
# but in practice this is no big deal, because <!--begin:mathinline-->$H$<!--end:mathinline-->
# has a sequence of discrete approximations that are fully bijective,
# which we get by iterating the functional equation a finite number of times
# and discretizing both sides to powers of two.
#
# As far as colormaps go, right now we mainly just want something that's dynamic and perceptually uniform,
# but later on we'll also want it to avoid becoming overly dark.
# It seems quite difficult to achieve all of those and still maintain some semblance of aesthetics,
# but Matplotlib's `plasma` does it pretty well in my opinion, so that's what we'll use.
#
# As we get started with code, if you want to follow along with this implementation, we need to get a couple things sorted first.
# First, you'll need these dependencies:
#
#     pip install numpy scipy matplotlib pillow
#
# Second, you'll need a way to visualize raw RGB image data.
# I suggest using a [Jupyter](https://jupyter.org/) notebook, which is how I made everything you see here.
# In that case we can do it conveniently like this:

# +
import io
import IPython.display

def show(rgba):
    stream = io.BytesIO()
    plt.imsave(stream, rgba, origin='lower', format='png')
    display(IPython.display.Image(stream.getvalue()))


# -

# (`matplotlib.pyplot.imshow` also works, but keep in mind that it likes to resize images, which can lead to misleading results.)
#
# With that out of the way, here's the implementation of all of that math:

# +
import matplotlib.pyplot as plt
import numpy as np

def hilbert_data(bits):
    v = np.array([[0.5]])
    for _ in range(bits):
        v = np.block([[v.T, v.T[::-1, ::-1] + 3], [v + 1, v + 2]])/4
    return v

hilbert_values = hilbert_data(bits=9)
cmap = plt.get_cmap('plasma')
hilbert_rgba = cmap(hilbert_values)
show(hilbert_rgba)
# -

# That's it! We can stop here, and just for the website favicon, there's arguably no reason to go any further.
# However, I'm not quite done yet.
# I think we can't do too much better than this with the pure color mapping approach,
# but in my opinion some of the most interesting fractal structure is difficult to get from this visualization,
# so we're going to look for a way to see it better in all its glory.
#
# But before making things more complicated, let's briefly pause and simplify.
# By looking at the `hilbert_values` data in its raw form,
# maybe we can build some intuition for how this thing behaves.
# By construction, the array `hilbert_values` has values in the interval <!--begin:mathinline-->$(0,1)$<!--end:mathinline-->,
# but we'll be renormalizing the values, and the reason for that should be clear momentarily.
# We'll also decrease the number of iterations so we can comfortably look at all of the values.

bits = 3
data = hilbert_data(bits) * 2**(2*bits) - 0.5
display(data[::-1])  # reverse the rows to align with the image


# Starting with 0 in the bottom left corner, we have all the numbers 0, 1, ..., 63, and
# we can always get from one number to the next by going one step up, down, left, or right.
# So it really is a single continuous path that seems to fill up a whole square, hence the term "space-filling."
# With the help of Matplotlib, we can even look at that path directly:

# +
# This may already be in NumPy, but I can't seem to find it.
def argsort_nd(array):
    """
    Return a tuple of index arrays

        (idx[0], ..., idx[n-1])

    such that array[idx[0], ..., idx[n-1]]
    sorts the values of `array`, where n = array.ndim.
    Each array idx[k] has length equal to array.size.
    """
    idx_flat = np.argsort(array, axis=None)
    idx = [0]*array.ndim
    for dim in range(array.ndim - 1, -1, -1):
        m = array.shape[dim]
        idx[dim] = (idx_flat % m)
        idx_flat //= m
    return tuple(idx)

def plot_path(bits):
    n = 2**bits
    hilbert_values = hilbert_data(bits)
    y, x = argsort_nd(hilbert_values)
    plt.plot(x, y)
    plt.axis('equal')
    plt.axis('off')

plot_path(bits=3)
plt.show()
# -

# And if we do one more iteration, we can see more of the recursive fractal structure emerge:

plot_path(bits=4)
plt.show()

# To make this even clearer, we can put levels 1-4 next to each other,
# so we can see how each subsequent iteration level connects together four small copies of the previous level.

for bits in range(1, 5):
    plt.subplot(2, 2, bits)
    plot_path(bits)
    plt.title(f"Level {bits}")
plt.show()

# To see this same type of structure in the original plasma-colormapped image, we need to start at the darkest color
# in the lower left corner and try to trace out a path that makes the colors change as continuously as possible
# while traversing over the whole square.
# Let's repeat the original image here for easy reference:

hilbert_values = hilbert_data(bits=9)
hilbert_rgba = cmap(hilbert_values)
show(hilbert_rgba)

# On my computer monitor, my eyes can only discern detail up to about level four here,
# which is a shame because the image uses nine iterations and thus
# technically should have much finer detail than that.
# Our next task will thus be to see if we can bring out a little bit more of the fractal structure
# without throwing out the whole colormap concept.

# ## Edge Enhancement
#
# We can try to get a better look at the fractal structure by enhancing color discontinuities.
# Here's the motivation: the color changes abruptly
# across certain horizontal and vertical lines.
# Those discontinuity lines happen to be exactly the "negative space" of the Hilbert curve,
# and we can treat them as such by darkening them, for example.
# Then maybe the curve itself will be easier to make out visually as the remaining positive space.
#
# We'll do this by applying a _discrete Laplacian_ operator,
# which finds pixels whose values differ significantly from averages over small neighborhoods.
# That will give each pixel a "score" of how important of an edge it appears to be on,
# and we can then use those scores to darken the corresponding pixels, i.e., shunt them into the
# visual background.
#
# I didn't try too hard to optimize the filter coefficients defined below,
# but the most important thing is for it to use a very small neighborhood to average over.
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
    coefficients = np.maximum(0, n + 1 - np.hypot(x, x[:, None]))
    coefficients /= -coefficients.sum()
    coefficients[n, n] += 1
    return coefficients

# SciPy implicitly zero-pads our data inside `convolve`, so if we
# temporarily shift all the values by some big number (4), the boundary
# of the image will be detected as edges.
# This is consistent with the positive/negative space idea.
edges = scipy.signal.convolve(hilbert_values + 4, edge_filter(1), mode='same')
mask = -np.log(np.abs(edges))
mask -= mask[mask.shape[0]//2, mask.shape[1]//2]
mask /= np.quantile(mask, 0.75)

mask = np.clip(mask, 0, 1)
# -

# Now we have a mask we can multiply over the image to darken the places where the color changes abruptly.
# We applied a log-scale transformation to further enhance the smallest details, and
# we also normalized the mask values so that:
#
# - the obvious discontinuity right in the middle gets the maximum penalty, and
# - the smoothest 25% of pixels are deemed pure foreground and won't be darkened at all.
#
# Here's the result.

hilbert_rgba[..., :3] *= mask[..., None]
show(hilbert_rgba)


# Now we can really see the structure, but it's got the opposite problem:
# there's so much detail that it's hard for the eye to follow.
# We'll try to address that next.
#
# ## Rounding Corners
#
# It would be easier to see the path of the space-filling curve across the square if we could see where it was turning
# right and left, right?
# To do that, we'll need to sacrifice some of the finest details in the masked image, but maybe it will be worth the trade.
# We'll do it by applying a corner-rounding morphology operation.
# The technical name for it is _opening_, but I'll call it something else because in my opinion the terms _opening_ and _closing_
# from morphology are not intuitive at all.
# So I'll call it "maxmin" because it has the form <!--begin:mathinline-->$\operatorname{max}(\operatorname{min}(\ldots))$<!--end:mathinline-->.
# To make the effect less abrupt, we'll apply several different corner radii and average over all the results.

# +
def maxmin_filter(data, stencil):
    """
    Apply a minimum filter followed by a maximum filter to the input image.
    """
    data0 = scipy.ndimage.minimum_filter(data, footprint=stencil, mode='constant')
    return scipy.ndimage.maximum_filter(data0, footprint=stencil, mode='constant')

def circular_stencil(n):
    """
    Return a binary mask of shape (2*n + 1, 2*n + 1) approximating a
    filled circle.
    """
    x = np.linspace(-1, 1, 2*n+1)
    y = x[:, None]
    return x*x + y*y <= 1

def round_corners(mask, roundness):
    """
    Compute a new mask with internal corners rounded off.
    The nonnegative integer `roundness` value controls how much rounding is done,
    with 0 meaning none.
    """
    masks = [mask]
    for n in range(1, roundness + 1):
        masks.append(maxmin_filter(mask, circular_stencil(n)))
    return np.mean(masks, axis=0)

roundness = 10
mask = round_corners(mask, roundness)
hilbert_rgba = cmap(hilbert_values)
hilbert_rgba[..., :3] *= mask[..., None]
show(hilbert_rgba)


# -

# It seems to have done the right thing, but it's pretty clear that our normalization is no longer correct.
# To fix that, let's move some of the normalization to after the corner rounding, and let's also add the possibility of
# brightening the top few percent of mask values, to partially compensate for the fact that we're generally making everything darker.

# +
def get_edge_mask(data, roundness):
    """
    Compute a mask to darken edge discontinuities.
    Internal corners will be rounded according to the nonnegative integer
    `roundness` value, with 0 meaning none.
    """
    edges = scipy.signal.convolve(data + 4, edge_filter(1), mode='same')
    mask = -np.log(np.abs(edges))
    mask -= mask[mask.shape[0]//2, mask.shape[1]//2]
    mask[mask < 0] = 0
    mask = round_corners(mask, roundness)
    mask /= np.quantile(mask, 0.75)
    return mask

def apply_mask(rgba, mask):
    """
    Darken or lighten image values according to the mask. Mask values should
    be nonnegative, with values less than 1 darkening and values greater than 1
    lightening.
    """
    v1 = mask.max() + 1.0
    mask0 = mask <= 1
    mask1 = mask0 ^ True

    rgba = rgba.copy()
    rgba[mask0, :3] *= mask[mask0, None]
    rgba[mask1, :3] = 1 - (1 - rgba[mask1, :3])*(v1 - mask[mask1, None])/(v1 - 1)
    return rgba

hilbert_rgba_raw = cmap(hilbert_values)
mask = get_edge_mask(hilbert_values, roundness=10)
hilbert_rgba = apply_mask(hilbert_rgba_raw, mask)
show(hilbert_rgba)
# -

# I think that strikes a pretty good compromise between our criteria.
# We can see a lot of small details, and it's possible to visually follow the path from the blue parts through
# purple, pink, orange, and finally yellow.
# Also, it isn't too bright or dark.
#
# If it's still hard to see the larger-scale structure, simply increasing the `roundness` value works pretty well,
# but it does increase the computation time a bit, and of course we lose even more detail:

show(apply_mask(hilbert_rgba_raw, get_edge_mask(hilbert_values, roundness=25)))


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
def shrink_image(rgba):
    """
    Shrink the input image by a factor of two. This is done carefully
    by first adding a one-pixel border, so that internal Hilbert
    curve borders get sharpened.
    """
    s0, s1 = rgba.shape[:2]
    rgba_padded = np.zeros((s0 + 2, s1 + 2, 4))
    rgba_padded[..., 3] = 1
    rgba_padded[1:-1, 1:-1, :] = rgba
    return rgba_padded.reshape(s0//2 + 1, 2,  s1//2 + 1, 2, 4).mean(axis=(1, 3))

hilbert_rgba_small = shrink_image(hilbert_rgba)
show(hilbert_rgba_small)
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
# The image we just created is 257 pixels by 257 pixels:

hilbert_rgba_small.shape


# Some quick math shows that a circle inscribed in a square of dimensions 460 pixels can contain the entire square of dimensions 257 pixels,
# with plenty of room to spare.
# Without going into a massive rant about GitHub's circular avatars,
# what made sense to me was to add a dark border that fades out into transparency.
# We can also offset the border to make it look like a shadow, which looks kind of cool.
#
# Here's the code for that, and the final result.

# +
def pad_image(rgba, dim=(460, 460), r=10, s=(-3, 3)):
    """
    Add a transparent border to the given image, padding it to the specified
    dimensions. The input image will be centered in the result.

    The image will have a black border that fades away from the image. A shift
    can also be applied to add the illusion of depth.
    """
    size0, size1 = rgba.shape[:2]
    offset0 = (dim[0] - size0)//2
    offset1 = (dim[1] - size1)//2
    rgba_padded = np.zeros((dim[0], dim[1], 4))
    rgba_padded[offset0:offset0 + size0, offset1:offset1 + size1] = rgba

    # Here's the black border and shadow. distance_transform_edt is a
    # little bit too much machinery for what we're doing here, but we
    # already have the dependency and it's a one-liner, so...
    bg = np.ones(rgba_padded.shape[:2])
    bg[offset0:offset0 + size0, offset1:offset1 + size1] = 0
    d = scipy.ndimage.distance_transform_edt(bg)
    alpha = np.maximum(0, r - d) / r
    alpha = np.roll(alpha, s[0], axis=0)
    alpha = np.roll(alpha, s[1], axis=1)

    rgba_padded[..., 3] = np.maximum(rgba_padded[..., 3], alpha)

    return rgba_padded

hilbert_rgba_final = pad_image(hilbert_rgba_small)
show(hilbert_rgba_final)
# -

# ## Creating Image Files
#
# It's time for this experiment to leave our little sandbox, which means we'll want image files we can send around and upload.
# The Pillow package makes this pretty convenient.

# +
import os
import PIL

def save(rgba, filename):
    image = PIL.Image.fromarray((rgba[::-1]*255).astype(np.uint8))
    image.save(filename)

os.makedirs('build', exist_ok=True)
save(hilbert_rgba_small, 'build/favicon.ico')  # Icon format, for the website
save(hilbert_rgba_small, 'build/hilbert.png')  # PNG format, for general use
save(hilbert_rgba_final, 'build/avatar.png')   # for GitHub
# -

# ## Acknowledgements
#
# A lot of heavy lifting is being done by the colormap. The
# scientific programming community has done some great work in creating
# colormaps that are both visually appealing and avoid distortion as
# perceived by humans, and both of those qualities are valuable
# here. Matplotlib's excellent `plasma` colormap, which we've been using here,
# was created by [St&eacute;fan van der Walt and Nathaniel
# Smith](https://bids.github.io/colormap/).
