<div align="center">
    <img src="https://github.com/dranjan.png">
</div>

This is the code I use to generate my website's favicon and my GitHub
avatar. To run it, create a Python3 `virtualenv` environment and
install the dependencies there:

- NumPy
- SciPy
- Pillow
- Matplotlib

Then run `python hilbert.py` after activating the `virtualenv` environment.
The outputs will be generated under `./build`.

Alternatively, if you have `uv`, you can simply `uv run hilbert.py`.

## What Is It?

It's a visualization of a Hilbert curve. A Hilbert curve is a
two-dimensional plane-filling curve, or more precisely a continuous
surjective mapping from the unit interval $[0, 1]$ to the unit square
$[0, 1]\times [0, 1]$. The mapping isn't injective, but it's
approximated arbitrarily well by injective mappings.

To make the visualization, we choose a one-dimensional colormap for the
input interval and push it forward with the surjective mapping to color
the square. As I said earlier, the mapping isn't injective, but that's
fine, since we can use the injective approximations I mentioned. That
works particularly well if our square image has a power-of-two number of
pixels, because that means we can conveniently make the injective
approximation visit each pixel exactly once.

Finally, we do some image processing to enhance edges and corners.
That's really optional, and whether this step improves the result or not
is a matter of personal taste. I find myself going back and
forth on it, and I can make both an aesthetic and mathematical case for
either version. Ultimately, I keep the extra postprocessing because:

1. it makes it easier to see how the interval snakes its way across the
   square, and
2. it makes the fractal structure easier to see.

Since website favicons are typically rendered at very small sizes, it
hardly makes any difference there at all, but for simplicity, I try to
use the same version everywhere.

## Acknowledgements

A lot of heavy lifting is being done by the colormap. The
scientific Python community has done some great work in creating
colormaps that are both visually appealing and avoid distortion as
perceived visually by humans, and both of those qualities are valuable
here. The one I've selected here is Matplotlib's `plasma`, which
was created by [St&eacute;fan van der Walt and Nathaniel
Smith](https://bids.github.io/colormap/).

## Copyright

Copyright Darsh Ranjan.

This software is released under the GNU General Public License,
version 3. See the file [COPYING](./COPYING) for details.
