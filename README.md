This is the code I use to generate my website's favicon and my GitHub avatar.
To run it, create a Python3 `virtualenv` environment and install the
dependencies there:

- NumPy
- SciPy
- Pillow
- Matplotlib

Then run `python run.py` after activating the `virtualenv` environment.
The outputs will be generated under `./build`.

Alternatively, if you have `uv`, you can simply `uv run run.py`.

## What Is It?

It's a visualization of a Hilbert curve, a two-dimensional plane-filling
curve. More precisely, it's a continuous mapping from the unit interval
$[0, 1]$ to the unit square $[0, 1]\times [0, 1]$. To complete the
visualization, choose any one-dimensional colormap for the input
interval and push it forward to the square. Here, we've chosen Matplotlib's
`plasma` colormap. Finally, we do some image processing to enhance edges
and corners. That's really optional, but it makes the result more
visually interesting in my opinion.

## Acknowledgements

A lot of heavy lifting is being done by Matplotlib's excellent
`plasma` colormap, so credit is due to [St&eacute;fan van der Walt and
Nathaniel Smith](https://bids.github.io/colormap/).

## Copyright

Copyright Darsh Ranjan.

This software is released under the GNU General Public License,
version 3. See the file [COPYING](./COPYING) for details.
