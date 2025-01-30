This is really kludgy. I hope to streamline a lot of this, but for now
here's a summary of how to work with this material.

The notebooks are intended for publication on my website, but the path
to getting them there is fairly wiggly.

## Prerequisites

To begin, the following Python dependencies are required:

- jupyter
- jupytext
- jekyllnb
- numpy
- scipy
- matplotlib
- pillow

Those should be installed in a Virtualenv.

TODO: handle that with proper tooling.

## Running It

The notebooks are maintained primarily as Python scripts, but they can also be edited in
the more conventional `.ipynb` form. The latter are ignored in Git and must be generated
from the `.py` sources, and resynced to them in order to track changes.

After checking out the repository, run `./convert.sh`. This will do a few things:

1. Recreate the `.ipynb` notebooks from the `.py` sources.
2. Convert each `.ipynb` to its Jekyll markdown form in `build/site`.
3. Process hand-rolled directives to fix things for Jekyll, since
   Jupyter and Jekyll markdown aren't 100% compatible.
4. Make a nice tarball of everything that can be overlayed on top of the Jekyll site.

When that completes successfully, follow the printed instructions to update the
Jekyll site.

TODO: merge this into the main repository.
TODO: configure the Github actions to run all of this transparently.

## Notebook Quirks

The following rules must be followed in order to work seamlessly with both Jupyter and Jekyll.
As mentioned above, these two systems aren't fully compatible, so there's a hand-rolled system
here to bridge the gap. The syntax is pretty brittle, but it's sufficient for now.

### Notebook Metadata

If the notebook contains any LaTeX formulas in Markdown cells, then the `jekyll` table in
the notebook's metadata must contain the field `katex: true`. This is detected and handled
appropriately in the main repository. The only purpose of this logic is to avoid loading
KaTeX assets for the pages that don't need them.

### Inline Math

Inline math must be written exactly like this:

    <!--begin:mathinline-->$PUT MATH HERE$<!--end:mathinline-->

The HTML comments are ignored in JupyterLab, but this repository's tooling will find them
and convert them into the proper syntax that is understood by Jekyll and KaTeX.

TODO: look into Pandoc as an alternative to Kramdown, which may be more
compatible with Jupyter.

### Display Math

Display math must be in separate paragraphs:

    Some text here

    $$
    display math here
    $$

    More normal text here

### Table of Contents

Put the HTML comment

    <!--insert:toc-->

where the table of contents should go. For a heading that should not appear in the table,
annotate it like this:

    # Some Heading <!--insert:no-toc-->
