## Initial setup

Install PDM (say, via `pipx`) and execute `./init.sh` in this directory.
This should install all Python dependencies in a local virtualenv
environment.

## Running It

After checking out the repository and setting up as described above,
run `_scripts/generate-nb.sh` in the repository root.
This will execute and process the Jupyter note book sources into a form
that can be consumed by Jekyll and ultimately published on the site.

The notebooks are maintained primarily as plain text (Markdown),
but they can also be edited in the more conventional `.ipynb` form.
The latter are ignored in Git and must be generated
from the `.md` sources, and resynced to them in order to track changes.

A primary annoyance, and the reason for a lot of the complexity in this workflow,
is that Jupyter and Jekyll markdown are fundamentally incompatible, and in
general Python tooling seems to have quite poor support for the latter.

To summarize what `generate-nb.sh` does internally:

1. Preprocess the Markdown sources to be more compatible with Jekyll.
   (The human-facing Markdown follows Jupyter syntax for more convenient
   authoring.)
2. Recreate the `.ipynb` notebooks from the preprocessed `.md` sources.
   (The Jekyll-specific stuff from the previous step
   should pass through this step unmolested.)
2. Convert each `.ipynb` to its final Jekyll markdown form.

## Notebook Quirks

The following rules must be followed in order to work seamlessly with both Jupyter and Jekyll.
As mentioned above, these two systems aren't fully compatible, so there's a hand-rolled system
here to bridge the gap.

### Notebook Metadata

If the notebook contains any LaTeX formulas in Markdown cells, then the `jekyll` table in
the notebook's metadata must contain the field `katex: true`. This is detected and handled
appropriately by the rest of the tooling. The only purpose of this logic is to avoid loading
KaTeX assets for the pages that don't need them.

### LaTeX Formulas

One of the main incompatibilities between Jupyter and Jekyll Markdown is the syntax
for embedded LaTeX formulas. The Markdown sources here should always follow the
Jupyter syntax conventions, which is `$...$` for inline math and

    $$
    ...
    $$

for display math. The Markdown preprocessing will parse those out and convert them to the
Jekyll syntax.

### Raw Markdown

The preprocessing, which uses `mdformat`, is a bit finicky,
and sometimes it's necessary to force it
to pass a small bit of Markdown syntax through unchanged,
like Kramdown directives which are very sensitive to positioning and spacing.
This can be done with a pseudo-HTML `<markdown>` block:

    <markdown>
    (... anything goes ...)
    </markdown>

Note that these will not render correctly in the Jupyter interface,
but if used sparingly, it's not too big of a deal.

### Table of Contents

To add and configure a table of contents, use `<markdown>` blocks as described above.
Put the block

    <markdown>
    - TOC
    {:toc}
    </markdown>

where the table of contents should go.
For a heading that should be omitted from the table, also use a `<markdown>` block:

    <markdown>
    # My Title
    {: .no_toc}
    </markdown>
