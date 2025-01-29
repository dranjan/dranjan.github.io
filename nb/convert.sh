#!/usr/bin/env bash

set -euxo pipefail

# Python Requirements: jupyterlab, jupytext, numpy, PIL, scipy, matplotlib,
#   jekyllnb, lxml[html_clean]

jupytext --to notebook --execute Favicon.py
jupyter jekyllnb --site-dir tmp Favicon.ipynb
sed -i 's/<!--begin:mathinline-->\$/\$\$/g; s/\$<!--end:mathinline-->/\$\$/g' tmp/Favicon.md
