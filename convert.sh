#!/usr/bin/env bash

set -euxo pipefail

# Python Requirements: jupyterlab, jupytext, numpy, PIL, scipy, matplotlib,
#   jekyllnb, lxml[html_clean]

jupytext --to notebook --execute favicon.py
jupyter jekyllnb --site-dir build/site --page-dir generated --image-dir assets/generated favicon.ipynb
sed -f fix-md.sed -i build/site/generated/favicon.md
tar czvf build/site.tar.gz -C build/site .

set +x

echo "###"
echo "### To finish, run:"
echo "###"
echo "###    tar xzvf build/site.tar.gz -C \${jekyll_site_root}"
echo "###"
echo "### but make sure your tree is clean first, since files may be overwritten."
