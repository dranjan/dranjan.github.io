#!/usr/bin/env bash

set -euxo pipefail

pushd $(dirname "${BASH_SOURCE[0]}")/..

export PDM_IGNORE_ACTIVE_VENV=1
mkdir -p _nb/build
cp _nb/favicon.md _nb/build/
pdm run -p _nb mdformat --extensions frontmatter --extensions jekyll_math --no-validate _nb/build/favicon.md
pdm run -p _nb jupytext --to notebook --execute _nb/build/favicon.md
pdm run -p _nb jupyter jekyllnb --site-dir . --page-dir generated --image-dir assets/generated _nb/build/favicon.ipynb

popd
