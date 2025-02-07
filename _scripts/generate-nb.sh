#!/usr/bin/env bash

set -euxo pipefail

pushd $(dirname "${BASH_SOURCE[0]}")/..

export PDM_IGNORE_ACTIVE_VENV=1
pdm run -p _nb jupytext --to notebook --execute _nb/favicon.py
pdm run -p _nb jupyter jekyllnb --site-dir . --page-dir generated --image-dir assets/generated _nb/favicon.ipynb
sed -f _scripts/fix-md.sed -i generated/favicon.md

popd
