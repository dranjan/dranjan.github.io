#!/usr/bin/env bash

set -euxo pipefail

pushd _nb
./convert.sh
popd

tar xzvf _nb/build/site.tar.gz
