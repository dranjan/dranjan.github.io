#!/usr/bin/env bash

set -euxo pipefail

pushd nb
./convert.sh
popd

tar xzvf nb/build/site.tar.gz
