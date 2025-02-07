#!/usr/bin/env bash

set -euxo pipefail

PDM_IGNORE_ACTIVE_VENV=1 pdm install
