#!/usr/bin/env bash

set -euxo pipefail

pushd $(dirname "${BASH_SOURCE[0]}")/..

# KaTeX
katex_src=$(bundle exec ruby _scripts/katex_path.rb)
katex_dst="assets/vendor/katex"
mkdir -p "${katex_dst}"
mkdir -p "${katex_dst}/fonts"
cp "${katex_src}/stylesheets/katex.css" "${katex_dst}/katex.css"
cp "${katex_src}/fonts"/*.woff2 "${katex_dst}/fonts/"

# Fira Sans and Fira Mono
fira_dst="assets/vendor/fira/fonts"
fira_src="https://free.bboxtype.com/embedfonts/fonts.php"

function get-fira-sans {
    local weight="$1"
    local name="$2"
    curl -fLsS "${fira_src}?family=FiraSans&weight=${weight}" > "${fira_dst}/FiraSans-${name}.woff"
}

function get-fira-mono {
    local weight="$1"
    local name="$2"
    curl -fLsS "${fira_src}?family=FiraMono&weight=${weight}" > "${fira_dst}/FiraMono-${name}.woff"
}


mkdir -p "${fira_dst}"
get-fira-sans 400 Regular
get-fira-sans 400i Italic
get-fira-sans 500 Medium
get-fira-sans 500i MediumItalic
get-fira-sans 600 SemiBold
get-fira-sans 600i SemiBoldItalic
get-fira-sans 700 Bold
get-fira-sans 700i BoldItalic
get-fira-mono 400 Regular
get-fira-mono 700 Bold

popd
