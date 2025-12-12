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
fira_src="https://github.com/bBoxType/FiraSans/raw/refs/heads/master"

function get-fira {
    local name="$2"
    local slant="$3"
    local url
    local font
    if [[ "$slant" == "Italic" ]]; then
        if [[ "$name" == "Regular" ]]; then
            name="Italic"
        else
            name="${name}Italic"
        fi
    fi
    case "$1" in
        sans)
            url="${fira_src}/Fira_Sans_4_3/Fonts/Fira_Sans_WEB_4301/Normal/${slant}"
            font="FiraSans"
            ;;
        mono)
            url="${fira_src}/Fira_Mono_3_2/Fonts/FiraMono_WEB_32"
            font="FiraMono"
            ;;
        *)
            return 1
    esac
    curl -fLsS "${url}/${font}-${name}.woff" > "${fira_dst}/${font}-${name}.woff"
}

mkdir -p "${fira_dst}"
for weight in {Regular,Medium,SemiBold,Bold}; do
    get-fira sans $weight Roman
    get-fira sans $weight Italic
done

get-fira mono Regular Roman
get-fira mono Bold Roman

popd
