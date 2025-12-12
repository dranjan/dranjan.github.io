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
    local slant
    local name
    local url
    local font
    name="$(get-weight-name "$2")"
    if [[ "$3" == "i" ]]; then
        slant=Italic
        if [[ "$name" == "Regular" ]]; then
            name="Italic"
        else
            name="${name}Italic"
        fi
    else
        slant=Roman
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

function get-weight-name {
    case "$1" in
        100) echo Thin ;;
        200) echo ExtraLight ;;
        300) echo Light ;;
        400) echo Regular ;;
        500) echo Medium ;;
        600) echo SemiBold ;;
        700) echo Bold ;;
        800) echo ExtraBold ;;
        900) echo Heavy ;;
        *) return 1
    esac
}

mkdir -p "${fira_dst}"
get-fira sans 400 r
get-fira sans 400 i
get-fira sans 500 r
get-fira sans 500 i
get-fira sans 600 r
get-fira sans 600 i
get-fira sans 700 r
get-fira sans 700 i
get-fira mono 400 r
get-fira mono 700 r

popd
