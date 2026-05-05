#!/bin/bash

function help() {
    echo "Usage: @0 [-c]"
    echo
    echo "Sets up a virtual python environment with the required packages"
    echo
    echo "Options:"
    echo "    -c, --cuda  Installs the cuda version for GPU instead of the CPU version"
    echo
}

function install() {
    if [ $# -eq 0 ]; then
        echo "Installing CPU version..."
        REQ="jax~=0.5.3"
    else
        echo "Installing GPU version..."
        REQ="jax[cuda12]~=0.5.3 jax-cuda12-plugin[with-cuda]~=0.5.3"
    fi
    python3 -m venv --upgrade-deps ./.venv
    source ./.venv/bin/activate
    pip3 install $REQ -r requirements.txt
}

# check for cuda
if [ $# -gt 2 ]; then
    echo "Error: Too many arguments: $@"
    echo
    help
    exit 1
elif [ $# -ne 1 ]; then
    install
else
    ARG=${1#-}   # remove first leading '-' if possible
    ARG=${ARG#-} # remove second leading '-' if possible
    case $ARG in
        h|help)
            help
            exit 0
            ;;
        c|cuda)
            install cuda
            ;;
        *)
            echo "Error: Unknown argument: $1"
            echo
            help
            exit 1
    esac
fi
