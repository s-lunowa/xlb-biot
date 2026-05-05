#!/bin/bash
. environment.sh
LC_NUMERIC=C
python3 examples/poroelasticity/2D.py -n 128 -l -1 -e 3
