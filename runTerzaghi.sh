#!/bin/bash
. environment.sh
LC_NUMERIC=C
levels=-1
n0=4

if [[ $# -ne 0 ]]; then
    refine=3
    for alpha in 1.0 0.9 0.8; do
        echo "alpha = $alpha ..."
        for v in $(seq 1 1 6); do
            python3 examples/poroelasticity/Terzaghi.py -n $n0 -r $refine -l $levels -a $alpha -c explicit --no-plots -s -e $v >/dev/null &
            python3 examples/poroelasticity/Terzaghi.py -n $n0 -r $refine -l $levels -a $alpha -c implicit --no-plots -s -e $v >/dev/null &
            python3 examples/poroelasticity/Terzaghi.py -n $n0 -r $refine -l $levels -a $alpha -c centered --no-plots -s -e $v >/dev/null
        done
    done
fi

refine=5
for alpha in 1.0 0.9 0.8; do
    echo "alpha = $alpha ..."
    for v in $(seq 1 1 6); do
        python3 examples/poroelasticity/Terzaghi.py -n $n0 -r $refine -l $levels -a $alpha -c explicit --no-plots -s -e $v
        python3 examples/poroelasticity/Terzaghi.py -n $n0 -r $refine -l $levels -a $alpha -c implicit --no-plots -s -e $v
        python3 examples/poroelasticity/Terzaghi.py -n $n0 -r $refine -l $levels -a $alpha -c centered --no-plots -s -e $v
    done
done