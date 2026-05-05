#!/bin/bash
. environment.sh
LC_NUMERIC=C
levels=-1
n0=4

if [[ $# -ne 0 ]]; then
	refine=3
	for alpha in $(seq 0.3 0.1 1.0); do
		echo "alpha = $alpha ..."
		for v in $(seq 1 1 6); do
			python3 examples/poroelasticity/periodic.py -n $n0 -l $levels -r $refine -a $alpha -c explicit --no-plots -s -e $v >/dev/null &
			python3 examples/poroelasticity/periodic.py -n $n0 -l $levels -r $refine -a $alpha -c implicit --no-plots -s -e $v >/dev/null &
			python3 examples/poroelasticity/periodic.py -n $n0 -l $levels -r $refine -a $alpha -c centered --no-plots -s -e $v >/dev/null
		done
	done
fi

refine=5
for alpha in $(seq 0.3 0.1 1.0); do
	echo "alpha = $alpha ..."
	for v in $(seq 1 1 6); do
		if [[ $alpha < 0.8 ]]; then
			python3 examples/poroelasticity/periodic.py -n $n0 -l $levels -r $refine -a $alpha -c explicit --no-plots -s -e $v
		fi
		if [[ $alpha < 0.9 ]]; then
			python3 examples/poroelasticity/periodic.py -n $n0 -l $levels -r $refine -a $alpha -c implicit --no-plots -s -e $v
		fi
		python3 examples/poroelasticity/periodic.py -n $n0 -l $levels -r $refine -a $alpha -c centered --no-plots -s -e $v
	done
done