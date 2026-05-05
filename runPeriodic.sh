#!/bin/bash
. environment.sh
LC_NUMERIC=C
n0=10
alpha=0.5

if [[ $# -ne 0 ]]; then
	refine=4
	for v in $(seq 0.1 0.1 1.5); do
		echo "N_E = $v ..."
		python3 examples/poroelasticity/periodic.py -n $n0 -r $refine -a $alpha -c explicit --no-plots -s -e $v >/dev/null &
		python3 examples/poroelasticity/periodic.py -n $n0 -r $refine -a $alpha -c implicit --no-plots -s -e $v >/dev/null &
		python3 examples/poroelasticity/periodic.py -n $n0 -r $refine -a $alpha -c centered --no-plots -s -e $v >/dev/null
	done
fi

refine=9
for v in $(seq 0.1 0.1 1.5); do
	python3 examples/poroelasticity/periodic.py -n $n0 -r $refine -a $alpha -c explicit --no-plots -s -e $v
	python3 examples/poroelasticity/periodic.py -n $n0 -r $refine -a $alpha -c implicit --no-plots -s -e $v
	python3 examples/poroelasticity/periodic.py -n $n0 -r $refine -a $alpha -c centered --no-plots -s -e $v
done
