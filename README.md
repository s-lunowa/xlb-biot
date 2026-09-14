# Linear Poroelastics Extension for XLB

This repository was **forked from [XLB](https://github.com/Autodesk/XLB)** and extends the framework
by lattice Boltzmann methods (LBM) to solve **(static) linear elasticity, diffusive flow and linear
poroelasticity**. Additionally, it implements a **multigrid** Lattice Boltzmann method for linear
elasticity to accelerate the (elliptic) pseudo-timestepping.

This repository belongs to the following article:

Stephan B. Lunowa and Barbara Wohlmuth,
*A lattice Boltzmann method for Biot's consolidation model of linear poroelasticity*,
Journal of Computational Physics 564 (2026), 115176, [doi: 10.1016/j.jcp.2026.115176](https://doi.org/10.1016/j.jcp.2026.115176).
Preprint: [arXiv:2409.11382](https://arxiv.org/abs/2409.11382)


## Overview of Changes

The existing XLB code is changed at the following places:

* The original README can be found in [XLB_README.md](./XLB_README.md)
* The [.gitignore](./.gitignore) also contains the output directory `results/`.
* The python [requirements.txt](./requirements.txt) are adapted to currently compatible versions,
  and an virtual environment with [setup](./setup.sh) allowing for CPU and GPU version.
* The lattice class in [src/lattice.py](./src/lattice.py) is extended to allow for a D2Q8 lattice
  which has no rest velocity (used for linear elasticity).
* The utility module in [src/utils.py](./src/utils.py) is extended by a class `Parameters`
  for simpler unit conversions.

The new base LBM solvers are located in `src`:

* `src/darcy.py` implements the diffusive flow solver
* `src/elasticity.py` implements the linear elasticity solver
* `src/poroelasticity.py` implements the linear poroelasticity solver

The newly implemented examples are located in `examples`:

* `examples/darcy/` contains diffusive flow examples.
* `examples/elasticity/` contains linear elasticity examples.
* `examples/poroelasticity/` contains linear poroelasticity examples.

The results are saved in subdirectories of `results/`.

Note that a full documentation is not yet available, but most functions include detailed **docstrings**
explaining their purpose and parameters.


## Numerical Experiments

The examples also contain several validation examples and analysis scripts.
For the poroelasticity examples, running all simulations reported in the article
is possible using the following scripts

* `runPeriodic.sh`: Periodic example without multigrid
* `runPeriodicMG.sh`: Periodic example with multigrid
* `runPeriodicMG_altgrad.sh`: Periodic example with multigrid and alternative pressure gradient (eq. 30)
* `runTerzaghi.sh`: Terzaghi consolidation with multigrid
* `runTerzaghi2D.sh`: 2D Terzaghi-type consolidation with multigrid


## Setup

To set up the python virtual environment run:

`setup.sh`

If you want to use the GPU instead of CPU, run:

`setup.sh --cuda`

After this, you can activate the python virtual environment via:

`. environment.sh`
