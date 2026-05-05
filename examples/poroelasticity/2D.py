"""
This example implements a quasi-2D poroelastic flow simulation (Terzaghi problem) using LBM.

The flow is modeled by the Darcy equation.
"""
import argparse
from jax import config
import numpy as np
import os

from src.poroelasticity import LBMPoroelasticity
from src.darcy import AdvectionDiffusionReactionBGK, ADR_Neumann_BounceBack, ADR_AntiBounceBack
from src.elasticity import LBMElasticity, ElasticityBounceBackFullway, ElasticityAntiBounceBack
from src.lattice import LatticeD2Q8, LatticeD2Q9
from src.utils import *

# enable 64bit operations
config.update("jax_enable_x64", True)

class ADR_BB(ADR_Neumann_BounceBack):
    def get_missing_indices(self, boundaryMask):
        # Find imissing, iknown 1-to-1 corresponding indices
        # Note: the "zero" index is used as default value here and won't affect BC computations
        nbd = len(self.indices[0])
        imissing = np.vstack([np.arange(self.lattice.q, dtype='uint8')] * nbd)
        iknown = np.vstack([self.lattice.opp_indices] * nbd)

        for i in range(nbd):
            if self.indices[0][i] == 0 and self.indices[1][i] == 1: # Remove lower left corner
                boundaryMask[i, 3] = False
                boundaryMask[i, 5] = False
                print(f"ADR LL {boundaryMask[i]}")
            elif self.indices[0][i] == 0 and self.indices[1][i] == self.ny-2: # Remove upper left corner
                boundaryMask[i, 3] = False
                boundaryMask[i, 7] = False
                print(f"ADR UL {boundaryMask[i]}")
            elif self.indices[0][i] == self.nx-1 and self.indices[1][i] == 1: # Remove lower right corner
                boundaryMask[i, 6] = False
                boundaryMask[i, 8] = False
                print(f"ADR LR {boundaryMask[i]}")
            elif self.indices[0][i] == self.nx-1 and self.indices[1][i] == self.ny-2: # Remove upper right corner
                boundaryMask[i, 4] = False
                boundaryMask[i, 6] = False
                print(f"ADR UR {boundaryMask[i]}")

        imissing[~boundaryMask] = 0
        iknown[~boundaryMask] = 0
        return imissing, iknown


class ADR_AntiBB(ADR_AntiBounceBack):
    def get_missing_indices(self, boundaryMask):
        # Find imissing, iknown 1-to-1 corresponding indices
        # Note: the "zero" index is used as default value here and won't affect BC computations
        nbd = len(self.indices[0])
        imissing = np.vstack([np.arange(self.lattice.q, dtype='uint8')] * nbd)
        iknown = np.vstack([self.lattice.opp_indices] * nbd)

        for i in range(nbd):
            if self.indices[0][i] == 0 and self.indices[1][i] == 1: # Remove lower left corner
                boundaryMask[i, 3] = False
                boundaryMask[i, 5] = False
                print(f"ADR LL {boundaryMask[i]}")
            elif self.indices[0][i] == 0 and self.indices[1][i] == self.ny-2: # Remove upper left corner
                boundaryMask[i, 3] = False
                boundaryMask[i, 7] = False
                print(f"ADR UL {boundaryMask[i]}")
            elif self.indices[0][i] == self.nx-1 and self.indices[1][i] == 1: # Remove lower right corner
                boundaryMask[i, 6] = False
                boundaryMask[i, 8] = False
                print(f"ADR LR {boundaryMask[i]}")
            elif self.indices[0][i] == self.nx-1 and self.indices[1][i] == self.ny-2: # Remove upper right corner
                boundaryMask[i, 4] = False
                boundaryMask[i, 6] = False
                print(f"ADR UR {boundaryMask[i]}")

        imissing[~boundaryMask] = 0
        iknown[~boundaryMask] = 0
        return imissing, iknown


class ElasticityAntiBB(ElasticityAntiBounceBack):
    def get_missing_indices(self, boundaryMask):
        # Find imissing, iknown 1-to-1 corresponding indices
        # Note: the "zero" index is used as default value here and won't affect BC computations
        nbd = len(self.indices[0])
        imissing = np.vstack([np.arange(self.lattice.q, dtype='uint8')] * nbd)
        iknown = np.vstack([self.lattice.opp_indices] * nbd)

        for i in range(nbd):
            if self.indices[0][i] == 0 and self.indices[1][i] == 1: # Remove lower left corner
                boundaryMask[i, 0] = False
                boundaryMask[i, 7] = False
                print(f"EL LL {boundaryMask[i]}")
            elif self.indices[0][i] == 0 and self.indices[1][i] == self.ny-2: # Remove upper left corner
                boundaryMask[i, 0] = False
                boundaryMask[i, 4] = False
                print(f"EL UL {boundaryMask[i]}")
            elif self.indices[0][i] == self.nx-1 and self.indices[1][i] == 1: # Remove lower right corner
                boundaryMask[i, 2] = False
                boundaryMask[i, 6] = False
                print(f"EL LR {boundaryMask[i]}")
            elif self.indices[0][i] == self.nx-1 and self.indices[1][i] == self.ny-2: # Remove upper right corner
                boundaryMask[i, 2] = False
                boundaryMask[i, 5] = False
                print(f"EL UR {boundaryMask[i]}")

        imissing[~boundaryMask] = 0
        iknown[~boundaryMask] = 0
        return imissing, iknown


class Darcy(AdvectionDiffusionReactionBGK):
    def initialize_macroscopic_fields(self):
        p = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.output_dtype, init_val=0.0, sharding=self.sharding)
        vel = self.distributed_array_init((self.nx, self.ny, self.dim), self.precisionPolicy.output_dtype, init_val=0.0, sharding=self.sharding)
        return p, vel

    def set_boundary_conditions(self):
        # apply bounce back boundary condition to the bottom wall (no flow)
        self.BCs.append(ADR_BB(tuple(self.boundingBoxIndices["bottom"].T), self.gridInfo, self.precisionPolicy))
        # apply anti bounce back boundary condition to the top wall (no pressure)
        self.BCs.append(ADR_AntiBB(tuple(self.boundingBoxIndices["top"].T), self.gridInfo, self.precisionPolicy))


class Elasticity(LBMElasticity):
    def __init__(self, **kwargs):
        self._BC_force = -kwargs['parameters'].get_nondimensional('force')
        super().__init__(**kwargs)

    def initialize_macroscopic_fields(self):
        eta = self.distributed_array_init((self.nx, self.ny, self.dim), self.precisionPolicy.output_dtype, init_val=0.0, sharding=self.sharding)
        sigma = self.distributed_array_init((self.nx, self.ny, 3), self.precisionPolicy.output_dtype, init_val=0.0, sharding=self.sharding)
        return eta, sigma

    def set_boundary_conditions(self):
        # apply bounce back boundary condition to the bottom wall (no displacement)
        self.BCs.append(ElasticityBounceBackFullway(tuple(self.boundingBoxIndices["bottom"].T), self.gridInfo, self.precisionPolicy))
        # apply anti bounce back boundary condition to the top wall
        force = np.fromfunction(lambda x,d: (d==1) * (1-np.cos(np.pi * ((2*x+1) / self.nx + 0.5))) * self._BC_force, (self.nx,2))
        self.BCs.append(ElasticityAntiBB(tuple(self.boundingBoxIndices["top"].T), self.gridInfo, self.precisionPolicy, force))

        for level in range(1, self.levels):
            gridInfo = self.grid_info(level)
            bb_indices = self.bounding_box_indices(level)
            self.BCs_MG[level] = [
                ElasticityBounceBackFullway(tuple(bb_indices["bottom"].T), gridInfo, self.precisionPolicy),
                ElasticityAntiBB(tuple(bb_indices["top"].T), gridInfo, self.precisionPolicy)
            ]

class Poroelasticity(LBMPoroelasticity):
    def __init__(self, **kwargs):
        super().__init__(Elasticity, Darcy,**kwargs)
        self._output_dir = kwargs.get('output_dir')
        parameters = kwargs.get('parameters')
        self._deta = parameters.get_unit('H')
        self._dt = parameters.get_unit('T')
        self._dp = parameters.get_unit('P')

    def output_data(self, **kwargs):
        eta = np.array(kwargs["eta"])[0] * self._deta
        sigma = np.array(kwargs["sigma"])[0] * self._dp / self.epsilon
        pressure = np.array(kwargs["pressure"])[0] * self._dp
        timestep = kwargs["timestep"]

        fields = {"eta_x": eta[:, 1:-1, 0], "eta_y": eta[:, 1:-1, 1], "sigma_xx": sigma[:, 1:-1, 0],
                  "sigma_xy": sigma[:, 1:-1, 1], "sigma_yy": sigma[:, 1:-1, 2], "p": pressure[:, 1:-1, 0]}
        save_fields_vtk(timestep // self.ioRate, fields, self._output_dir)
        # save_BCs_vtk(timestep, self.BCs, self.gridInfo, self._output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2D Terzaghi-type poroelastic simulation using LBM".')
    parser.add_argument('-a', '--alpha', type=float, help="The Biot-Willis coefficient between 0 and 1 (default: 1).", default=1.0)
    parser.add_argument('-n', '--nx', type=int, help="The number of cells in x-direction (default: 20).", default=20)
    parser.add_argument('-e', '--elastic_steps', type=float, help="The relative number of elasticity steps per timestep (default: 2.0).", default=2.0)
    parser.add_argument('-l', '--levels', type=int, help="The number of multigrid levels for elasticity (default: 1).", default=1)
    parser.add_argument('-c', '--coupling', type=str, help="The coupling method (default: 'centered').", default='centered', choices=['explicit', 'implicit', 'centered'])
    parser.add_argument('-g', '--gradient', help="Whether to use the alternative gradient (default: False).", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    params = Parameters()
    params.add_units(L=1.0, H=1.0, T=1.0, P=1.0) # SI units
    params.add_parameters(
        L_tot  = (1e2,  {'L': 1}),                   # Total length
        E      = (1e5,  {'P': 1}),                   # Young's modulus
        nu     = (0.9,  {}),                         # Poisson's ratio
        c0     = (1e-6, {'P': -1}),                  # Constrained specific storage coefficient
        alpha  = (args.alpha, {}),                   # Biot-Willis param.
        force  = (1e4,  {'L': -1, 'H': 1, 'P': 1}),  # Stress applied as BC
        kappa  = (1e-9, {'L': 2, 'T': -1, 'P': -1}), # fluid diffusivity (permeability over viscosity)
        T_tot  = (2e6,  {'T': 1})                    # total time
    )
    # 2D conversions
    params.set_parameter('lambda', params.get('E') * params.get('nu') / (1 - params.get('nu')**2), {'P': 1}) # Lame's 1st param.
    params.set_parameter('mu', 0.5 * params.get('E') / (1 + params.get('nu')), {'P': 1}) # Lame's 2nd param. (shear modulus)

    print("### DIMENSIONAL " + '#'*44)
    params.print(nondimensional=False)

    output_dir = os.path.abspath("./results/poroelasticity/2D")
    os.makedirs(output_dir, exist_ok=True)

    nx = int(args.nx)
    print("\n### LBM NON-DIMENSIONAL " + '#'*36)
    params.add_units(L = params.get('L_tot') / nx, H = 1.0, T = params.get('T_tot') / nx**2, P = params.get('c0')**-1)
    params.print()
    nt = int(params.get_nondimensional('T_tot'))
    if nt % 100 != 0:
        nt_new = 100 * round(nt / 100)
        print(f"Adjusted total timesteps from {nt} to {nt_new} to match output rate.")
        print(f"Relative change: {((nt_new - nt) / nt)}")
        nt = nt_new

    precision = "f64/f64"
    levels = args.levels
    if levels < 1:
        levels = int(np.log2(nx))
        while nx % 2**(levels-1) != 0:
            levels -= 1
        ne = int(args.elastic_steps)
    else:
        ne = max(1, round(nx * args.elastic_steps))

    kwargs = {
        'lattice_elasticity': LatticeD2Q8(precision),
        'lattice_darcy': LatticeD2Q9(precision),
        'elasticity_steps': ne,
        'elasticity_MG_levels': levels,
        'MG_BCs': {"bottom": True, "top": True},
        'MG_n_smoothing': 4,
        'coupling_method': args.coupling,
        'alpha': params.get_nondimensional('alpha'),
        'lambda': params.get_nondimensional('lambda'),
        'mu': params.get_nondimensional('mu'),
        'epsilon': params.get_unit('L') / params.get_unit('H'),
        'omega': 1.0 / (3*params.get_nondimensional('kappa') + 0.5),
        'parameters': params,
        'nx': nx,
        'ny': nx+2, # for BC
        'nz': 0,
        'precision': precision,
        'io_rate': max(1, nt // 100),
        'print_info_rate': 1000,
        'output_dir': output_dir
    }
    sim = Poroelasticity(**kwargs)
    sim.run(nt)
