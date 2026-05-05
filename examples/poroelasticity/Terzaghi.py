"""
This example implements a quasi-2D poroelastic flow simulation (Terzaghi problem) using LBM.

The flow is modeled by the Darcy equation.
"""
import argparse
from jax import config
import numpy as np
from matplotlib import pyplot
from tikzplotlib import save as saveTIKZ
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
            if self.indices[0][i] == 1 and self.indices[1][i] == 0: # Remove lower left corner
                boundaryMask[i, 1] = False
                boundaryMask[i, 4] = False
                print(f"ADR LL {boundaryMask[i]}")
            elif self.indices[0][i] == 1 and self.indices[1][i] == self.ny-1: # Remove upper left corner
                boundaryMask[i, 2] = False
                boundaryMask[i, 8] = False
                print(f"ADR UL {boundaryMask[i]}")
            elif self.indices[0][i] == self.nx-2 and self.indices[1][i] == 0: # Remove lower right corner
                boundaryMask[i, 1] = False
                boundaryMask[i, 7] = False
                print(f"ADR LR {boundaryMask[i]}")
            elif self.indices[0][i] == self.nx-2 and self.indices[1][i] == self.ny-1: # Remove upper right corner
                boundaryMask[i, 2] = False
                boundaryMask[i, 5] = False
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
            if self.indices[0][i] == 1 and self.indices[1][i] == 0: # Remove lower left corner
                boundaryMask[i, 1] = False
                boundaryMask[i, 4] = False
                print(f"ADR LL {boundaryMask[i]}")
            elif self.indices[0][i] == 1 and self.indices[1][i] == self.ny-1: # Remove upper left corner
                boundaryMask[i, 2] = False
                boundaryMask[i, 8] = False
                print(f"ADR UL {boundaryMask[i]}")
            elif self.indices[0][i] == self.nx-2 and self.indices[1][i] == 0: # Remove lower right corner
                boundaryMask[i, 1] = False
                boundaryMask[i, 7] = False
                print(f"ADR LR {boundaryMask[i]}")
            elif self.indices[0][i] == self.nx-2 and self.indices[1][i] == self.ny-1: # Remove upper right corner
                boundaryMask[i, 2] = False
                boundaryMask[i, 5] = False
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
            if self.indices[0][i] == 1 and self.indices[1][i] == 0: # Remove lower left corner
                boundaryMask[i, 1] = False
                boundaryMask[i, 5] = False
                print(f"EL LL {boundaryMask[i]}")
            elif self.indices[0][i] == 1 and self.indices[1][i] == self.ny-1: # Remove upper left corner
                boundaryMask[i, 3] = False
                boundaryMask[i, 6] = False
                print(f"EL UL {boundaryMask[i]}")
            elif self.indices[0][i] == self.nx-2 and self.indices[1][i] == 0: # Remove lower right corner
                boundaryMask[i, 1] = False
                boundaryMask[i, 4] = False
                print(f"EL LR {boundaryMask[i]}")
            elif self.indices[0][i] == self.nx-2 and self.indices[1][i] == self.ny-1: # Remove upper right corner
                boundaryMask[i, 3] = False
                boundaryMask[i, 7] = False
                print(f"EL UR {boundaryMask[i]}")

        imissing[~boundaryMask] = 0
        iknown[~boundaryMask] = 0
        return imissing, iknown


class TerzaghiDarcy(AdvectionDiffusionReactionBGK):
    def initialize_macroscopic_fields(self):
        p = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.output_dtype, init_val=0.0, sharding=self.sharding)
        vel = self.distributed_array_init((self.nx, self.ny, self.dim), self.precisionPolicy.output_dtype, init_val=0.0, sharding=self.sharding)
        return p, vel

    def set_boundary_conditions(self):
        # apply bounce back boundary condition to the left wall (no flow)
        self.BCs.append(ADR_BB(tuple(self.boundingBoxIndices["left"].T), self.gridInfo, self.precisionPolicy))
        # apply anti bounce back boundary condition to the right wall (no pressure)
        self.BCs.append(ADR_AntiBB(tuple(self.boundingBoxIndices["right"].T), self.gridInfo, self.precisionPolicy))


class TerzaghiElasticity(LBMElasticity):
    def __init__(self, **kwargs):
        self._BC_force = -kwargs['parameters'].get_nondimensional('force')
        super().__init__(**kwargs)

    def initialize_macroscopic_fields(self):
        eta = self.distributed_array_init((self.nx, self.ny, self.dim), self.precisionPolicy.output_dtype, init_val=0.0, sharding=self.sharding)
        sigma = self.distributed_array_init((self.nx, self.ny, 3), self.precisionPolicy.output_dtype, init_val=0.0, sharding=self.sharding)
        return eta, sigma

    def set_boundary_conditions(self):
        # apply bounce back boundary condition to the left wall (no displacement)
        self.BCs.append(ElasticityBounceBackFullway(tuple(self.boundingBoxIndices["left"].T), self.gridInfo, self.precisionPolicy))
        # apply anti bounce back boundary condition to the right wall
        force = np.stack((np.full((self.ny,), self._BC_force), np.zeros((self.ny,))), axis=-1)
        self.BCs.append(ElasticityAntiBB(tuple(self.boundingBoxIndices["right"].T), self.gridInfo, self.precisionPolicy, force))

        for level in range(1, self.levels):
            gridInfo = self.grid_info(level)
            bb_indices = self.bounding_box_indices(level)
            self.BCs_MG[level] = [
                ElasticityBounceBackFullway(tuple(bb_indices["left"].T), gridInfo, self.precisionPolicy),
                ElasticityAntiBB(tuple(bb_indices["right"].T), gridInfo, self.precisionPolicy)
            ]

class TerzaghiPoroelasticity(LBMPoroelasticity):
    def __init__(self, **kwargs):
        super().__init__(TerzaghiElasticity, TerzaghiDarcy,**kwargs)
        self._output_dir = kwargs.get('output_dir')
        self._save_vtk = kwargs.get('save_vtk')
        parameters = kwargs.get('parameters')
        self._N_exact = kwargs.get('N_exact', 100) + 1
        self._deta = parameters.get_unit('H')
        self._dx = parameters.get_unit('L')
        self._dt = parameters.get_unit('T')
        self._dp = parameters.get_unit('P')
        self._etaInf = parameters.get_nondimensional('etaInf')

    def pressure(self, timestep):
        if timestep > 0:
            time_coeff = -0.25 * np.pi**2 * timestep * self._dt
            space_coeff = 0.5 * np.pi * self._dx
            summand = lambda x, j: (-1)**j / (2*j + 1) * np.cos((2*j + 1) * space_coeff * (x - 0.5)) * np.exp((2*j + 1)**2 * time_coeff)
            coeff = np.vectorize( lambda x: 4.0 / np.pi * np.sum(summand(x, np.arange(self._N_exact))) )
            return np.fromfunction(lambda x,*_: coeff(x), (self.darcy.nx,self.darcy.ny,1))
        else:
            return np.ones((self.darcy.nx,self.darcy.ny,1))

    def subsidence(self, timestep):
        if timestep > 0:
            time_coeff = -0.25 * np.pi**2 * timestep * self._dt
            summand = lambda j: np.exp((2*j + 1)**2 * time_coeff) / (2*j + 1)**2
            eta = self._etaInf + (1.0 - self._etaInf) * 8.0 / np.pi**2 * np.sum(summand(np.arange(self._N_exact)))
        else:
            eta = 1.0
        return eta

    def run(self, t_max):
        self._l2_subsidence = np.empty((t_max // self.ioRate + 1,))
        self._err_l2_sub    = np.empty_like(self._l2_subsidence)
        self._l2_pressure   = np.empty_like(self._l2_subsidence)
        self._err_l2_pres   = np.empty_like(self._l2_subsidence)

        super().run(t_max)

    def output_data(self, **kwargs):
        eta = np.array(kwargs["eta"])[0]
        sigma = np.array(kwargs["sigma"])[0]
        pressure = np.array(kwargs["pressure"])[0]
        timestep = kwargs["timestep"]

        subsidence = self.subsidence(timestep)
        self._l2_subsidence[timestep // self.ioRate] = subsidence
        subsidence_diff = np.mean(eta[-2, :, 0]) + subsidence # subsidence has opposite sign
        self._err_l2_sub[timestep // self.ioRate] = subsidence_diff

        p_exact = self.pressure(timestep)
        self._l2_pressure[timestep // self.ioRate] = np.linalg.norm(p_exact[1:-1])
        p_diff = pressure - p_exact
        self._err_l2_pres[timestep // self.ioRate] = np.linalg.norm(p_diff[1:-1])

        output_times = [1e-4, 2.5e-4, 1e-3, 2.5e-3, 1e-2, 2.5e-2, 0.1, 0.25, 0.5, 0.75, 1.0]
        for t in output_times:
             if (timestep + 0.5*self.ioRate) * self._dt >= t and (timestep - 0.5*self.ioRate) * self._dt < t:
                print(f"### Time {timestep * self._dt} ###")
                for t,p in zip(np.arange(self._dx/2, 1.0, self._dx), np.mean(pressure[1:-1], axis=1)[:, 0]):
                    print(f"{t} {p}")
                print("", flush=True)
                break

        if self._save_vtk:
            eta = eta * self._deta
            sigma = sigma * self._dp / self.epsilon
            pressure = pressure * self._dp
            p_exact = p_exact * self._dp
            p_diff = p_diff * self._dp
            fields = {"eta_x": eta[..., 0], "eta_y": eta[..., 1], "sigma_xx": sigma[..., 0], "sigma_xy": sigma[..., 1], "sigma_yy": sigma[..., 2],
                      "p": pressure[..., 0], "p_ex": p_exact[..., 0], "diff_p": p_diff[..., 0]}
            save_fields_vtk(timestep // self.ioRate, fields, self._output_dir)
            # save_BCs_vtk(timestep, self.BCs, self.gridInfo, self._output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quasi-2D Terzaghi poroelastic simulation using LBM".')
    parser.add_argument('-a', '--alpha', type=float, help="The Biot-Willis coefficient between 0 and 1 (default: 1).", default=1.0)
    parser.add_argument('-n', '--nx', type=int, help="The number of cells in x-direction (default: 20).", default=20)
    parser.add_argument('-e', '--elastic_steps', type=float, help="The relative number of elasticity steps per timestep (default: 1.0).", default=1.0)
    parser.add_argument('-l', '--levels', type=int, help="The number of multigrid levels for elasticity (default: 1).", default=1)
    parser.add_argument('-r', '--refinements', type=int, help="The number of refinements for convergence study (default: 0).", default=0)
    parser.add_argument('-c', '--coupling', type=str, help="The coupling method (default: 'centered').", default='centered', choices=['explicit', 'implicit', 'centered'])
    parser.add_argument('-g', '--gradient', help="Whether to use the alternative gradient (default: False).", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-p', '--plots', help="Whether to show individual convergence plots (default: True).", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-s', '--save_fig', help="Whether to save the plots as figures (default: False).", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-v', '--vtk', help="Whether to save vtk files (default: False).", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    precision = "f64/f64"
    lattice_elasticity = LatticeD2Q8(precision)
    lattice_darcy = LatticeD2Q9(precision)

    params = Parameters()
    params.add_units(L=1.0, H=1.0, T=1.0, P=1.0) # SI units
    params.add_parameters(
        L_tot   = (1e0, {'L': 1}),          # Total length
        E       = (1.0, {'P': 1}),          # Young's modulus
        nu      = (0.8, {}),                # Poisson's ratio
        c0      = (1e0, {'P': -1}),         # Constrained specific storage coefficient
        alpha   = (args.alpha, {})          # Biot-Willis param.
    )
    # 2D conversions
    params.set_parameter('lambda', params.get('E') * params.get('nu') / (1 - params.get('nu')**2), {'P': 1}) # Lame's 1st param.
    params.set_parameter('mu', 0.5 * params.get('E') / (1 + params.get('nu')), {'P': 1}) # Lame's 2nd param. (shear modulus)
    # further parameters
    params.add_parameters(
        force  = (params.get('alpha') / params.get('c0') + (params.get('lambda') + 2 * params.get('mu')) / params.get('alpha'), {'L': -1, 'H': 1, 'P': 1}), # Stress applied as BC
        Kpm    = (params.get('lambda') + 2 * params.get('mu') + params.get('alpha')**2 / params.get('c0'), {'P': 1}), # undrainded bulk modulus plus shear modulus
    )
    params.add_parameters(
        p0     = (params.get('alpha') * params.get('force') / params.get('c0') / params.get('Kpm'),          {'P': 1}),   # initial pressure
        eta0   = (params.get('force') * params.get('L_tot') / params.get('Kpm'),                             {'H': 1}),   # initial displacement (on the right)
        etaInf = (params.get('force') * params.get('L_tot') / (params.get('lambda') + 2 * params.get('mu')), {'H': 1}),   # final displacement (on the right)
        kappa  = (params.get('Kpm') / (params.get('lambda') + 2 * params.get('mu')), {'L': 2, 'T': -1, 'P': -1}) # fluid diffusivity (permeability over viscosity)
    )
    params.set_parameter('c_f', params.get('kappa') * (params.get('lambda') + 2 * params.get('mu')) / params.get('c0') / params.get('Kpm'), {'L': 2, 'T': -1}) # effective diffusion
    params.set_parameter('T_tot', params.get('L_tot')**2 / params.get('c_f'), {'T': 1}) # total time

    print("### DIMENSIONAL " + '#'*44)
    params.print(nondimensional=False)

    # NON-DIMENSIONAL
    params.add_units(L = params.get('L_tot'), H = params.get('eta0'), T = params.get('T_tot'), P = params.get('p0'))
    for k,v in {'L_tot': 1.0, 'c0': 1.0, 'p0': 1.0, 'eta0': 1.0}.items():
        if(np.abs(params.get_nondimensional(k) - v) > 1e-6 * np.abs(v)):
            raise RuntimeError(f"Wrong conversion of parameter '{k}'.")

    output_dir = os.path.abspath("./results/poroelasticity/Terzaghi")
    os.makedirs(output_dir, exist_ok=True)

    if args.levels < 0:
        nx_list = np.array([args.nx * 2**r for r in range(args.refinements+1)])
    else:
        nx_list = np.array([args.nx * (r+1) for r in range(args.refinements+1)])
    err_sub = np.empty_like(nx_list, dtype=float)
    err_pres = np.empty_like(nx_list, dtype=float)

    for i, nx in enumerate(nx_list):
        print("\n### LBM NON-DIMENSIONAL " + '#'*36)
        beta = 0.25
        params.add_units(L = params.get('L_tot') / nx, H = params.get('eta0'), T = params.get('T_tot') * beta / nx**2, P = params.get('p0'))
        params.print()

        nx = int(nx)
        ny = 4
        nt = int(params.get_nondimensional('T_tot'))
        levels = args.levels
        if levels < 1:
            levels = int(np.log2(nx))
            while nx % 2**(levels-1) != 0:
                levels -= 1
            ny = nx
            ne = int(args.elastic_steps)
        else:
            ne = max(1, round(nx * args.elastic_steps))
        coupling = args.coupling
        IO_rate = max(1, nt // 400) if args.vtk else 1

        filename = os.path.join(output_dir, f"error_a{args.alpha:.3f}_nx{nx:03d}_ne{ne:03d}_l{levels:03d}_{coupling}{'_altgrad' if args.gradient else ''}.npz")

        run_simulation = True
        if not args.vtk:
            try:
                data = np.load(filename)
                run_simulation = False
            except:
                print(f'Could not read file "{filename}". Run simulation...')
        if run_simulation:
            kwargs = {
                'lattice_elasticity': lattice_elasticity,
                'lattice_darcy': lattice_darcy,
                'elasticity_steps': ne,
                'elasticity_MG_levels': levels,
                'MG_BCs': {"left": True, "right": True},
                'MG_n_smoothing': 4,
                'coupling_method': coupling,
                'alpha': params.get_nondimensional('alpha'),
                'lambda': params.get_nondimensional('lambda'),
                'mu': params.get_nondimensional('mu'),
                'epsilon': params.get_unit('L') / params.get_unit('H'),
                'omega': 1.0 / (3*params.get_nondimensional('kappa') + 0.5),
                'parameters': params,
                'nx': nx+2, # for BC
                'ny': ny,
                'nz': 0,
                'precision': precision,
                'io_rate': IO_rate,
                'save_vtk': args.vtk,
                'print_info_rate': 1000,
                'output_dir': output_dir
            }
            sim = TerzaghiPoroelasticity(**kwargs)
            sim.run(nt)
            data = { 'err_l2_sub': sim._err_l2_sub, 'err_l2_pres': sim._err_l2_pres,
                     'rel_err_sub': np.sqrt(np.sum(sim._err_l2_sub**2) / np.sum(sim._l2_subsidence**2)),
                     'rel_err_pres': np.sqrt(np.sum(sim._err_l2_pres**2) / np.sum(sim._l2_pressure**2))
            }
            if not args.vtk:
                np.savez_compressed(filename, **data)

        if args.plots:
            times = np.arange(0.0, nt+1, IO_rate)
            pyplot.plot(times, data['err_l2_sub'], color="#3070B3", label = 'L2 subs.')
            if 'sim' in locals() and nx == 128 and ne == 3 and coupling == 'centered':
                S = 4*nx
                for t, ex, diff in zip(times[:S:nx] / nt, sim._l2_subsidence[:S:nx], data['err_l2_sub'][:S:nx]):
                    print(f"{t:.4f} {ex:.4f} {diff:.4e}")
                for t, ex, diff in zip(times[S::S] / nt, sim._l2_subsidence[S::S], data['err_l2_sub'][S::S]):
                    print(f"{t:.4f} {ex:.4f} {diff:.4e}")
            pyplot.plot(times, data['err_l2_pres'] / nx, color="#EF9067", label = 'e^k_p')
            pyplot.xlabel('Time Step')
            pyplot.ylabel('Error')
            pyplot.yscale('log')
            pyplot.legend(loc = 'upper right')
            pyplot.grid(linestyle = '--', which = 'both')
            pyplot.title(f"alpha = {args.alpha}, N_x = {nx}, N_E = {ne}, {coupling}{' with alt. grad.' if args.gradient else ''}")
            if args.save_fig:
                filename = os.path.join(output_dir, f"a{args.alpha:.3f}_nx{nx:03d}_ne{ne:03d}_l{levels:03d}_{coupling}{'_altgrad' if args.gradient else ''}")
                pyplot.savefig(filename + ".eps")
                saveTIKZ(filename + ".tex")
                pyplot.close()
            else:
                pyplot.show()

        err_sub[i] = data['rel_err_sub']
        err_pres[i] = data['rel_err_pres']

    if nx_list.size > 1:
        dx_list = nx_list.astype(float)**-1
        pyplot.loglog(dx_list, err_sub, color="#3070B3", label = 'L2 subs.')
        pyplot.loglog(dx_list, err_pres, color="#EF9067", label = 'e_p')
        pyplot.loglog(dx_list, dx_list, 'k:', label = 'O(dx)')
        pyplot.xlabel('dx')
        pyplot.ylabel('Rel. Error')
        pyplot.legend(loc = 'upper left')
        pyplot.grid(linestyle = '--', which = 'both')
        if args.levels < 1:
            pyplot.title(f'alpha = {args.alpha}, N_E = {ne}, {coupling}{" with alt. grad." if args.gradient else ""}')
        else:
            pyplot.title(f'alpha = {args.alpha}, N_E = {args.elastic_steps} N_x, {coupling}{" with alt. grad." if args.gradient else ""}')
        if args.save_fig:
            if args.levels < 1:
                filename = os.path.join(output_dir, f"convergence_a{args.alpha:.3f}_ne{ne}_{coupling}{'_altgrad' if args.gradient else ''}")
            else:
                filename = os.path.join(output_dir, f"convergence_a{args.alpha:.3f}_ne{args.elastic_steps:.3f}nx_{coupling}{'_altgrad' if args.gradient else ''}")
            pyplot.savefig(filename + ".eps")
            saveTIKZ(filename + ".tex")
            pyplot.close()
        else:
            pyplot.show()
