"""
This example implements a 2D Darcy flow simulation using the lattice Boltzmann method (LBM).
Zero Neumann boundary conditions are applied.
"""
import argparse
import numpy as np
from matplotlib import pyplot
import os

from src.darcy import AdvectionDiffusionReactionBGK, ADR_Neumann_BounceBack
from src.lattice import LatticeD2Q9
from src.utils import *

def sumFunction(nx, ny, kls, scale=lambda k,l: 1):
    def fun(f,g, nx, ny, k, l):
        # shift by 2 due to BC cells
        return np.fromfunction(lambda x,y,d: scale(k,l) * f(k*np.pi*(x-0.5)/(nx-2)) * g(l*np.pi*(y-0.5)/(ny-2)), (nx,ny,1))

    val = np.zeros((nx, ny, 1))
    for j, fg in enumerate([(np.cos, np.cos), (np.cos, np.sin), (np.sin, np.cos), (np.sin, np.sin)]):
        for i in range(len(kls[j]) // 2):
            val += fun(*fg, nx, ny, kls[j][2*i], kls[j][2*i+1])
    return val

class NeumannDarcy(AdvectionDiffusionReactionBGK):
    def __init__(self, **kwargs):
        self._kls = [[2,4, 4,2], [], [], []]
        super().__init__(**kwargs)
        self._output_dir = kwargs.get('output_dir')
        self._save_vtk = kwargs.get('save_vtk')

    def run(self, t_max):
        self._l2_p_exact = np.empty((t_max // self.ioRate + 1,))
        self._lInf_p_exact = np.empty_like(self._l2_p_exact)
        self._err_l2_p = np.empty_like(self._l2_p_exact)
        self._err_lInf_p = np.empty_like(self._l2_p_exact)

        super().run(t_max)

    def get_source(self):
        K = (1.0 / self.omega - 0.5) / 3
        # shift by 2 due to BC cells
        source = sumFunction(self.nx, self.ny, self._kls, lambda k,l: 0.5 * K * (k**2 + l**2) * np.pi**2 / (self.nx-2)**2)
        source[(0, -1), :] = 0
        source[:, (0, -1)] = 0
        return self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=source)

    def initialize_macroscopic_fields(self):
        p = sumFunction(self.nx, self.ny, self._kls)
        p = self.distributed_array_init(p.shape, self.precisionPolicy.output_dtype, init_val=p, sharding=self.sharding)
        p -= 0.5 * self.source
        vel = self.distributed_array_init((self.nx, self.ny, self.dim), self.precisionPolicy.output_dtype, init_val=0.0, sharding=self.sharding)
        return p, vel

    def set_boundary_conditions(self):
        # concatenate the indices of the left, right, top and bottom walls
        walls = np.concatenate((self.boundingBoxIndices["left"], self.boundingBoxIndices["right"], self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"]))
        # apply bounce back boundary condition to the walls
        self.BCs.append(ADR_Neumann_BounceBack(tuple(walls.T), self.gridInfo, self.precisionPolicy))

    def output_data(self, **kwargs):
        pressure = np.array(kwargs["rho"])[0]
        timestep = kwargs["timestep"]
        Kt = (1.0 / self.omega - 0.5) / 3 * timestep / self.nx**2 # viscous scaling

        p_exact = sumFunction(self.nx, self.ny, self._kls, lambda k,l: 0.5 + 0.5*np.exp(-Kt * (k**2 + l**2) * np.pi**2))
        #p_exact = sumFunction(self.nx, self.ny, self._kls, lambda k,l: np.exp(-Kt * (k**2 + l**2) * np.pi**2))
        self._l2_p_exact[timestep // self.ioRate] = np.linalg.norm(p_exact[1:-1, 1:-1].ravel())
        self._lInf_p_exact[timestep // self.ioRate] = np.linalg.norm(p_exact[1:-1, 1:-1].ravel(), ord=np.Inf)

        p_diff = pressure - p_exact
        self._err_l2_p[timestep // self.ioRate] = np.linalg.norm(p_diff[1:-1, 1:-1].ravel())
        self._err_lInf_p[timestep // self.ioRate] = np.linalg.norm(p_diff[1:-1, 1:-1].ravel(), ord=np.Inf)

        if self._save_vtk:
            fields = {"p": pressure[..., 0], "p_ex": p_exact[..., 0], "diff_p": p_diff[..., 0]}
            save_fields_vtk(timestep, fields, self._output_dir)
            # save_BCs_vtk(timestep, self.BCs, self.gridInfo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2D periodic Darcy flow simulation using LBM".')
    parser.add_argument('-n', '--nx', type=int, help="The number of cells in x-direction (default: 20).", default=20)
    parser.add_argument('-r', '--refinements', type=int, help="The number of refinements for convergence study (default: 0).", default=0)
    parser.add_argument('-p', '--plots', help="Whether to show individual convergence plots (default: True).", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-v', '--vtk', help="Whether to save vtk files (default: False).", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    precision = "f32/f32"
    lattice = LatticeD2Q9(precision)

    K = 0.1
    omega = 1.0 / (3*K + 0.5)

    output_dir = os.path.abspath("./results/darcy/neumann_bb")
    os.makedirs(output_dir, exist_ok=True)

    nx_list = np.array([args.nx * 2**r for r in range(args.refinements+1)])
    err_l2_p = np.empty_like(nx_list, dtype=float)
    err_lInf_p = np.empty_like(nx_list, dtype=float)

    for i, nx in enumerate(nx_list):
        nx = int(nx) + 2
        ny = nx
        nt = (nx-2)**2

        kwargs = {
            'lattice': lattice,
            'omega': omega,
            'nx': nx,
            'ny': ny,
            'nz': 0,
            'precision': precision,
            'io_rate': max(1, nt // 100),
            'save_vtk': args.vtk,
            'print_info_rate': 1000,
            'output_dir': output_dir
        }

        sim = NeumannDarcy(**kwargs)
        sim.run(nt)

        if args.plots:
            times = np.arange(0.0, nt+1, sim.ioRate)
            pyplot.plot(times, sim._err_l2_p / sim.nx, '-', color="#3070B3", label = 'L^2 e_p')
            pyplot.plot(times, sim._err_lInf_p, '--', color="#EF9067", label = 'L^inf e_p')
            pyplot.xlabel('Time Step')
            pyplot.ylabel('Error')
            pyplot.yscale('log')
            pyplot.legend(loc = 'lower right')
            pyplot.grid(linestyle = '--', which = 'both')
            pyplot.title(f'K = {K}, nx = {nx}')
            pyplot.show()

        err_l2_p[i] = np.sqrt(np.sum(sim._err_l2_p**2) / np.sum(sim._l2_p_exact**2))
        err_lInf_p[i] = np.max(sim._err_lInf_p) / np.max(sim._lInf_p_exact)

    if nx_list.size > 1:
        pyplot.loglog(nx_list, err_l2_p, '-', color="#3070B3", label = 'L^2 e_p')
        pyplot.loglog(nx_list, err_lInf_p, '--', color="#EF9067", label = 'L^inf e_p')
        pyplot.loglog(nx_list, nx_list[0] * nx_list.astype(float)**-1, 'k:', label = 'O(nx^-1)')
        pyplot.loglog(nx_list, nx_list[0]**2 * nx_list.astype(float)**-2, 'k:', label = 'O(nx^-2)')
        pyplot.xlabel('nx')
        pyplot.ylabel('Rel. Error')
        pyplot.legend(loc = 'lower left')
        pyplot.grid(linestyle = '--', which = 'both')
        pyplot.title(f'K = {K}')
        pyplot.show()
