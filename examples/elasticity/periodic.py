"""
This example implements a 2D periodic elasticity simulation using the lattice Boltzmann method (LBM).

The values are taken from section 7.3.1 of
    Boolakee, Geier, De Lorenzis (2023)
    A new lattice Boltzmann scheme for linear elastic solids: periodic problems
    Comput. Methods Appl. Mech. Engrg. 404, https://doi.org/10.1016/j.cma.2022.115756
"""
import argparse
from jax import config
import numpy as np
from matplotlib import pyplot
import os

#from src.boundary_conditions import *
from src.elasticity import LBMElasticity
from src.lattice import LatticeD2Q8
from src.utils import *

def sumFunction(nx, ny, kls, scale=lambda k,l: 1):
    def fun(f,g, nx, ny, k, l):
        return np.fromfunction(lambda x,y,d: scale(k,l) * f(k*np.pi*(x+0.5)/nx) * g(l*np.pi*(y+0.5)/ny), (nx,ny,1))

    val = np.zeros((nx, ny, 1))
    for j, fg in enumerate([(np.cos, np.cos), (np.cos, np.sin), (np.sin, np.cos), (np.sin, np.sin)]):
        for i in range(len(kls[j]) // 2):
            val += fun(*fg, nx, ny, kls[j][2*i], kls[j][2*i+1])
    return val

class PeriodicElasticity(LBMElasticity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._output_dir = kwargs.get('output_dir')
        self._save_vtk = kwargs.get('save_vtk')

    def run(self, t_max):
        self._err_l2_eta = np.empty((t_max // self.ioRate + 1,))
        self._err_lInf_eta = np.empty((t_max // self.ioRate + 1,))
        self._err_l2_sigma = np.empty((t_max // self.ioRate + 1,))
        self._err_lInf_sigma = np.empty((t_max // self.ioRate + 1,))

        self._eta_exact = np.concatenate((sumFunction(self.nx, self.ny, [[], [2,2], [], []], lambda k,l: 9e-4),
                                          sumFunction(self.nx, self.ny, [[], [], [2,2], []], lambda k,l: 7e-4)), axis=-1)
        self._l2_eta_exact = np.linalg.norm(self._eta_exact.ravel())

        self._sigma_exact = np.concatenate((sumFunction(self.nx, self.ny, [[], [], [], [2,2]], lambda k,l: np.pi * (-3.2e-3*self.lamda - 3.6e-3*self.mu)),  # sigma xx
                                            sumFunction(self.nx, self.ny, [[2,2], [], [], []], lambda k,l: np.pi * 3.2e-3*self.mu),                         # sigma xy
                                            sumFunction(self.nx, self.ny, [[], [], [], [2,2]], lambda k,l: np.pi * (-3.2e-3*self.lamda - 2.8e-3*self.mu))), # sigma yy
                                            axis=-1)
        self._l2_sigma_exact = np.linalg.norm(self._sigma_exact.ravel())

        super().run(t_max)

    def get_force(self):
        # epsilon**2 "=" L^2 / U in the non-dimensionalization
        epsilon = 1.0/self.nx

        force = np.concatenate((sumFunction(self.nx, self.ny, [[], [2,2], [], []], lambda k,l: epsilon**2 * np.pi**2 * (1.36e-2*self.mu+6.4e-3*self.lamda)),
                                sumFunction(self.nx, self.ny, [[], [], [2,2], []], lambda k,l: epsilon**2 * np.pi**2 * (1.20e-2*self.mu+6.4e-3*self.lamda))),
                                axis=-1, dtype=self.precisionPolicy.compute_dtype)
        return self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=force)

    def output_data(self, **kwargs):
        # epsilon "=" L / U in the non-dimensionalization
        epsilon = 1.0/self.nx

        eta = np.array(kwargs["eta"])[0]
        sigma = np.array(kwargs["sigma"])[0] / epsilon
        timestep = kwargs["timestep"]

        eta_diff = eta - self._eta_exact
        self._err_l2_eta[timestep // self.ioRate] = np.linalg.norm(eta_diff.ravel()) / self._l2_eta_exact
        self._err_lInf_eta[timestep // self.ioRate] = np.linalg.norm(eta_diff.ravel(), ord=np.Inf) * self.nx / self._l2_eta_exact

        sigma_diff = sigma - self._sigma_exact
        self._err_l2_sigma[timestep // self.ioRate] = np.linalg.norm(sigma_diff.ravel()) / self._l2_sigma_exact
        self._err_lInf_sigma[timestep // self.ioRate] = np.linalg.norm(sigma_diff.ravel(), ord=np.Inf) * self.nx / self._l2_sigma_exact

        if self._save_vtk:
            fields = {"eta_x": eta[..., 0], "eta_y": eta[..., 1], "sigma_xx": sigma[..., 0], "sigma_xy": sigma[..., 1], "sigma_yy": sigma[..., 2],
                    "eta_ex_x": self._eta_exact[..., 0], "eta_ex_y": self._eta_exact[..., 1], "sigma_ex_xx": self._sigma_exact[..., 0], "sigma_ex_xy": self._sigma_exact[..., 1], "sigma_ex_yy": self._sigma_exact[..., 2],
                    "diff_eta_x": eta_diff[..., 0], "diff_eta_y": eta_diff[..., 1], "diff_sigma_xx": sigma_diff[..., 0], "diff_sigma_xy": sigma_diff[..., 1], "diff_sigma_yy": sigma_diff[..., 2],}
            save_fields_vtk(timestep, fields, self._output_dir)
            # save_BCs_vtk(timestep, self.BCs, self.gridInfo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2D periodic elasticity simulation using LBM".')
    parser.add_argument('-n', '--nx', type=int, help="The number of cells in x-direction (default: 20).", default=20)
    parser.add_argument('-r', '--refinements', type=int, help="The number of refinements for convergence study (default: 0).", default=0)
    parser.add_argument('-p', '--plots', help="Whether to show individual convergence plots (default: True).", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-v', '--vtk', help="Whether to save vtk files (default: False).", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    precision = "f32/f32"
    lattice = LatticeD2Q8(precision)

    nu = 0.8
    E = 0.11
    #E = 0.085

    lamda = E * nu / (1 - nu**2)
    mu = 0.5 * E / (1 + nu)

    output_dir = os.path.abspath("./results/elasticity/periodic")
    os.makedirs(output_dir, exist_ok=True)

    nx_list = np.array([args.nx * 2**r for r in range(args.refinements+1)])
    err_l2_eta = np.empty_like(nx_list, dtype=float)
    err_lInf_eta = np.empty_like(nx_list, dtype=float)
    err_l2_sigma = np.empty_like(nx_list, dtype=float)
    err_lInf_sigma = np.empty_like(nx_list, dtype=float)

    for i, nx in enumerate(nx_list):
        nx = int(nx)
        ny = nx
        nt = 4 * nx**2

        kwargs = {
            'lattice': lattice,
            'lambda': lamda,
            'mu': mu,
            'nx': nx,
            'ny': ny,
            'nz': 0,
            'precision': precision,
            'io_rate': max(1, nt // 100),
            'save_vtk': args.vtk,
            'print_info_rate': 1000,
            'output_dir': output_dir
        }

        sim = PeriodicElasticity(**kwargs)
        sim.run(nt)

        if args.plots:
            times = np.arange(0.0, nt+1, sim.ioRate)
            pyplot.plot(times, sim._err_l2_eta, '-', color="#3070B3", label = 'L^2 displacement')
            pyplot.plot(times, sim._err_l2_sigma, '-', color="#EF9067", label = 'L^2 stress')
            pyplot.plot(times, sim._err_lInf_eta, '--', color="#3070B3", label = 'L^inf displacement')
            pyplot.plot(times, sim._err_lInf_sigma, '--', color="#EF9067", label = 'L^inf stress')
            pyplot.xlabel('Time Step')
            pyplot.ylabel('Rel. Error')
            pyplot.yscale('log')
            # pyplot.ylim((10**-3,2))
            pyplot.legend(loc = 'upper right')
            pyplot.grid(linestyle = '--', which = 'both')
            pyplot.title(f'E = {E}, nu = {nu}, nx = {nx}')
            pyplot.show()

        err_l2_eta[i] = sim._err_l2_eta[-1]
        err_lInf_eta[i] = sim._err_lInf_eta[-1]
        err_l2_sigma[i] = sim._err_l2_sigma[-1]
        err_lInf_sigma[i] = sim._err_lInf_sigma[-1]

    if nx_list.size > 1:
        pyplot.loglog(nx_list, err_l2_eta, '-', color="#3070B3", label = 'L^2 displacement')
        pyplot.loglog(nx_list, err_l2_sigma, '-', color="#EF9067", label = 'L^2 stress')
        pyplot.loglog(nx_list, err_lInf_eta, '--', color="#3070B3", label = 'L^inf displacement')
        pyplot.loglog(nx_list, err_lInf_sigma, '--', color="#EF9067", label = 'L^inf stress')
        pyplot.loglog(nx_list, nx_list[0]**2 * nx_list.astype(float)**-2, 'k:', label = 'O(nx^-2)')
        pyplot.xlabel('nx')
        pyplot.ylabel('Rel. Error')
        pyplot.legend(loc = 'upper right')
        pyplot.grid(linestyle = '--', which = 'both')
        pyplot.title(f'E = {E}, nu = {nu}')
        pyplot.show()
