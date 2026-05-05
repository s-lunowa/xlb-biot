"""
This example implements a 2D periodic poroelastic flow simulation using the lattice Boltzmann method (LBM).

The flow is modeled by the Darcy equation.
"""
import argparse
from jax import config
import numpy as np
from matplotlib import pyplot
import os

#from src.boundary_conditions import *
from src.poroelasticity import LBMPoroelasticity
from src.darcy import AdvectionDiffusionReactionBGK
from src.elasticity import LBMElasticity
from src.lattice import LatticeD2Q8, LatticeD2Q9
from src.utils import *

# enable 64bit operations
config.update("jax_enable_x64", True)

def sumFunction(nx, ny, kls, scale=lambda k,l: 1):
    def fun(f,g, nx, ny, k, l):
        return np.fromfunction(lambda x,y,d: scale(k,l) * f(k*np.pi*(x+0.5)/nx) * g(l*np.pi*(y+0.5)/ny), (nx,ny,1))

    val = np.zeros((nx, ny, 1))
    for j, fg in enumerate([(np.cos, np.cos), (np.cos, np.sin), (np.sin, np.cos), (np.sin, np.sin)]):
        for i in range(len(kls[j]) // 2):
            val += fun(*fg, nx, ny, kls[j][2*i], kls[j][2*i+1])
    return val

class PeriodicElasticity(LBMElasticity):
    def initialize_macroscopic_fields(self):
        eta = np.concatenate((sumFunction(nx, ny, [[], [2,2], [], []], lambda k,l: 9e-4),
                              sumFunction(nx, ny, [[], [], [2,2], []], lambda k,l: 7e-4)), axis=-1)
        eta = self.distributed_array_init(eta.shape, self.precisionPolicy.output_dtype, init_val=eta, sharding=self.sharding)
        sigma = np.concatenate((sumFunction(nx, ny, [[], [], [], [2,2]], lambda k,l: np.pi / self.nx * (-3.2e-3*self.lamda - 3.6e-3*self.mu)),  # sigma xx
                                sumFunction(nx, ny, [[2,2], [], [], []], lambda k,l: np.pi / self.nx * 3.2e-3*mu),                    # sigma xy
                                sumFunction(nx, ny, [[], [], [], [2,2]], lambda k,l: np.pi / self.nx * (-3.2e-3*self.lamda - 2.8e-3*self.mu))), # sigma yy
                                axis=-1)
        sigma = self.distributed_array_init(sigma.shape, self.precisionPolicy.output_dtype, init_val=sigma, sharding=self.sharding)
        return eta, sigma

    def get_force(self):
        # epsilon**2 "=" L^2 / U in the non-dimensionalization
        epsilon = 1.0/self.nx

        force = np.concatenate((sumFunction(self.nx, self.ny, [[], [2,2], [], []], lambda k,l: epsilon**2 * np.pi**2 * 1.6e-3*self.mu),
                                np.zeros((self.nx, self.ny, 1))), axis=-1)
        return self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=force)

class PeriodicPoroelasticity(LBMPoroelasticity):
    def __init__(self, **kwargs):
        super().__init__(PeriodicElasticity, AdvectionDiffusionReactionBGK,**kwargs)
        self._output_dir = kwargs.get('output_dir')
        self._save_vtk = kwargs.get('save_vtk')

    @partial(jit, static_argnums=(0,))
    def get_darcy_source(self, timestep):
        mu = self.elasticity.mu
        lamda = self.elasticity.lamda
        K = (1.0 / self.darcy.omega - 0.5) / 3
        source = sumFunction(self.darcy.nx, self.darcy.ny, [[], [], [], [2,2]], lambda k,l: -K*8*np.pi**3/self.darcy.nx**2 * (6e-3*mu+3.2e-3*lamda) / self.alpha)
        return self.darcy.distributed_array_init((self.darcy.nx, self.darcy.ny, 1), self.darcy.precisionPolicy.compute_dtype, init_val=source)

    def run(self, t_max):
        self._err_l2_eta = np.empty((t_max // self.ioRate + 1,))
        self._err_lInf_eta = np.empty((t_max // self.ioRate + 1,))
        self._err_l2_dt = np.empty((t_max // self.ioRate + 1,))
        self._err_lInf_dt = np.empty((t_max // self.ioRate + 1,))
        self._err_l2_sigma = np.empty((t_max // self.ioRate + 1,))
        self._err_lInf_sigma = np.empty((t_max // self.ioRate + 1,))
        self._err_l2_p = np.empty((t_max // self.ioRate + 1,))
        self._err_lInf_p = np.empty((t_max // self.ioRate + 1,))

        nx = self.elasticity.nx
        ny = self.elasticity.ny
        mu = self.elasticity.mu
        lamda = self.elasticity.lamda

        self._eta_exact = np.concatenate((sumFunction(nx, ny, [[], [2,2], [], []], lambda k,l: 9e-4),
                                          sumFunction(nx, ny, [[], [], [2,2], []], lambda k,l: 7e-4)), axis=-1)
        self._l2_eta_exact = np.linalg.norm(self._eta_exact.ravel()) / nx

        self._sigma_exact = np.concatenate((sumFunction(nx, ny, [[], [], [], [2,2]], lambda k,l: np.pi * (-3.2e-3*lamda - 3.6e-3*mu)),  # sigma xx
                                            sumFunction(nx, ny, [[2,2], [], [], []], lambda k,l: np.pi * 3.2e-3*mu),                    # sigma xy
                                            sumFunction(nx, ny, [[], [], [], [2,2]], lambda k,l: np.pi * (-3.2e-3*lamda - 2.8e-3*mu))), # sigma yy
                                            axis=-1)
        self._l2_sigma_exact = np.linalg.norm(self._sigma_exact.ravel()) / nx

        self._p_exact = sumFunction(nx, ny, [[], [], [], [2,2]], lambda k,l: -np.pi * (6e-3*mu + 3.2e-3*lamda) / self.alpha)
        self._l2_p_exact = np.linalg.norm(self._p_exact.ravel()) / nx

        def initialize_macroscopic_fields():
            p = self.darcy.distributed_array_init(self._p_exact.shape, self.darcy.precisionPolicy.output_dtype, init_val=self._p_exact, sharding=self.darcy.sharding)
            vel = self.darcy.distributed_array_init((self.darcy.nx, self.darcy.ny, self.darcy.dim), self.darcy.precisionPolicy.output_dtype, init_val=0.0, sharding=self.darcy.sharding)
            return p, vel
        self.darcy.initialize_macroscopic_fields = initialize_macroscopic_fields

        super().run(t_max)

    def output_data(self, **kwargs):
        nx = self.elasticity.nx

        timestep = kwargs["timestep"]
        eta = np.array(kwargs["eta"])[0]
        sigma = np.array(kwargs["sigma"])[0] / self.epsilon
        pressure = np.array(kwargs["pressure"])[0]
        dt_div_eta = np.array(kwargs["dt_div_eta"])[0] / self.epsilon * nx**2 # viscous scaling

        eta_diff = eta - self._eta_exact
        self._err_l2_eta[timestep // self.ioRate] = np.linalg.norm(eta_diff.ravel()) / nx  / self._l2_eta_exact
        self._err_lInf_eta[timestep // self.ioRate] = np.linalg.norm(eta_diff.ravel(), ord=np.Inf)  / self._l2_eta_exact

        self._err_l2_dt[timestep // self.ioRate] = np.linalg.norm(dt_div_eta) / nx
        self._err_lInf_dt[timestep // self.ioRate] = np.linalg.norm(dt_div_eta.ravel(), ord=np.Inf)

        sigma_diff = sigma - self._sigma_exact
        self._err_l2_sigma[timestep // self.ioRate] = np.linalg.norm(sigma_diff.ravel()) / nx / self._l2_sigma_exact
        self._err_lInf_sigma[timestep // self.ioRate] = np.linalg.norm(sigma_diff.ravel(), ord=np.Inf) / self._l2_sigma_exact

        p_diff = pressure - self._p_exact
        self._err_l2_p[timestep // self.ioRate] = np.linalg.norm(p_diff.ravel()) / nx  / self._l2_p_exact
        self._err_lInf_p[timestep // self.ioRate] = np.linalg.norm(p_diff.ravel(), ord=np.Inf)  / self._l2_p_exact

        if self._save_vtk:
            fields = {"eta_x": eta[..., 0], "eta_y": eta[..., 1], "sigma_xx": sigma[..., 0], "sigma_xy": sigma[..., 1], "sigma_yy": sigma[..., 2],
                      "eta_ex_x": self._eta_exact[..., 0], "eta_ex_y": self._eta_exact[..., 1], "sigma_ex_xx": self._sigma_exact[..., 0], "sigma_ex_xy": self._sigma_exact[..., 1], "sigma_ex_yy": self._sigma_exact[..., 2],
                      "diff_eta_x": eta_diff[..., 0], "diff_eta_y": eta_diff[..., 1], "diff_sigma_xx": sigma_diff[..., 0], "diff_sigma_xy": sigma_diff[..., 1], "diff_sigma_yy": sigma_diff[..., 2],
                      "p": pressure[..., 0], "p_ex": self._p_exact[..., 0], "diff_p": p_diff[..., 0], "dt_div_eta": dt_div_eta[..., 0]}
            save_fields_vtk(timestep, fields, self._output_dir)
            # save_BCs_vtk(timestep, self.BCs, self.gridInfo, self._output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2D periodic poroelastic simulation using LBM".')
    parser.add_argument('-a', '--alpha', type=float, help="The Biot-Willis coefficient between 0 and 1 (default: 1.0).", default=1.0)
    parser.add_argument('-n', '--nx', type=int, help="The number of cells in x-direction (default: 20).", default=20)
    parser.add_argument('-e', '--elastic_steps', type=int, help="The number of elasticity steps per timestep (default: 1).", default=1)
    parser.add_argument('-l', '--levels', type=int, help="The number of multigrid levels for elasticity (default: 1).", default=1)
    parser.add_argument('-r', '--refinements', type=int, help="The number of refinements for convergence study (default: 0).", default=0)
    parser.add_argument('-c', '--coupling', type=str, help="The coupling method (default: 'centered').", default='centered', choices=['explicit', 'implicit', 'centered'])
    parser.add_argument('-p', '--plots', help="Whether to show individual convergence plots (default: True).", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-s', '--save_fig', help="Whether to save the plots as figures (default: False).", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-v', '--vtk', help="Whether to save vtk files (default: False).", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    precision = "f64/f64"
    lattice_elasticity = LatticeD2Q8(precision)
    lattice_darcy = LatticeD2Q9(precision)

    nu = 0.8
    E = 0.11
    K = 0.1

    lamda = E * nu / (1 - nu**2)
    mu = 0.5 * E / (1 + nu)
    omega = 1.0 / (3*K + 0.5)

    output_dir = os.path.abspath("./results/poroelasticity/periodic_stationary")
    os.makedirs(output_dir, exist_ok=True)

    nx_list = np.array([args.nx * (r+1) for r in range(args.refinements+1)])
    err_l2_eta = np.empty_like(nx_list, dtype=float)
    err_lInf_eta = np.empty_like(nx_list, dtype=float)
    err_l2_dt = np.empty_like(nx_list, dtype=float)
    err_lInf_dt = np.empty_like(nx_list, dtype=float)
    err_l2_sigma = np.empty_like(nx_list, dtype=float)
    err_lInf_sigma = np.empty_like(nx_list, dtype=float)
    err_l2_p = np.empty_like(nx_list, dtype=float)
    err_lInf_p = np.empty_like(nx_list, dtype=float)

    for i, nx in enumerate(nx_list):
        nx = int(nx)
        ny = nx
        nt = nx**2
        ne = args.elastic_steps
        if args.levels < 1:
            levels = int(np.log2(nx))
            while nx % 2**(levels-1) != 0:
                levels -= 1
        else:
            levels = args.levels
        coupling = args.coupling

        kwargs = {
            'lattice_elasticity': lattice_elasticity,
            'lattice_darcy': lattice_darcy,
            'elasticity_steps': ne,
            'elasticity_MG_levels': levels,
            'coupling_method': coupling,
            'alpha': args.alpha,
            'lambda': lamda,
            'mu': mu,
            'omega': omega,
            'nx': nx,
            'ny': ny,
            'nz': 0,
            'precision': precision,
            'io_rate': max(1, nt // 100) if args.vtk else 1,
            'save_vtk': args.vtk,
            'print_info_rate': 1000,
            'output_dir': output_dir
        }

        sim = PeriodicPoroelasticity(**kwargs)
        sim.run(nt)

        if args.plots:
            times = np.arange(0.0, nt+1, sim.ioRate)
            pyplot.plot(times, sim._err_l2_eta, 'b-', label = 'L2 disp')
            pyplot.plot(times, sim._err_lInf_eta, 'b--', label = 'Linf disp')
            pyplot.plot(times, sim._err_l2_sigma, 'r-', label = 'L2 stress')
            pyplot.plot(times, sim._err_lInf_sigma, 'r--', label = 'Linf stress')
            pyplot.plot(times, sim._err_l2_p, 'g-', label = 'L2 e_p')
            pyplot.plot(times, sim._err_lInf_p, 'g--', label = 'Linf e_p')
            pyplot.plot(times, sim._err_l2_dt, 'k-', label = 'L2 d/dt div disp')
            pyplot.plot(times, sim._err_lInf_dt, 'k--', label = 'LInf d/dt div disp')
            pyplot.xlabel('Time Step')
            pyplot.ylabel('Rel. Error')
            pyplot.yscale('log')
            pyplot.ylim(bottom=1e-9)
            pyplot.legend(loc = 'lower right')
            pyplot.grid(linestyle = '--', which = 'both')
            pyplot.title(f'E = {E}, nu = {nu}, K = {K}, alpha = {args.alpha}, nx = {nx}, ne = {ne}, {coupling}')
            if args.save_fig:
                filename = os.path.join(output_dir, f"error_E{E:.3f}_nu{nu:.3f}_K{K:.3f}_a{args.alpha:.3f}_nx{nx:03d}_ne{ne:03d}_l{levels:03d}_{coupling}.npz")
                pyplot.savefig(filename)
                pyplot.close()
            else:
                pyplot.show()

        err_l2_eta[i] = sim._err_l2_eta[-1]
        err_lInf_eta[i] = sim._err_lInf_eta[-1]
        err_l2_sigma[i] = sim._err_l2_sigma[-1]
        err_lInf_sigma[i] = sim._err_lInf_sigma[-1]
        err_l2_p[i] = sim._err_l2_p[-1]
        err_lInf_p[i] = sim._err_lInf_p[-1]
        err_l2_dt[i] = sim._err_l2_dt[-1]
        err_lInf_dt[i] = sim._err_lInf_dt[-1]

    if nx_list.size > 1:
        pyplot.loglog(nx_list, err_l2_eta, 'b-', label = 'L2 disp')
        pyplot.loglog(nx_list, err_lInf_eta, 'b--', label = 'Linf disp')
        pyplot.loglog(nx_list, err_l2_sigma, 'r-', label = 'L2 stress')
        pyplot.loglog(nx_list, err_lInf_sigma, 'r--', label = 'Linf stress')
        pyplot.loglog(nx_list, err_l2_p, 'g-', label = 'L2 e_p')
        pyplot.loglog(nx_list, err_lInf_p, 'g--', label = 'Linf e_p')
#        pyplot.loglog(nx_list, err_l2_dt, '-', c='C1', label = 'L2 e_div')
#        pyplot.loglog(nx_list, err_lInf_dt, '--', c='C1', label = 'Linf e_div')
        pyplot.loglog(nx_list, nx_list[0]**2 * nx_list.astype(float)**-2, 'k:', label = 'O(nx^-2)')
        pyplot.xlabel('nx')
        pyplot.ylabel('Rel. Error')
        pyplot.legend(loc = 'lower left')
        pyplot.grid(linestyle = '--', which = 'both')
        pyplot.title(f'E = {E}, nu = {nu}, K = {K}, alpha = {args.alpha}, ne = {ne}, {coupling}')
        if args.save_fig:
            filename = os.path.join(output_dir, f"convergence_E{E:.3f}_nu{nu:.3f}_K{K:.3f}_a{args.alpha:.3f}_ne{ne:03d}_{coupling}.eps")
            pyplot.savefig(filename)
            pyplot.close()
        else:
            pyplot.show()
