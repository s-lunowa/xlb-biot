"""
This example implements a 2D periodic poroelastic flow simulation using the lattice Boltzmann method (LBM).

The flow is modeled by the Darcy equation.
"""
import argparse
from jax import config
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot
from tikzplotlib import save as saveTIKZ
import os

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

class PeriodicDarcy(AdvectionDiffusionReactionBGK):
    def initialize_macroscopic_fields(self):
        p = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.output_dtype, init_val=0.0, sharding=self.sharding)
        vel = self.distributed_array_init((self.nx, self.ny, self.dim), self.precisionPolicy.output_dtype, init_val=0.0, sharding=self.sharding)
        return p, vel

class PeriodicElasticity(LBMElasticity):
    def initialize_macroscopic_fields(self):
        eta = self.distributed_array_init((self.nx, self.ny, self.dim), self.precisionPolicy.output_dtype, init_val=0.0, sharding=self.sharding)
        sigma = self.distributed_array_init((self.nx, self.ny, 3), self.precisionPolicy.output_dtype, init_val=0.0, sharding=self.sharding)
        return eta, sigma

class PeriodicPoroelasticity(LBMPoroelasticity):
    def __init__(self, **kwargs):
        super().__init__(PeriodicElasticity, PeriodicDarcy,**kwargs)
        self._output_dir = kwargs.get('output_dir')
        self._save_vtk = kwargs.get('save_vtk')

    @partial(jit, static_argnums=(0,))
    def get_elastic_force(self, timestep):
        K = (1.0 / self.darcy.omega - 0.5) / 3
        timefactor = 1 - jnp.exp(-8 * jnp.pi**2 * K * timestep / self.elasticity.nx**2) # viscous scaling

        force = jnp.concatenate((sumFunction(self.elasticity.nx, self.elasticity.ny, [[], [2,2], [], []], lambda k,l: self.epsilon**2 * timefactor * jnp.pi**2 * 8 * self.elasticity.mu),
                                jnp.zeros((self.elasticity.nx, self.elasticity.ny, 1))), axis=-1)
        return self.elasticity.distributed_array_init((self.elasticity.nx, self.elasticity.ny, 2), self.elasticity.precisionPolicy.compute_dtype, init_val=force)

    @partial(jit, static_argnums=(0,))
    def get_darcy_source(self, timestep):
        K = (1.0 / self.darcy.omega - 0.5) / 3
        timefactor = jnp.exp(-8 * jnp.pi**2 * K * timestep / self.elasticity.nx**2) # viscous scaling
        factor = -jnp.pi**3 * K / self.darcy.nx**2 * (128 * self.alpha * timefactor + (128 * self.elasticity.lamda + 240 * self.elasticity.mu) / self.alpha)
        source = sumFunction(self.darcy.nx, self.darcy.ny, [[], [], [], [2,2]], lambda k,l: factor)
        return self.darcy.distributed_array_init((self.darcy.nx, self.darcy.ny, 1), self.darcy.precisionPolicy.compute_dtype, init_val=source)

    def run(self, t_max):
        self._l2_eta_exact     = np.empty((t_max // self.ioRate + 1,))
        self._lInf_eta_exact   = np.empty_like(self._l2_eta_exact)
        self._err_l2_eta       = np.empty_like(self._l2_eta_exact)
        self._err_lInf_eta     = np.empty_like(self._l2_eta_exact)
        self._l2_sigma_exact   = np.empty_like(self._l2_eta_exact)
        self._lInf_sigma_exact = np.empty_like(self._l2_eta_exact)
        self._err_l2_sigma     = np.empty_like(self._l2_eta_exact)
        self._err_lInf_sigma   = np.empty_like(self._l2_eta_exact)
        self._l2_p_exact       = np.empty_like(self._l2_eta_exact)
        self._lInf_p_exact     = np.empty_like(self._l2_eta_exact)
        self._err_l2_p         = np.empty_like(self._l2_eta_exact)
        self._err_lInf_p       = np.empty_like(self._l2_eta_exact)
        self._l2_p_grad_exact  = np.empty_like(self._l2_eta_exact)
        self._lInf_p_grad_exact = np.empty_like(self._l2_eta_exact)
        self._err_l2_p_grad    = np.empty_like(self._l2_eta_exact)
        self._err_lInf_p_grad  = np.empty_like(self._l2_eta_exact)
        self._err_l2_dtdiv_eta = np.empty_like(self._l2_eta_exact)
        self._err_lInf_dtdiv_eta = np.empty_like(self._l2_eta_exact)
        self._l2_dtdiv_eta_exact = np.empty_like(self._l2_eta_exact)
        self._lInf_dtdiv_eta_exact = np.empty_like(self._l2_eta_exact)

        super().run(t_max)

    def output_data(self, **kwargs):
        K = (1.0 / self.darcy.omega - 0.5) / 3
        nx = self.elasticity.nx
        ny = self.elasticity.ny

        eta = np.array(kwargs["eta"])[0]
        dt_div_eta = np.array(kwargs["dt_div_eta"])[0] / self.epsilon**3
        sigma = np.array(kwargs["sigma"])[0] / self.epsilon
        pressure = np.array(kwargs["pressure"])[0]
        pressure_grad = np.array(kwargs["pressure_grad"])[0] / self.epsilon
        timestep = kwargs["timestep"]
        timefactor = 1 - np.exp(-8 * np.pi**2 * K * timestep / nx**2) # viscous scaling

        eta_exact = np.concatenate((sumFunction(nx, ny, [[], [2,2], [], []], lambda k,l: 4.5 * timefactor),
                                    sumFunction(nx, ny, [[], [], [2,2], []], lambda k,l: 3.5 * timefactor)), axis=-1)
        self._l2_eta_exact[timestep // self.ioRate] = np.linalg.norm(eta_exact)
        self._lInf_eta_exact[timestep // self.ioRate] = np.linalg.norm(eta_exact.ravel(), ord=np.Inf)

        eta_diff = eta - eta_exact
        self._err_l2_eta[timestep // self.ioRate] = np.linalg.norm(eta_diff)
        self._err_lInf_eta[timestep // self.ioRate] = np.linalg.norm(eta_diff.ravel(), ord=np.Inf)

        dt_div_eta_exact = sumFunction(nx, ny, [[], [], [], [2,2]], lambda k,l: -128 * np.pi**3 * K * np.exp(-8 * np.pi**2 * K * timestep / nx**2))
        self._l2_dtdiv_eta_exact[timestep // self.ioRate] = np.linalg.norm(dt_div_eta_exact)
        self._lInf_dtdiv_eta_exact[timestep // self.ioRate] = np.linalg.norm(dt_div_eta_exact.ravel(), ord=np.Inf)

        dt_div_eta_diff = dt_div_eta - dt_div_eta_exact
        self._err_l2_dtdiv_eta[timestep // self.ioRate] = np.linalg.norm(dt_div_eta_diff)
        self._err_lInf_dtdiv_eta[timestep // self.ioRate] = np.linalg.norm(dt_div_eta_diff.ravel(), ord=np.Inf)

        sigma_exact = np.concatenate((sumFunction(nx, ny, [[], [], [], [2,2]], lambda k,l: -np.pi * timefactor * (16 * self.elasticity.lamda + 18 * self.elasticity.mu)),  # sigma xx
                                      sumFunction(nx, ny, [[2,2], [], [], []], lambda k,l:  np.pi * timefactor *  16 * self.elasticity.mu),                                # sigma xy
                                      sumFunction(nx, ny, [[], [], [], [2,2]], lambda k,l: -np.pi * timefactor * (16 * self.elasticity.lamda + 14 * self.elasticity.mu))), # sigma yy
                                      axis=-1)
        self._l2_sigma_exact[timestep // self.ioRate] = np.linalg.norm(sigma_exact)
        self._lInf_sigma_exact[timestep // self.ioRate] = np.linalg.norm(sigma_exact.ravel(), ord=np.Inf)

        sigma_diff = sigma - sigma_exact
        self._err_l2_sigma[timestep // self.ioRate] = np.linalg.norm(sigma_diff)
        self._err_lInf_sigma[timestep // self.ioRate] = np.linalg.norm(sigma_diff.ravel(), ord=np.Inf)

        p_exact = sumFunction(nx, ny, [[], [], [], [2,2]], lambda k,l: -np.pi / self.alpha * timefactor * (16 * self.elasticity.lamda + 30 * self.elasticity.mu))
        self._l2_p_exact[timestep // self.ioRate] = np.linalg.norm(p_exact)
        self._lInf_p_exact[timestep // self.ioRate] = np.linalg.norm(p_exact.ravel(), ord=np.Inf)

        p_diff = pressure - p_exact
        self._err_l2_p[timestep // self.ioRate] = np.linalg.norm(p_diff)
        self._err_lInf_p[timestep // self.ioRate] = np.linalg.norm(p_diff.ravel(), ord=np.Inf)

        p_grad_exact = np.concatenate((sumFunction(nx, ny, [[], [2,2], [], []], lambda k,l: -2 * np.pi**2 / self.alpha * timefactor * (16 * self.elasticity.lamda + 30 * self.elasticity.mu)),
                                       sumFunction(nx, ny, [[], [], [2,2], []], lambda k,l: -2 * np.pi**2 / self.alpha * timefactor * (16 * self.elasticity.lamda + 30 * self.elasticity.mu))),
                                      axis=-1)
        self._l2_p_grad_exact[timestep // self.ioRate] = np.linalg.norm(p_grad_exact)
        self._lInf_p_grad_exact[timestep // self.ioRate] = np.linalg.norm(p_grad_exact.ravel(), ord=np.Inf)

        p_grad_diff = pressure_grad - p_grad_exact
        self._err_l2_p_grad[timestep // self.ioRate] = np.linalg.norm(p_grad_diff)
        self._err_lInf_p_grad[timestep // self.ioRate] = np.linalg.norm(p_grad_diff.ravel(), ord=np.Inf)

        if self._save_vtk:
            fields = {"eta_x": eta[..., 0], "eta_y": eta[..., 1], "sigma_xx": sigma[..., 0], "sigma_xy": sigma[..., 1], "sigma_yy": sigma[..., 2],
                      "eta_ex_x": eta_exact[..., 0], "eta_ex_y": eta_exact[..., 1], "sigma_ex_xx": sigma_exact[..., 0], "sigma_ex_xy": sigma_exact[..., 1], "sigma_ex_yy": sigma_exact[..., 2],
                      "diff_eta_x": eta_diff[..., 0], "diff_eta_y": eta_diff[..., 1], "diff_sigma_xx": sigma_diff[..., 0], "diff_sigma_xy": sigma_diff[..., 1], "diff_sigma_yy": sigma_diff[..., 2],
                      "p": pressure[..., 0], "p_ex": p_exact[..., 0], "diff_p": p_diff[..., 0],
                      "dp_dx": pressure_grad[..., 0], "dp_dy": pressure_grad[..., 1], "dp_ex_dx": p_grad_exact[..., 0], "dp_ex_dy": p_grad_exact[..., 1], "diff_dp_dx": p_grad_diff[..., 0], "diff_dp_dy": p_grad_diff[..., 1] }
            save_fields_vtk(timestep, fields, self._output_dir)
            # save_BCs_vtk(timestep, self.BCs, self.gridInfo, self._output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2D periodic poroelastic simulation using LBM".')
    parser.add_argument('-a', '--alpha', type=float, help="The Biot-Willis coefficient between 0 and 1 (default: 1.0).", default=1.0)
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

    nu = 0.8
    E = 0.11
    K = 0.1

    lamda = E * nu / (1 - nu**2)
    mu = 0.5 * E / (1 + nu)
    omega = 1.0 / (3*K + 0.5)

    output_dir = os.path.abspath("./results/poroelasticity/periodic")
    os.makedirs(output_dir, exist_ok=True)

    if args.levels < 0:
        nx_list = np.array([args.nx * 2**r for r in range(args.refinements+1)])
    else:
        nx_list = np.array([args.nx * (r+1) for r in range(args.refinements+1)])
    err_l2_eta = np.empty_like(nx_list, dtype=float)
    err_lInf_eta = np.empty_like(nx_list, dtype=float)
    err_l2_sigma = np.empty_like(nx_list, dtype=float)
    err_lInf_sigma = np.empty_like(nx_list, dtype=float)
    err_l2_p = np.empty_like(nx_list, dtype=float)
    err_lInf_p = np.empty_like(nx_list, dtype=float)
    err_l2_dtdiv_eta = np.empty_like(nx_list, dtype=float)
    err_lInf_dtdiv_eta = np.empty_like(nx_list, dtype=float)

    for i, nx in enumerate(nx_list):
        nx = int(nx)
        ny = nx
        nt = nx**2
        levels = args.levels
        if levels < 1:
            levels = int(np.log2(nx))
            while nx % 2**(levels-1) != 0:
                levels -= 1
            ne = int(args.elastic_steps)
        else:
            ne = max(1, round(nx * args.elastic_steps))
        coupling = args.coupling
        IO_rate = max(1, nt // 100) if args.vtk else 1

        filename = os.path.join(output_dir, f"error_E{E:.3f}_nu{nu:.3f}_K{K:.3f}_a{args.alpha:.3f}_nx{nx:03d}_ne{ne:03d}_l{levels:03d}_{coupling}{'_altgrad' if args.gradient else ''}.npz")
        try:
            data = np.load(filename)
        except:
            kwargs = {
                'lattice_elasticity': lattice_elasticity,
                'lattice_darcy': lattice_darcy,
                'elasticity_steps': ne,
                'elasticity_MG_levels': levels,
                'coupling_method': coupling,
                'alternative_gradient': args.gradient,
                'alpha': args.alpha,
                'lambda': lamda,
                'mu': mu,
                'omega': omega,
                'nx': nx,
                'ny': ny,
                'nz': 0,
                'precision': precision,
                'io_rate': IO_rate,
                'save_vtk': args.vtk,
                'print_info_rate': 1000,
                'output_dir': output_dir
            }
            sim = PeriodicPoroelasticity(**kwargs)
            sim.run(nt)
            data = { 'err_l2_eta': sim._err_l2_eta, 'err_lInf_eta': sim._err_lInf_eta,
                     'err_l2_sigma': sim._err_l2_sigma, 'err_lInf_sigma': sim._err_lInf_sigma,
                     'err_l2_p': sim._err_l2_p, 'err_lInf_p': sim._err_lInf_p,
                     'err_l2_p_grad': sim._err_l2_p_grad, 'err_lInf_p_grad': sim._err_lInf_p_grad,
                     'err_l2_dtdiv_eta': sim._err_l2_dtdiv_eta, 'err_lInf_dtdiv_eta': sim._err_lInf_dtdiv_eta,
                     'rel_err_l2_eta': np.sqrt(np.sum(sim._err_l2_eta**2) / np.sum(sim._l2_eta_exact**2)),
                     'rel_err_lInf_eta': np.max(sim._err_lInf_eta) / np.max(sim._lInf_eta_exact),
                     'rel_err_l2_sigma': np.sqrt(np.sum(sim._err_l2_sigma**2) / np.sum(sim._l2_sigma_exact**2)),
                     'rel_err_lInf_sigma': np.max(sim._err_lInf_sigma) / np.max(sim._lInf_sigma_exact),
                     'rel_err_l2_p': np.sqrt(np.sum(sim._err_l2_p**2) / np.sum(sim._l2_p_exact**2)),
                     'rel_err_lInf_p': np.max(sim._err_lInf_p) / np.max(sim._lInf_p_exact),
                     'rel_err_l2_p_grad': np.sqrt(np.sum(sim._err_l2_p_grad**2) / np.sum(sim._l2_p_grad_exact**2)),
                     'rel_err_lInf_p_grad': np.max(sim._err_lInf_p_grad) / np.max(sim._lInf_p_grad_exact),
                     'rel_err_l2_dtdiv_eta': np.sqrt(np.sum(sim._err_l2_dtdiv_eta**2) / np.sum(sim._l2_dtdiv_eta_exact**2)),
                     'rel_err_lInf_dtdiv_eta': np.max(sim._err_lInf_dtdiv_eta) / np.max(sim._lInf_dtdiv_eta_exact)
            }
            np.savez_compressed(filename, **data)

        if args.plots:
            times = np.arange(0.0, nt+1, IO_rate)
            pyplot.plot(times, data['err_l2_eta'] / nx, '-', color="#3070B3", label = 'e^k_eta')
            pyplot.plot(times, data['err_lInf_eta'], '--', color="#3070B3", label = 'Linf eta')
            pyplot.plot(times, data['err_l2_sigma'] / nx, '-', color="#EF9067", label = 'e^k_sigma')
            pyplot.plot(times, data['err_lInf_sigma'], '--', color="#EF9067", label = 'Linf sigma')
            pyplot.plot(times, data['err_l2_p'] / nx, '-', color="#A2AD00", label = 'e^k_p')
            pyplot.plot(times, data['err_lInf_p'], '--', color="#A2AD00", label = 'Linf p')
            pyplot.plot(times, data['err_l2_p_grad'] / nx, 'y-', label = 'e^k_grad_p')
            pyplot.plot(times, data['err_lInf_p_grad'], 'y--', label = 'Linf grad p')
            pyplot.plot(times, data['err_l2_dtdiv_eta'] / nx, 'k-', label = 'L2 dt_div')
            pyplot.plot(times, data['err_lInf_dtdiv_eta'], 'k--', label = 'Linf dt_div')
            pyplot.xlabel('Time Step k')
            pyplot.ylabel('Error')
            pyplot.yscale('log')
            pyplot.ylim(bottom=1e-6)
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

        err_l2_eta[i] = data['rel_err_l2_eta']
        err_lInf_eta[i] = data['rel_err_lInf_eta']
        err_l2_sigma[i] = data['rel_err_l2_sigma']
        err_lInf_sigma[i] = data['rel_err_lInf_sigma']
        err_l2_p[i] = data['rel_err_l2_p']
        err_lInf_p[i] = data['rel_err_lInf_p']
        err_l2_dtdiv_eta[i] = data['rel_err_l2_dtdiv_eta']
        err_lInf_dtdiv_eta[i] = data['rel_err_lInf_dtdiv_eta']

    if nx_list.size > 1:
        dx_list = nx_list.astype(float)**-1
        pyplot.loglog(dx_list, err_l2_eta, color="#3070B3", label = 'e_eta')
#        pyplot.loglog(dx_list, err_lInf_eta, '--', color="#3070B3", label = 'Linf eta')
        pyplot.loglog(dx_list, err_l2_sigma, color="#EF9067", label = 'e_sigma')
#        pyplot.loglog(dx_list, err_lInf_sigma, '--', color="#EF9067", label = 'Linf stress')
        pyplot.loglog(dx_list, err_l2_p, color="#A2AD00", label = 'e_p')
#        pyplot.loglog(dx_list, err_lInf_p, '--', color="#A2AD00", label = 'Linf e_p')
#        pyplot.loglog(dx_list, err_l2_dtdiv_eta, '-', c='C1', label = 'L2 dt_div')
#        pyplot.loglog(dx_list, err_lInf_dtdiv_eta, '--', c='C1', label = 'Linf dt_div')
        pyplot.loglog(dx_list, dx_list**2 / dx_list[0], 'k:', label = 'O(dx^2)')
#        pyplot.loglog(dx_list, dx_list, 'k--', label = 'O(dx)')
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
