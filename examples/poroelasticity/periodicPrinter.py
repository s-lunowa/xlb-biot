"""
This example plots results of 2D periodic poroelastic flow simulations using the lattice Boltzmann method (LBM).

The flow is modeled by the Darcy equation.
"""
import argparse
import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot
from tikzplotlib import save as saveTIKZ
import os

def plot_data(ax, column, grid_nx, grid_ne, grid, data, title, relative, logarithmic, row=None):
    if row is not None:
        ax = ax[row]

#    cmap = 'plasma'
    cmap = 'viridis'
    alpha = 0.9
    err_max = 0
    dne = (grid_ne[-1,-1] - grid_ne[0,0]) * 0.005
    if logarithmic:
        bounds = (grid_nx[0,0]*0.95, grid_nx[-1,-1]/0.95, grid_ne[0,0]-dne, grid_ne[-1,-1]+dne)
    else:
        bounds = (grid_nx[0,0]-0.5, grid_nx[-1,-1]+0.5, grid_ne[0,0]-dne, grid_ne[-1,-1]+dne)
    if column == 0:
        if logarithmic:
            ax[0].set_xlim((bounds[0]*0.95, bounds[1]/0.95))
            ax[0].set_xscale('log', base=2)
        else:
            ax[0].set_xlim((bounds[0]-0.5, bounds[1]+0.5))
        ax[0].set_ylim((bounds[2]-dne, bounds[3]+dne))
        ax[0].set_ylabel('N_E / N_x' if relative else 'N_E')

    plt = ax[column].scatter(x, y, c=np.log10(data), cmap=cmap, vmin=-3.5)
    vmin, vmax = plt.get_clim()
    if vmax > err_max:
        extend = 'max'
    else:
        extend = 'neither'
    vmax = err_max
    plt.set_clim(vmax=vmax)
    levels = np.arange(-3.5, vmax, 0.25)
    ex_levels = np.arange(-3, 0, 0.5)
    levels = np.setdiff1d(levels, ex_levels)

    ax[column].imshow(grid.T, extent=bounds, origin='lower', cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    ax[column].contour(grid_nx, grid_ne, grid, levels, colors='k')
    CS = ax[column].contour(grid_nx, grid_ne, grid, ex_levels, colors='k')
    ax[column].clabel(CS, fontsize=14)

    fig.colorbar(plt, ax=ax[column], shrink=0.8, extend=extend)
    ax[column].set_title(title)
    ax[column].set_aspect('auto')

    if row is None or row == 1:
        ax[column].set_xlabel('N_x')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2D periodic poroelastic simulation using LBM".')
    parser.add_argument('-a', '--alpha', type=float, help="The Biot-Willis coefficient between 0 and 1 (default: 1.0).", default=1.0)
    parser.add_argument('-min', '--minimum', type=int, help="The minimal number of cells in x-direction (default: 10).", default=10)
    parser.add_argument('-max', '--maximum', type=int, help="The maximal number of cells in x-direction (default: 100).", default=100)
    parser.add_argument('-emin', '--elast_minimum', type=int, help="The minimal number of elasticity steps (default: 1).", default=1)
    parser.add_argument('-emax', '--elast_maximum', type=int, help="The maximal number of elasticity steps (default: 100).", default=100)
    parser.add_argument('-c', '--coupling', type=str, help="The coupling method (default: 'centered').", default='centered', choices=['explicit', 'implicit', 'centered'])
    parser.add_argument('-g', '--gradient', help="Whether to use the alternative gradient (default: False).", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-inf', '--inf', help="Whether to show L^inf error plots (default: False).", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-ddt', '--ddt_div_eta', help="Whether to show d/dt div eta error plots (default: False).", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-r', '--relative', help="Whether to show the relative numbers in figures (default: False).", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-MG', '--multigrid', help="Whether to show multigrid results (default: False).", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-s', '--save_fig', help="Whether to save the plots as figures (default: False).", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    nu = 0.8
    E = 0.11
    K = 0.1
    output_dir = os.path.abspath("./results/poroelasticity/periodic")

    nne = np.arange(args.elast_minimum, args.elast_maximum+1)
    grid_nx, grid_ne = np.meshgrid(np.arange(args.minimum, args.maximum+1), (nne / args.maximum) if args.relative else nne, indexing='ij')
    coupling = args.coupling

    points = []
    err_l2_eta = []
    err_l2_sigma = []
    err_l2_p = []
    err_l2_dtdiv_eta = []
    err_lInf_eta = []
    err_lInf_sigma = []
    err_lInf_p = []
    err_lInf_dtdiv_eta = []

    for nx, ne in zip(grid_nx.flat, grid_ne.flat):
        if args.relative:
            ne = max(1, round(nx * ne))
            if [nx, ne/nx] in points: continue

        levels = 1
        if args.multigrid:
            levels = int(np.log2(nx))
            while nx % 2**(levels-1) != 0:
                levels -= 1

        filename = os.path.join(output_dir, f"error_E{E:.3f}_nu{nu:.3f}_K{K:.3f}_a{args.alpha:.3f}_nx{nx:03d}_ne{ne:03d}_l{levels:03d}_{coupling}{'_altgrad' if args.gradient else ''}.npz")
        if os.path.exists(filename):
            with np.load(filename) as data:
                try:
                    err_l2_eta.append(data['rel_err_l2_eta'])
                    err_l2_sigma.append(data['rel_err_l2_sigma'])
                    err_l2_p.append(data['rel_err_l2_p'])
                    if args.ddt_div_eta:
                        err_l2_dtdiv_eta.append(data['rel_err_l2_dtdiv_eta'])
                    if args.inf:
                        err_lInf_eta.append(data['rel_err_lInf_eta'])
                        err_lInf_sigma.append(data['rel_err_lInf_sigma'])
                        err_lInf_p.append(data['rel_err_lInf_p'])
                        if args.ddt_div_eta:
                            err_lInf_dtdiv_eta.append(data['rel_err_lInf_dtdiv_eta'])
                    if args.relative:
                        points.append([nx, ne/nx])
                    else:
                        points.append([nx, ne])
                except:
                    print(f'Could not read file "{filename}".')
                    N = len(points)
                    if len(err_l2_eta) > N: err_l2_eta.pop()
                    if len(err_l2_sigma) > N: err_l2_sigma.pop()
                    if len(err_l2_p) > N: err_l2_p.pop()
                    if len(err_l2_dtdiv_eta) > N: err_l2_dtdiv_eta.pop()
                    if len(err_lInf_eta) > N: err_lInf_eta.pop()
                    if len(err_lInf_sigma) > N: err_lInf_sigma.pop()
                    if len(err_lInf_p) > N: err_lInf_p.pop()
                    if len(err_lInf_dtdiv_eta) > N: err_lInf_dtdiv_eta.pop()

    if len(points) == 0:
        print("No data could be found.")
    else:
        x = [x for x,y in points]
        y = [y for x,y in points]
        grid_l2_eta = griddata(points, np.log10(err_l2_eta), (grid_nx, grid_ne))
        grid_l2_sigma = griddata(points, np.log10(err_l2_sigma), (grid_nx, grid_ne))
        grid_l2_p = griddata(points, np.log10(err_l2_p), (grid_nx, grid_ne))
        if args.ddt_div_eta:
            grid_l2_dtdiv_eta = griddata(points, np.log10(err_l2_dtdiv_eta), (grid_nx, grid_ne))
        if args.inf:
            grid_lInf_eta = griddata(points, np.log10(err_lInf_eta), (grid_nx, grid_ne))
            grid_lInf_sigma = griddata(points, np.log10(err_lInf_sigma), (grid_nx, grid_ne))
            grid_lInf_p = griddata(points, np.log10(err_lInf_p), (grid_nx, grid_ne))
            if args.ddt_div_eta:
                grid_lInf_dtdiv_eta = griddata(points, np.log10(err_lInf_dtdiv_eta), (grid_nx, grid_ne))

        if args.inf:
            if args.ddt_div_eta:
                fig, ax = pyplot.subplots(2, 4, figsize=(20, 10), sharex='all', sharey='all', layout='constrained')
            else:
                fig, ax = pyplot.subplots(2, 3, figsize=(18, 10), sharex='all', sharey='all', layout='constrained')

            plot_data(ax, 0, grid_nx, grid_ne, grid_l2_eta, err_l2_eta, 'L^2 displacement', args.relative, args.multigrid, row=0)
            plot_data(ax, 1, grid_nx, grid_ne, grid_l2_sigma, err_l2_sigma, 'L^2 stress', args.relative, args.multigrid, row=0)
            plot_data(ax, 2, grid_nx, grid_ne, grid_l2_p, err_l2_p, 'L^2 pressure', args.relative, args.multigrid, row=0)
            if args.ddt_div_eta:
                plot_data(ax, 3, grid_nx, grid_ne, grid_l2_dtdiv_eta, err_l2_dtdiv_eta, 'L^2 d/dt div eta', args.relative, args.multigrid, row=0)

            plot_data(ax, 0, grid_nx, grid_ne, grid_lInf_eta, err_lInf_eta, 'L^inf displacement', args.relative, args.multigrid, row=1)
            plot_data(ax, 1, grid_nx, grid_ne, grid_lInf_sigma, err_lInf_sigma, 'L^inf stress', args.relative, args.multigrid, row=1)
            plot_data(ax, 2, grid_nx, grid_ne, grid_lInf_p, err_lInf_p, 'L^inf pressure', args.relative, args.multigrid, row=1)
            if args.ddt_div_eta:
                plot_data(ax, 3, grid_nx, grid_ne, grid_lInf_dtdiv_eta, err_lInf_dtdiv_eta, 'L^inf d/dt div eta', args.relative, args.multigrid, row=1)
        else:
            if args.ddt_div_eta:
                fig, ax = pyplot.subplots(1, 4, figsize=(20, 5), sharex='all', sharey='all', layout='constrained')
            else:
                fig, ax = pyplot.subplots(1, 3, figsize=(20, 6), sharex='all', sharey='all', layout='constrained')
            plot_data(ax, 0, grid_nx, grid_ne, grid_l2_eta, err_l2_eta, 'L^2 displacement', args.relative, args.multigrid)
            plot_data(ax, 1, grid_nx, grid_ne, grid_l2_sigma, err_l2_sigma, 'L^2 stress', args.relative, args.multigrid)
            plot_data(ax, 2, grid_nx, grid_ne, grid_l2_p, err_l2_p, 'L^2 pressure', args.relative, args.multigrid)
            if args.ddt_div_eta:
                plot_data(ax, 3, grid_nx, grid_ne, grid_l2_dtdiv_eta, err_l2_dtdiv_eta, 'L^2 d/dt div eta', args.relative, args.multigrid)

    if args.save_fig:
        filename = os.path.join(output_dir, f"error_a{args.alpha:.3f}_{coupling}{'_MG' if args.multigrid else ''}{'_altgrad' if args.gradient else ''}")
        pyplot.savefig(filename + ".png")
        saveTIKZ(filename + ".tex")
    pyplot.show()
