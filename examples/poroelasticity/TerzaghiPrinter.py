"""
This example plots results of quasi-2D poroelastic flow simulation (Terzaghi problem) using LBM.

The flow is modeled by the Darcy equation.
"""
import argparse
import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot
from tikzplotlib import save as saveTIKZ
import os

def plot_data(ax, column, grid_nx, grid_ne, grid, data, title, logarithmic):
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
        if logarithmic or grid_ne[-1,-1] > 10:
            ax[0].set_ylabel('N_E')
        else:
            ax[0].set_ylabel('N_E / N_x')

    plt = ax[column].scatter(x, y, c=np.log10(data), cmap=cmap, vmin=-2.5)
    vmin, vmax = plt.get_clim()
    if vmax > err_max:
        extend = 'max'
    else:
        extend = 'neither'
    vmax = err_max
    plt.set_clim(vmax=vmax)
    levels = np.arange(-2.75, vmax, 0.25)
    ex_levels = np.arange(-2.5, 0, 0.5)
    levels = np.setdiff1d(levels, ex_levels)

    ax[column].imshow(grid.T, extent=bounds, origin='lower', cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    ax[column].contour(grid_nx, grid_ne, grid, levels, colors='k')
    CS = ax[column].contour(grid_nx, grid_ne, grid, ex_levels, colors='k')
    ax[column].clabel(CS, fontsize=14)

    fig.colorbar(plt, ax=ax[column], shrink=0.8, extend=extend)
    ax[column].set_title(title)
    ax[column].set_aspect('auto')
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
    parser.add_argument('-r', '--relative', help="Whether to show the relative numbers in figures (default: False).", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-MG', '--multigrid', help="Whether to show multigrid results (default: False).", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-s', '--save_fig', help="Whether to save the plots as figures (default: False).", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    output_dir = os.path.abspath("./results/poroelasticity/Terzaghi")

    nne = np.arange(args.elast_minimum, args.elast_maximum+1)
    grid_nx, grid_ne = np.meshgrid(np.arange(args.minimum, args.maximum+1), (nne / args.maximum) if args.relative else nne, indexing='ij')
    coupling = args.coupling

    points = []
    err_l2_sub = []
    err_l2_p = []

    for nx, ne in zip(grid_nx.flat, grid_ne.flat):
        if args.relative:
            ne = max(1, round(nx * ne))
            if [nx, ne/nx] in points: continue

        levels = 1
        if args.multigrid:
            levels = int(np.log2(nx))
            while nx % 2**(levels-1) != 0:
                levels -= 1

        filename = os.path.join(output_dir, f"error_a{args.alpha:.3f}_nx{nx:03d}_ne{ne:03d}_l{levels:03d}_{coupling}{'_altgrad' if args.gradient else ''}.npz")
        if os.path.exists(filename):
            with np.load(filename) as data:
                try:
                    err_l2_sub.append(data['rel_err_sub'])
                    err_l2_p.append(data['rel_err_pres'])
                    if args.relative:
                        points.append([nx, ne/nx])
                    else:
                        points.append([nx, ne])
                except:
                    print(f'Could not read file "{filename}".')
                    N = len(points)
                    if len(err_l2_sub) > N: err_l2_sub.pop()
                    if len(err_l2_p) > N: err_l2_p.pop()

    if len(points) == 0:
        print("No data could be found.")
    else:
        x = [x for x,y in points]
        y = [y for x,y in points]
        grid_l2_sub = griddata(points, np.log10(err_l2_sub), (grid_nx, grid_ne))
        grid_l2_p = griddata(points, np.log10(err_l2_p), (grid_nx, grid_ne))

        fig, ax = pyplot.subplots(1, 2, figsize=(15, 6), sharex='all', sharey='all', layout='constrained')
        plot_data(ax, 0, grid_nx, grid_ne, grid_l2_sub, err_l2_sub, 'L2 subsidence', args.multigrid)
        plot_data(ax, 1, grid_nx, grid_ne, grid_l2_p, err_l2_p, 'L2 pressure', args.multigrid)

        if args.save_fig:
            filename = os.path.join(output_dir, f"error_a{args.alpha:.3f}_{coupling}{'_MG' if args.multigrid else ''}{'_altgrad' if args.gradient else ''}")
            pyplot.savefig(filename + ".png")
            saveTIKZ(filename + ".tex")
        pyplot.show()
