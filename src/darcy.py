# Standard Libraries
import time

# Third-Party Libraries
import jax
import jax.numpy as jnp
import numpy as np
from termcolor import colored

# JAX-related imports
from jax import jit
from jax.experimental.multihost_utils import process_allgather

# functools imports
from functools import partial

# Local/Custom Libraries
from src.base import LBMBase
from src.boundary_conditions import BoundaryCondition, BounceBackHalfway
from src.utils import downsample_field

class AdvectionDiffusionReactionBGK(LBMBase):
    """
    Advection Diffusion Reaction (ADR) Model based on the BGK model.

    Note that `get_source` is used to get the source term.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vel = kwargs.get("vel", None)
        if self.vel is None:
            print("The advection is set to zero.")
        self.source = self.get_source()

    @partial(jit, static_argnums=(0,3), inline=True)
    def update_macroscopic(self, f, external_source=None, gradient=False):
        """
        This function computes the macroscopic variable (density) based on the distribution functions (f).

        The density is computed as the sum of the distribution functions over all lattice directions. 

        Parameters
        ----------
        f: jax.numpy.ndarray
            The distribution functions.
        external_source: jax.jax.numpy.ndarray, optional
            The external souce term
        gradient: bool, optional
            Whether to also return the gradient

        Returns
        -------
        rho: jax.numpy.ndarray
            Computed density.
        grad: jax.numpy.ndarray
            Computed density gradient, if `gradient is True`.
        """
        if self.source is not None and external_source is not None:
            rho = jnp.sum(f, axis=-1, keepdims=True) + 0.5 * (self.source + external_source)
        elif self.source is not None:
            rho = jnp.sum(f, axis=-1, keepdims=True) + 0.5 * self.source
        elif external_source is not None:
            rho = jnp.sum(f, axis=-1, keepdims=True) + 0.5 * external_source
        else:
            rho = jnp.sum(f, axis=-1, keepdims=True)

        if gradient:
            c = jnp.array(self.c, dtype=self.precisionPolicy.compute_dtype).T
            grad = jnp.dot(f, (-3*self.omega) * c)
            return rho, grad
        else:
            return rho

    @partial(jit, static_argnums=(0,), inline=True)
    def gradient(self, q, scale=1.0):
        '''
        This function performs calculation of the scaled gradient (periodic) and applies (Anti)BounceBackHalfway conditions.

        This is based on the paper:
            T. Lee and P. F. Fischer (2006)
            Eliminating parasitic currents in the lattice Boltzmann equation method for nonideal gases
            Physical Review E, https://doi.org/10.1103/PhysRevE.74.046709

        To enable multi-GPU/TPU functionality, streaming is used.

        Parameters
        ----------
        q: jax.numpy.ndarray
            The macroscopic quantity (density).
        scale: float, optional
            The scaling factor. Default is 1.

        Returns
        -------
        grad: jax.numpy.ndarray
            Computed scaled gradient of the macroscopic quantity.
        '''
        p_in = jnp.tile(q, self.lattice.q)
        p_out = self.streaming(p_in)
        for bc in self.BCs:
            if bc.name == 'ADR_Neumann_BounceBack' or bc.name == 'ADR_AntiBounceBack':
                p_out = p_out.at[bc.indices].set(bc.apply(p_out, p_in))
        return jnp.matmul(p_out, -3 * scale * self.w[..., None] * self.c.T)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f, external_source=None):
        """
        BGK collision step for lattice.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        rho = self.update_macroscopic(f, external_source)

        if self.vel is None:
            feq = rho * self.w
        else:
            feq = self.equilibrium(rho, self.vel, cast_output=False)

        fneq = f - feq
        fout = f - self.omega * fneq

        if self.source is not None and external_source is not None:
            fout = fout + (self.source + external_source) * ((1 - 0.5 * self.omega) * self.w)
        elif self.source is not None:
            fout = fout + self.source * ((1 - 0.5 * self.omega) * self.w)
        elif external_source is not None:
            fout = fout + external_source * ((1 - 0.5 * self.omega) * self.w)

        return self.precisionPolicy.cast_to_output(fout)

    @partial(jit, static_argnums=(0, 3), donate_argnums=(1,))
    def step(self, f_poststreaming, timestep, return_fpost=False, external_source=None):
        """
        This function performs a single step of the LBM simulation.

        It first performs the collision step, which is the relaxation of the distribution functions 
        towards their equilibrium values. It then applies the respective boundary conditions to the 
        post-collision distribution functions.

        The function then performs the streaming step, which is the propagation of the distribution 
        functions in the lattice. It then applies the respective boundary conditions to the post-streaming 
        distribution functions.

        Parameters
        ----------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions.
        timestep: int
            The current timestep of the simulation.
        return_fpost: bool, optional
            If True, the function also returns the post-collision distribution functions.
        external_source: jax.numpy.ndarray, optional
            The external source term.

        Returns
        -------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions after the simulation step.
        f_postcollision: jax.numpy.ndarray or None
            The post-collision distribution functions after the simulation step, or None if 
            return_fpost is False.
        """
        f_postcollision = self.collision(f_poststreaming, external_source)
        f_postcollision = self.apply_bc(f_postcollision, f_poststreaming, timestep, "PostCollision")
        f_poststreaming = self.streaming(f_postcollision)
        f_poststreaming = self.apply_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming")

        if return_fpost:
            return f_poststreaming, f_postcollision
        else:
            return f_poststreaming, None

    def run(self, t_max):
        """
        This function runs the LBM simulation for a specified number of time steps.

        It first initializes the distribution functions and then enters a loop where it performs the 
        simulation steps (collision, streaming, and boundary conditions) for each time step.

        The function can also print the progress of the simulation, save the simulation data, and 
        compute the performance of the simulation in million lattice updates per second (MLUPS).

        Parameters
        ----------
        t_max: int
            The total number of time steps to run the simulation.
        Returns
        -------
        f: jax.numpy.ndarray
            The distribution functions after the simulation.
        """
        f = self.assign_fields_sharded()
        start_step = 0

        if self.computeMLUPS:
            start = time.time()

        if self.ioRate > 0 and start_step % self.ioRate == 0:
            # Save the simulation data
            print(f"Saving data at timestep {start_step}/{t_max}")
            rho = downsample_field(self.update_macroscopic(f), self.downsamplingFactor)
            # Gather the data from all processes and convert it to numpy arrays (move to host memory)
            rho = process_allgather(rho)
            # Save the data
            self.handle_io_timestep(start_step, f, None, rho, None)

        # Loop over all time steps
        for timestep in range(start_step + 1, t_max + 1):
            io_flag = self.ioRate > 0 and (timestep % self.ioRate == 0 or timestep == t_max)
            print_iter_flag = self.printInfoRate> 0 and timestep % self.printInfoRate== 0

            if io_flag:
                # Update the macroscopic variable and save the previous values (for error computation)
                rho_prev = downsample_field(self.update_macroscopic(f), self.downsamplingFactor)
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho_prev = process_allgather(rho_prev)

            # Perform one time-step (collision, streaming, and boundary conditions)
            f, fstar = self.step(f, timestep, return_fpost=self.returnFpost)
            # Print the progress of the simulation
            if print_iter_flag:
                print(colored("Timestep ", 'blue') + colored(f"{timestep}", 'green') + colored(" of ", 'blue') + colored(f"{t_max}", 'green') + colored(" completed", 'blue'))

            if io_flag:
                # Save the simulation data
                print(f"Saving data at timestep {timestep}/{t_max}")
                rho = downsample_field(self.update_macroscopic(f), self.downsamplingFactor)
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho = process_allgather(rho)
                # Save the data
                self.handle_io_timestep(timestep, f, fstar, rho, rho_prev)

            # Start the timer for the MLUPS computation after the first timestep (to remove compilation overhead)
            if self.computeMLUPS and timestep == 1:
                jax.block_until_ready(f)
                start = time.time()

        if self.computeMLUPS:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(f)
            end = time.time()
            if self.dim == 2:
                print(colored("Domain: ", 'blue') + colored(f"{self.nx} x {self.ny}", 'green') if self.dim == 2 else colored(f"{self.nx} x {self.ny} x {self.nz}", 'green'))
                print(colored("Number of voxels: ", 'blue') + colored(f"{self.nx * self.ny}", 'green') if self.dim == 2 else colored(f"{self.nx * self.ny * self.nz}", 'green'))
                print(colored("MLUPS: ", 'blue') + colored(f"{self.nx * self.ny * t_max / (end - start) / 1e6}", 'red'))

            elif self.dim == 3:
                print(colored("Domain: ", 'blue') + colored(f"{self.nx} x {self.ny} x {self.nz}", 'green'))
                print(colored("Number of voxels: ", 'blue') + colored(f"{self.nx * self.ny * self.nz}", 'green'))
                print(colored("MLUPS: ", 'blue') + colored(f"{self.nx * self.ny * self.nz * t_max / (end - start) / 1e6}", 'red'))

        return f

    def handle_io_timestep(self, timestep, f, fstar, rho, rho_prev):
        """
        This function handles the input/output (I/O) operations at each time step of the simulation.

        It prepares the data to be saved and calls the output_data function, which can be overwritten 
        by the user to customize the I/O operations.

        Parameters
        ----------
        timestep: int
            The current time step of the simulation.
        f: jax.numpy.ndarray
            The post-streaming distribution functions at the current time step.
        fstar: jax.numpy.ndarray
            The post-collision distribution functions at the current time step.
        rho: jax.numpy.ndarray
            The macroscopic (density) field at the current time step.
        """
        kwargs = {
            "timestep": timestep,
            "rho": rho,
            "rho_prev": rho_prev,
            "f_poststreaming": f,
            "f_postcollision": fstar
        }
        self.output_data(**kwargs)

    def get_source(self):
        """
        This function computes the source to be applied in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the source according to the specific 
        problem being solved.

        By default, it does nothing and returns None. When overwritten, it could implement a constant 
        source term.

        Returns
        -------
        source: jax.numpy.ndarray
            The source to be applied.
        """
        pass

class ADR_AntiBounceBack(BoundaryCondition):
    """
    Halfway anti-bounce-back (Dirichlet) boundary condition for an ADR LBM simulation.

    This class implements a halfway anti-bounce-back boundary condition. The boundary condition is applied after
    the streaming step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "DiffusionAntiBounceBack".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostStreaming".
    needsExtraConfiguration : bool
        Whether the boundary condition needs extra configuration before it can be applied. For this class, it is True.
    isSolid : bool
        Whether the boundary condition represents a solid boundary. For this class, it is True.
    rho : array-like, optional
        The prescribed value of the macroscopic variable (density) for the boundary condition. Zero BC is assumed if rho=None (default).
    """
    def __init__(self, indices, gridInfo, precision_policy, rho=None):
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "ADR_AntiBounceBack"
        self.implementationStep = "PostStreaming"
        self.needsExtraConfiguration = True
        self.isSolid = True
        self.rho = rho

    def configure(self, boundaryMask):
        """
        Configures the boundary condition.

        Parameters
        ----------
        boundaryMask : array-like
            The grid mask for the boundary voxels.

        Returns
        -------
        None

        Notes
        -----
        This method performs an index shift for the halfway anti-bounce-back boundary condition. It updates
        the indices of the boundary nodes to be the indices of fluid nodes adjacent of the solid nodes.
        """
        # Perform index shift for halfway BB.
        hasFluidNeighbour = ~boundaryMask[:, self.lattice.opp_indices]
        nbd_orig = len(self.indices[0])
        idx = np.array(self.indices).T
        idx_trg = []
        for i in range(self.lattice.q):
            idx_trg.append(idx[hasFluidNeighbour[:, i], :] + self.lattice.c[:, i])
        indices_new, array_indices = np.unique(np.vstack(idx_trg), axis=0, return_index=True)
        self.indices = tuple(indices_new.T)
        nbd_modified = len(self.indices[0])
        if (nbd_orig != nbd_modified) and self.rho is not None:
            if isinstance(self.rho, float) or self.rho.size == 1:
                self.rho = jnp.full((nbd_modified, 1), self.rho)
            else:
                print("WARNING: Assuming a correct macroscopic (density) vector to impose at the BC cells!")
                self.rho = self.rho[array_indices]

    @partial(jit, static_argnums=(0,))
    def impose_boundary_rho(self, fbd, bindex):
        tmp = 2 * self.lattice.w * self.rho
        fbd = fbd.at[bindex, self.imissing].add(tmp[bindex, self.iknown])
        return fbd

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the halfway bounce-back boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.
        """
        nbd = len(self.indices[0])
        bindex = np.arange(nbd)[:, None]
        fbd = fout[self.indices]

        fbd = fbd.at[bindex, self.imissing].set(-fin[self.indices][bindex, self.iknown])
        if self.rho is not None:
            fbd = self.impose_boundary_rho(fbd, bindex)
        return fbd

class ADR_Neumann_BounceBack(BounceBackHalfway):
    """
    Halfway bounce-back (Neumann) boundary condition for an ADR LBM simulation.

    This class implements a halfway bounce-back boundary condition. The boundary condition is applied after
    the streaming step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "ADR_Neumann_BounceBack".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostStreaming".
    needsExtraConfiguration : bool
        Whether the boundary condition needs extra configuration before it can be applied. For this class, it is True.
    isSolid : bool
        Whether the boundary condition represents a solid boundary. For this class, it is True.
    """
    def __init__(self, indices, gridInfo, precision_policy):
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "ADR_Neumann_BounceBack"

class ADR_EquilibriumBC(BoundaryCondition):
    """
    Equilibrium (Dirichlet) boundary condition for an ADR LBM simulation.

    This class implements an equilibrium boundary condition, where the distribution function at the boundary nodes is
    set to the equilibrium distribution function. The boundary condition is applied after the streaming step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "EquilibriumBC".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostStreaming".
    out : jax.numpy.ndarray
        The equilibrium distribution function at the boundary nodes.
    """

    def __init__(self, indices, gridInfo, precision_policy, rho, vel=None):
        super().__init__(indices, gridInfo, precision_policy)
        if vel is None:
            self.out = self.precisionPolicy.cast_to_output(rho * self.lattice.w)
        else:
            self.out = self.precisionPolicy.cast_to_output(self.equilibrium(rho, vel))
        self.name = "ADR_EquilibriumBC"
        self.implementationStep = "PostStreaming"

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the equilibrium boundary condition.

        Parameters
        ----------
        fout : jax.numpy.ndarray
            The output distribution functions.
        fin : jax.numpy.ndarray
            The input distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The modified output distribution functions after applying the boundary condition.

        Notes
        -----
        This method applies the equilibrium boundary condition by setting the output distribution functions at the
        boundary nodes to the equilibrium distribution function.
        """
        return self.out
