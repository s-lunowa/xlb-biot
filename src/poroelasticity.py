# Standard Libraries
import os
import time

# Third-Party Libraries
import jax
from termcolor import colored

# JAX-related imports
from jax import jit
from jax.experimental.multihost_utils import process_allgather

# functools imports
from functools import partial

# Local/Custom Libraries
from src.utils import downsample_field

jax.config.update("jax_spmd_mode", 'allow_all')
# Disables annoying TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LBMPoroelasticity(object):
    """
    LBMPoroelasticity: A class that represents a base for Lattice Boltzmann Method simulations of poroelasticity.

    The flow is modeled directly by Darcy equations.

    Parameters
    ----------
        LBMElasticity (Class type): the class type implementing the elasticity LBM with IC and BCs.
        LBMDarcy (Class type): the class type implementing the Darcy LBM with IC and BCs.
        lattice_elasticity (object): The lattice object for elasticity that contains the lattice structure and weights.
        lattice_darcy (object): The lattice object for Darcy flow that contains the lattice structure and weights.
        lambda (float): The first Lame parameter for the elasticity LBM simulation.
        mu (float): The second Lame parameter for the elasticity LBM simulation.
        omega (float): The relaxation parameter for the Darcy flow LBM simulation.
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        nz (int, optional): Number of grid points in the z-direction. Defaults to 0.
        precision (str, optional): A string specifying the precision used for the simulation. Defaults to "f32/f32".
    """

    def __init__(self, LBMElasticity, LBMDarcy, **kwargs):
        kwargs_elasticity = kwargs.copy()
        kwargs_elasticity['lattice'] = kwargs.pop('lattice_elasticity')
        kwargs_elasticity['MG_levels'] = kwargs.pop('elasticity_MG_levels', 1)
        self.elasticity = LBMElasticity(**kwargs_elasticity)
        kwargs['lattice'] = kwargs.pop('lattice_darcy')
        self.darcy = LBMDarcy(**kwargs)

        self.elasticitySteps = kwargs.get('elasticity_steps', 1)
        self.alpha = kwargs.get('alpha')
        self.epsilon = kwargs.get('epsilon', 1/self.elasticity.nx)
        self.couplingMethod = kwargs.get('coupling_method', 'centered')
        self.alternative_gradient = kwargs.get('alternative_gradient', False)

        self.downsamplingFactor = kwargs.get("downsampling_factor", 1)
        self.printInfoRate = kwargs.get("print_info_rate", 100)
        self.ioRate = kwargs.get("io_rate", 0)
        self.computeMLUPS = kwargs.get("compute_MLUPS", False)

        if self.computeMLUPS:
            self.ioRate = 0
            self.printInfoRate = 0

        self.useDynamicForce = self.get_elastic_force(0) is not None
        self.useDynamicSource = self.get_darcy_source(0) is not None

        self.show_simulation_parameters()

    @property
    def elasticitySteps(self):
        return self._elasticitySteps

    @elasticitySteps.setter
    def elasticitySteps(self, value):
        if not isinstance(value, int) or value < 1:
            raise TypeError("elasticitySteps must be a positive integer.")
        self._elasticitySteps = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not isinstance(value, float) or value < 0 or value > 1:
            raise TypeError("alpha must be a float between 0 and 1.")
        self._alpha = value

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if not isinstance(value, float) or value <= 0:
            raise TypeError("epsilon must be a positive float.")
        self._epsilon = value

    @property
    def alternative_gradient(self):
        return self._alternative_gradient

    @alternative_gradient.setter
    def alternative_gradient(self, value):
        if not isinstance(value, bool):
            raise TypeError("alternative_gradient must be a boolean.")
        self._alternative_gradient = value

    @property
    def couplingMethod(self):
        return self._couplingMethod

    @couplingMethod.setter
    def couplingMethod(self, value):
        if value == 'explicit':
            self.couplingRatio = 0.0
        elif value == 'implicit':
            self.couplingRatio = 1.0
        elif value == 'centered':
            self.couplingRatio = 0.5
        else:
            raise TypeError("couplingMethod must be 'explicit', 'implicit', 'centered', or 'alt_implicit', 'alt_centered'.")
        self._couplingMethod = value

    @property
    def couplingRatio(self):
        return self._couplingRatio

    @couplingRatio.setter
    def couplingRatio(self, value):
        if not isinstance(value, float) or value < 0.0 or value > 1.0:
            raise TypeError("couplingRatio must be a float beween 0 and 1")
        self._couplingRatio = value

    @property
    def downsamplingFactor(self):
        return self._downsamplingFactor

    @downsamplingFactor.setter
    def downsamplingFactor(self, value):
        if not isinstance(value, int):
            raise TypeError("downsamplingFactor must be an integer")
        self._downsamplingFactor = value

    @property
    def printInfoRate(self):
        return self._printInfoRate

    @printInfoRate.setter
    def printInfoRate(self, value):
        if not isinstance(value, int):
            raise TypeError("printInfoRate must be an integer")
        self._printInfoRate = value

    @property
    def ioRate(self):
        return self._ioRate

    @ioRate.setter
    def ioRate(self, value):
        if not isinstance(value, int):
            raise TypeError("ioRate must be an integer")
        self._ioRate = value

    @property
    def computeMLUPS(self):
        return self._computeMLUPS

    @computeMLUPS.setter
    def computeMLUPS(self, value):
        if not isinstance(value, bool):
            raise TypeError("computeMLUPS must be a boolean")
        self._computeMLUPS = value

    def show_simulation_parameters(self):
        attributes_to_show = [
            'alpha', 'couplingMethod', 'elasticitySteps', 'alternative_gradient',
            'downsamplingFactor', 'printInfoRate', 'ioRate', 'computeMLUPS'
        ]

        descriptive_names = {
            'alpha': 'Biot-Willis Constant',
            'couplingMethod': 'Coupling Method',
            'elasticitySteps': 'Elasticity Steps',
            'alternative_gradient': 'Alternative Gradient',
            'downsamplingFactor': 'Downsampling Factor',
            'printInfoRate': 'Print Info Rate',
            'ioRate': 'I/O Rate',
            'computeMLUPS': 'Compute MLUPS',
        }
        simulation_name = self.__class__.__name__

        print(colored(f'**** Simulation Parameters for {simulation_name} ****', 'green'))

        header = f"{colored('Parameter', 'blue'):>30} | {colored('Value', 'yellow')}"
        print(header)
        print('-' * 50)

        for attr in attributes_to_show:
            value = getattr(self, attr, 'Attribute not set')
            descriptive_name = descriptive_names.get(attr, attr)  # Use the attribute name as a fallback
            row = f"{colored(descriptive_name, 'blue'):>30} | {colored(value, 'yellow')}"
            print(row)

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
        f_elasticity: jax.numpy.ndarray
            The distribution functions for elasticity after the simulation.
        f: jax.numpy.ndarray
            The distribution functions for Darcy flow after the simulation.
        """
        start_step = 0
        dt_div_eta = self.initialize_macroscopic_fields()
        if dt_div_eta is None:
            dt_div_eta = self.darcy.distributed_array_init((self.darcy.nx, self.darcy.ny, 1), self.darcy.precisionPolicy.compute_dtype, init_val=0.0)

        f_darcy = self.darcy.assign_fields_sharded()
        source = (-self.alpha/self.epsilon) * dt_div_eta
        if self.useDynamicSource:
            source += self.get_darcy_source(start_step)
        f_darcy = f_darcy - 0.5 * source * self.darcy.w
        pressure = self.darcy.update_macroscopic(f_darcy, source)

        f_elasticity = self.elasticity.assign_fields_sharded()
        if self.alternative_gradient:
            force = self.darcy.gradient(pressure, -self.alpha * self.epsilon)
        else:
            _, pressure_grad = self.darcy.update_macroscopic(f_darcy, gradient=True)
            pressure_grad += self.darcy.gradient(source, 0.5)
            force = (-self.alpha * self.epsilon) * pressure_grad
        if self.useDynamicForce:
            force += self.get_elastic_force(start_step)
        eta, sigma = self.elasticity.initialize_macroscopic_fields()
        if eta is not None and sigma is not None:
            f_elasticity = self.elasticity.equilibrium(eta, sigma, external_force=force)

        if self.computeMLUPS:
            start = time.time()

        # Loop over all time steps
        for timestep in range(start_step, t_max + 1):
            io_flag = self.ioRate > 0 and (timestep % self.ioRate == 0 or timestep == t_max)
            print_iter_flag = self.printInfoRate> 0 and timestep % self.printInfoRate== 0

            # Perform one time-step (collision, streaming, and boundary conditions) with exchange
            f_elasticity, f_darcy, eta, sigma, dt_div_eta, pressure, pressure_grad = self.step(f_elasticity, f_darcy, dt_div_eta, timestep)

            # Print the progress of the simulation
            if print_iter_flag:
                print(colored("Timestep ", 'blue') + colored(f"{timestep}", 'green') + colored(" of ", 'blue') + colored(f"{t_max}", 'green') + colored(" completed", 'blue'))

            if io_flag:
                # Save the simulation data
                print(f"Saving data at timestep {timestep}/{t_max}")
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                eta = process_allgather(downsample_field(eta, self.downsamplingFactor))
                sigma = process_allgather(downsample_field(sigma, self.downsamplingFactor))
                pressure = process_allgather(downsample_field(pressure, self.downsamplingFactor))
                pressure_grad = process_allgather(downsample_field(pressure_grad, self.downsamplingFactor))
                dt_div_eta_ = process_allgather(downsample_field(dt_div_eta, self.downsamplingFactor))

                # Save the data
                self.handle_io_timestep(timestep, f_elasticity, f_darcy, eta, sigma, pressure, pressure_grad, dt_div_eta_)

            # Start the timer for the MLUPS computation after the first timestep (to remove compilation overhead)
            if self.computeMLUPS and timestep == 1:
                jax.block_until_ready(f_elasticity)
                jax.block_until_ready(f_darcy)
                start = time.time()

        if self.computeMLUPS:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(f_elasticity)
            jax.block_until_ready(f_darcy)
            end = time.time()
            nx = self.elasticity.nx
            ny = self.elasticity.ny
            nz = self.elasticity.nz
            dim = self.elasticity.dim
            print(colored("Domain: ", 'blue') + (colored(f"{nx} x {ny}", 'green') if dim == 2 else colored(f"{nx} x {ny} x {nz}", 'green')))
            print(colored("Number of voxels: ", 'blue') + (colored(f"{nx * ny}", 'green') if dim == 2 else colored(f"{nx * ny * nz}", 'green')))
            print(colored("MLUPS: ", 'blue') + (colored(f"{nx * ny * t_max / (end - start) / 1e6}", 'red') if dim == 2 else colored(f"{nx * ny * nz * t_max / (end - start) / 1e6}", 'red')))

        return f_elasticity, f_darcy

    #@partial(jit, static_argnums=(0,), donate_argnums=(1,2,3))
    def step(self, f_elasticity, f_darcy, dt_div_eta, timestep):
        """
        This function performs a single step of the LBM simulation.

        It couples elasticity to darcy flow partially explicityly and implicitly (direct).
        """
        fprev_elasticity = f_elasticity.copy()
        source_base = ((1.0 - self.couplingRatio) * -self.alpha / self.epsilon) * dt_div_eta
        if self.useDynamicSource:
            source_base += self.get_darcy_source(timestep)

        if self.alternative_gradient:
            pressure_grad_base = self.darcy.gradient(self.darcy.update_macroscopic(f_darcy, source_base))
        else:
            _, pressure_grad_base = self.darcy.update_macroscopic(f_darcy, gradient=True)
            pressure_grad_base += self.darcy.gradient(source_base, 0.5)

        force = (-self.alpha * self.epsilon) * pressure_grad_base - self.darcy.gradient(self.elasticity.update_divergence(fprev_elasticity), 0.5 * self.couplingRatio * self.alpha**2)
        if self.useDynamicForce:
            force += self.get_elastic_force(timestep)
        nonlinear_force = lambda f_el: self.darcy.gradient(self.elasticity.update_divergence(f_el), 0.5 * self.couplingRatio * self.alpha**2)
        for k in range(self.elasticitySteps):
            f_elasticity, *_ = self.elasticity.step(f_elasticity, timestep, False, force, nonlinear_force=nonlinear_force)
        eta, sigma = self.elasticity.update_macroscopic(f_elasticity, force + nonlinear_force(f_elasticity))
        dt_div_eta = self.elasticity.update_divergence(f_elasticity - fprev_elasticity)

        source = source_base - (self.couplingRatio * self.alpha / self.epsilon) * dt_div_eta
        pressure = self.darcy.update_macroscopic(f_darcy, source)
        if self.alternative_gradient:
            pressure_grad = self.darcy.gradient(pressure)
        else:
            pressure_grad = pressure_grad_base - self.darcy.gradient(dt_div_eta, 0.5 * self.couplingRatio * self.alpha / self.epsilon)

        f_darcy, *_ = self.darcy.step(f_darcy, timestep, False, source)

        return f_elasticity, f_darcy, eta, sigma, dt_div_eta, pressure, pressure_grad

    def handle_io_timestep(self, timestep, f_elasticity, f_darcy, eta, sigma, pressure, pressure_grad, dt_div_eta):
        """
        This function handles the input/output (I/O) operations at each time step of the simulation.

        It prepares the data to be saved and calls the output_data function, which can be overwritten 
        by the user to customize the I/O operations.

        Parameters
        ----------
        timestep: int
            The current time step of the simulation.
        f_elasticity: jax.numpy.ndarray
            The post-streaming distribution functions for elasticity at the current time step.
        f_darcy: jax.numpy.ndarray
            The post-streaming distribution functions for Darcy flow at the current time step.
        eta: jax.numpy.ndarray
            The elastic displacement field at the current time step.
        sigma: jax.numpy.ndarray
            The elastic stress field at the current time step.
        pressure: jax.numpy.ndarray
            The Darcy pressure field at the current time step.
        pressure_grad: jax.numpy.ndarray
            The gradient of the Darcy pressure field at the current time step.
        dt_div_eta: jax.numpy.ndarray
            The coupling field at the current time step.
        """
        kwargs = {
            "timestep": timestep,
            "eta": eta,
            "sigma": sigma,
            "dt_div_eta": dt_div_eta,
            "pressure": pressure,
            "pressure_grad": pressure_grad,
            "f_poststreaming_elasticity": f_elasticity,
            "f_poststreaming_darcy": f_darcy,
        }
        self.output_data(**kwargs)

    def output_data(self, **kwargs):
        """
        This function is intended to be overwritten by the user to customize the input/output (I/O) 
        operations of the simulation.

        By default, it does nothing. When overwritten, it could save the simulation data to files, 
        display the simulation results in real time, send the data to another process for analysis, etc.

        Parameters
        ----------
        **kwargs: dict
            A dictionary containing the simulation data to be outputted. The keys are the names of the 
            data fields, and the values are the data fields themselves.
        """
        pass

    @partial(jit, static_argnums=(0,))
    def get_elastic_force(self, timestep):
        """
        This function computes the dynamic force to be applied to the solid in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the force according to the specific 
        problem being solved.

        By default, it does nothing and returns None. When overwritten, it could implement a dynamic force term.

        Parameters
        ----------
        timestep: int
            The current time step of the simulation.

        Returns
        -------
        force: jax.numpy.ndarray
            The force to be applied to the elastic solid.
        """
        pass

    @partial(jit, static_argnums=(0,))
    def get_darcy_source(self, timestep):
        """
        This function computes the dynamic source to be applied to the Darcy flow in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the source according to the specific 
        problem being solved.

        By default, it does nothing and returns None. When overwritten, it could implement a dynamic source term.

        Parameters
        ----------
        timestep: int
            The current time step of the simulation.

        Returns
        -------
        source: jax.numpy.ndarray
            The source to be applied to the Darcy flow.
        """
        pass

    def initialize_macroscopic_fields(self):
        """
        This function initializes the macroscopic field (dt_div_eta) to its default value.
        The otherfields are set in self.elasticity.initialize_macroscopic_fields (displacement and stress)
        and self.darcy.initialize_macroscopic_fields (pressure).

        The default dt_div_eta is 0.

        Note: This function is a placeholder and should be overridden in a subclass or in an instance of the class
        to provide specific initial conditions.

        Returns
        -------
            None, None: The default density and velocity, both None. This indicates that the actual values should be set elsewhere.
        """
        print("WARNING: Default initial conditions assumed: dt_div_eta = 0")
        print("         To set explicit initial dt_div_eta, use self.initialize_macroscopic_fields.")
        return None