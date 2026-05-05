# Standard Libraries
import os
import time

# Third-Party Libraries
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from termcolor import colored

# JAX-related imports
from jax import jit, lax, vmap
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import process_allgather
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec, PositionalSharding, Mesh

# functools imports
from functools import partial

# Local/Custom Libraries
from src.utils import downsample_field
from src.boundary_conditions import BoundaryCondition

jax.config.update("jax_spmd_mode", 'allow_all')
# Disables annoying TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LBMElasticity(object):
    """
    LBMElasticity: A class that represents a base for Lattice Boltzmann Method simulations of elasticity.

    Parameters
    ----------
        lattice (object): The lattice object that contains the lattice structure and weights.
        lambda (float): The first Lame parameter for the LBM simulation.
        mu (float): The second Lame parameter for the LBM simulation.
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        nz (int, optional): Number of grid points in the z-direction. Defaults to 0.
        precision (str, optional): A string specifying the precision used for the simulation. Defaults to "f32/f32".
        MG_BCs (dict, optional): A dictionary specifying the boundary conditions for the multigrid solver.
            The keys should be "left", "right", "bottom", "top", "front", and "back", and the boolean values indicate whether
            a boundary condition is applied.Defaults to an empty dictionary, which means that all boundaries are periodic.
        MG_levels (int, optional): The number of multigrid levels to use. Defaults to 1.
        MG_relaxation (float, optional): The relaxation factor to use for the multigrid solver. Defaults to 0.75.
        MG_n_smoothing (int, optional): The number of smoothing steps to perform at each level of the multigrid solver. Defaults to 2.
    """

    def __init__(self, **kwargs):
        self.lamda = kwargs.get("lambda")
        self.mu = kwargs.get("mu")
        self.nx = kwargs.get("nx")
        self.ny = kwargs.get("ny")
        self.nz = kwargs.get("nz")

        self.mg_BCs = kwargs.get("MG_BCs", {})
        self.levels = kwargs.get("MG_levels", 1)
        self.gamma = kwargs.get("MG_relaxation", 0.75)
        self.k_smooth = kwargs.get("MG_n_smoothing", 2)

        self.precision = kwargs.get("precision")
        computedType, storedType = self.set_precisions(self.precision)
        self.precisionPolicy = jmp.Policy(compute_dtype=computedType,
                                            param_dtype=computedType, output_dtype=storedType)

        self.lattice = kwargs.get("lattice")
        self.downsamplingFactor = kwargs.get("downsampling_factor", 1)
        self.printInfoRate = kwargs.get("print_info_rate", 100)
        self.ioRate = kwargs.get("io_rate", 0)
        self.returnFpost = kwargs.get("return_fpost", False)
        self.computeMLUPS = kwargs.get("compute_MLUPS", False)
        self.nDevices = jax.device_count()
        self.backend = jax.default_backend()

        if self.computeMLUPS:
            self.ioRate = 0
            self.printInfoRate = 0

        # Check for distributed mode
        if self.nDevices > jax.local_device_count():
            print("WARNING: Running in distributed mode. Make sure that jax.distributed.initialize is called before performing any JAX computations.")

        self.c = self.lattice.c
        self.q = self.lattice.q
        self.w = self.lattice.w
        self.dim = self.lattice.d
        self._create_transfer_matrices()

        # Adjust the number of grid points in the x direction, if necessary.
        # If the number of grid points is not divisible by the number of devices
        # it increases the number of grid points to the next multiple of the number of devices.
        # This is done in order to accommodate the domain sharding per XLA device
        nx, ny, nz = kwargs.get("nx"), kwargs.get("ny"), kwargs.get("nz")
        if None in {nx, ny, nz}:
            raise ValueError("nx, ny, and nz must be provided. For 2D examples, nz must be set to 0.")
        self.nx = nx
        if nx % self.nDevices:
            self.nx = nx + (self.nDevices - nx % self.nDevices)
            print("WARNING: nx increased from {} to {} in order to accommodate domain sharding per XLA device.".format(nx, self.nx))
        self.ny = ny
        self.nz = nz

        self.show_simulation_parameters()

        # Store grid information
        self.gridInfo = {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "dim": self.lattice.d,
            "lattice": self.lattice
        }

        P = PartitionSpec

        # Define the right permutation
        self.rightPerm = [(i, (i + 1) % self.nDevices) for i in range(self.nDevices)]
        # Define the left permutation
        self.leftPerm = [((i + 1) % self.nDevices, i) for i in range(self.nDevices)]

        # Set up the sharding and streaming for 2D and 3D simulations
        if self.dim == 2:
            self.devices = mesh_utils.create_device_mesh((self.nDevices, 1, 1))
            self.mesh = Mesh(self.devices, axis_names=("x", "y", "value"))
            self.sharding = NamedSharding(self.mesh, P("x", "y", "value"))

            self.streaming = jit(shard_map(self.streaming_m, mesh=self.mesh,
                                                      in_specs=P("x", None, None), out_specs=P("x", None, None), check_rep=False))

        # Set up the sharding and streaming for 2D and 3D simulations
        elif self.dim == 3:
            self.devices = mesh_utils.create_device_mesh((self.nDevices, 1, 1, 1))
            self.mesh = Mesh(self.devices, axis_names=("x", "y", "z", "value"))
            self.sharding = NamedSharding(self.mesh, P("x", "y", "z", "value"))

            self.streaming = jit(shard_map(self.streaming_m, mesh=self.mesh,
                                                      in_specs=P("x", None, None, None), out_specs=P("x", None, None, None), check_rep=False))

        else:
            raise ValueError(f"dim = {self.dim} not supported")

        # Compute the bounding box indices for boundary conditions
        self.boundingBoxIndices = self.bounding_box_indices()
        # Create boundary data for the simulation
        self._create_boundary_data()
        self.force = self.get_force()

    @property
    def lattice(self):
        return self._lattice

    @lattice.setter
    def lattice(self, value):
        if value is None:
            raise ValueError("Lattice type must be provided.")
        if self.nz == 0 and value.name not in ['D2Q8']:
            raise ValueError("For 2D simulations, lattice type must be LatticeD2Q8.")
        if self.nz != 0:
            raise ValueError("3D simulations are not supported.")

        self._lattice = value

    @property
    def lamda(self):
        return self._lamda

    @lamda.setter
    def lamda(self, value):
        if value is None:
            raise ValueError("lambda must be provided")
        if not isinstance(value, float):
            raise TypeError("lambda must be a float")
        self._lamda = value

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        if value is None:
            raise ValueError("mu must be provided")
        if not isinstance(value, float):
            raise TypeError("mu must be a float")
        self._mu = value

    @property
    def levels(self):
        return self._levels

    @levels.setter
    def levels(self, value):
        if value is None:
            raise ValueError("levels must be provided")
        if not isinstance(value, int) or value <= 0:
            raise TypeError("levels must be a positive int")
        denominator = 2**(value-1)
        if (self.nx - int(self.mg_BCs["left"]) - int(self.mg_BCs["right"])) % denominator != 0 \
                or (self.ny - int(self.mg_BCs["bottom"]) - int(self.mg_BCs["top"])) % denominator != 0 \
                or (self.nz - int(self.mg_BCs["front"]) - int(self.mg_BCs["back"])) % denominator != 0:
            raise ValueError("nx,ny,nz (-BCs) must be divisible by 2**(levels-1)")
        self._levels = value

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if value is None:
            raise ValueError("gamma must be provided")
        if not isinstance(value, float):
            raise TypeError("gamma must be a float")
        if value < 0.0 or value > 1.0:
            raise ValueError("gamma must be between 0 and 1")
        self._gamma = value

    @property
    def k_smooth(self):
        return self._k_smooth

    @k_smooth.setter
    def k_smooth(self, value):
        if value is None:
            raise ValueError("k_smooth must be provided")
        if not isinstance(value, int):
            raise TypeError("k_smooth must be an integer")
        if value < 1:
            raise ValueError("k_smooth must be positive")
        self._k_smooth = value

    @property
    def mg_BCs(self):
        return self._mg_BCs

    @mg_BCs.setter
    def mg_BCs(self, value):
        if not isinstance(value, dict):
            raise TypeError("mg_BCs must be a dictionary")
        for key, bc in value.items():
            if key not in ["left", "right", "bottom", "top", "front", "back"]:
                raise ValueError(f"Invalid key '{key}' in mg_BCs. Valid keys are 'left', 'right', 'bottom', 'top', 'front', 'back'.")
            if not isinstance(bc, bool):
                raise TypeError("All values in mg_BCs must be boolean values")
        self._mg_BCs = {"left": False, "right": False, "bottom": False, "top": False, "front": False, "back": False}
        self._mg_BCs.update(value)

    @property
    def nx(self):
        return self._nx

    @nx.setter
    def nx(self, value):
        if value is None:
            raise ValueError("nx must be provided")
        if not isinstance(value, int):
            raise TypeError("nx must be an integer")
        self._nx = value

    @property
    def ny(self):
        return self._ny

    @ny.setter
    def ny(self, value):
        if value is None:
            raise ValueError("ny must be provided")
        if not isinstance(value, int):
            raise TypeError("ny must be an integer")
        self._ny = value

    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, value):
        if value is None:
            raise ValueError("nz must be provided")
        if not isinstance(value, int):
            raise TypeError("nz must be an integer")
        self._nz = value

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value):
        if not isinstance(value, str):
            raise TypeError("precision must be a string")
        self._precision = value

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
    def returnFpost(self):
        return self._returnFpost

    @returnFpost.setter
    def returnFpost(self, value):
        if not isinstance(value, bool):
            raise TypeError("returnFpost must be a boolean")
        self._returnFpost = value

    @property
    def computeMLUPS(self):
        return self._computeMLUPS

    @computeMLUPS.setter
    def computeMLUPS(self, value):
        if not isinstance(value, bool):
            raise TypeError("computeMLUPS must be a boolean")
        self._computeMLUPS = value

    @property
    def nDevices(self):
        return self._nDevices

    @nDevices.setter
    def nDevices(self, value):
        if not isinstance(value, int):
            raise TypeError("nDevices must be an integer")
        self._nDevices = value

    def show_simulation_parameters(self):
        attributes_to_show = [
            'lamda', 'mu', 'nx', 'ny', 'nz', 'dim', 'levels', 'gamma', 'k_smooth',
            'precision', 'lattice', 'downsamplingFactor', 'printInfoRate', 'ioRate',
            'computeMLUPS', 'backend', 'nDevices'
        ]

        descriptive_names = {
            'lamda': 'Lambda',
            'mu': 'Mu',
            'nx': 'Grid Points in X',
            'ny': 'Grid Points in Y',
            'nz': 'Grid Points in Z',
            'dim': 'Dimensionality',
            'levels': 'Multigrid levels',
            'gamma': 'Multigrid relaxation factor',
            'k_smooth': 'Multigrid smoothing steps',
            'precision': 'Precision Policy',
            'lattice': 'Lattice Type',
            'downsamplingFactor': 'Downsampling Factor',
            'printInfoRate': 'Print Info Rate',
            'ioRate': 'I/O Rate',
            'computeMLUPS': 'Compute MLUPS',
            'backend': 'Backend',
            'nDevices': 'Number of Devices'
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

    def _create_boundary_data(self):
        """
        Create boundary data for the Lattice Boltzmann simulation by setting boundary conditions,
        creating grid mask, and preparing local masks and normal arrays.
        """
        self.BCs = []
        self.BCs_MG = { level : [] for level in range(1, self.levels) }

        self.set_boundary_conditions()
        # Accumulate the indices of all BCs to create the grid mask with FALSE along directions that
        # stream into a boundary voxel.
        solid_halo_list = [np.array(bc.indices).T for bc in self.BCs if bc.isSolid]
        solid_halo_voxels = np.unique(np.vstack(solid_halo_list), axis=0) if solid_halo_list else None

        # Create the grid mask on each process
        start = time.time()
        grid_mask = self.create_grid_mask(solid_halo_voxels)
        print("Time to create the grid mask:", time.time() - start)

        start = time.time()
        for bc in self.BCs:
            assert bc.implementationStep in ['PostStreaming', 'PostCollision']
            bc.create_local_mask_and_normal_arrays(grid_mask)
        print("Time to create the local masks and normal arrays:", time.time() - start)

        for level in range(1, self.levels):
            # Accumulate the indices of all MG BCs to create the grid mask with FALSE along directions that
            # stream into a boundary voxel.
            solid_halo_list = [np.array(bc.indices).T for bc in self.BCs_MG[level] if bc.isSolid]
            solid_halo_voxels = np.unique(np.vstack(solid_halo_list), axis=0) if solid_halo_list else None

            # Create the grid mask on each process
            start = time.time()
            grid_mask = self.create_grid_mask(solid_halo_voxels, level)
            print("Time to create the grid mask:", time.time() - start)

            start = time.time()
            for bc in self.BCs_MG[level]:
                assert bc.implementationStep in ['PostStreaming', 'PostCollision']
                bc.create_local_mask_and_normal_arrays(grid_mask)
            print("Time to create the local masks and normal arrays at level", level, ":", time.time() - start)

    @partial(jit, static_argnums=(0, 1, 2, 4))
    def distributed_array_init(self, shape, type, init_val=0, sharding=None):
        """
        Initialize a distributed array using JAX, with a specified shape, data type, and initial value.
        Optionally, provide a custom sharding strategy.

        Parameters
        ----------
            shape (tuple): The shape of the array to be created.
            type (dtype): The data type of the array to be created.
            init_val (scalar, optional): The initial value to fill the array with. Defaults to 0.
            sharding (Sharding, optional): The sharding strategy to use. Defaults to `self.sharding`.

        Returns
        -------
            jax.numpy.ndarray: A JAX array with the specified shape, data type, initial value, and sharding strategy.
        """
        if sharding is None:
            sharding = self.sharding
        x = jnp.full(shape=shape, fill_value=init_val, dtype=type)
        return jax.lax.with_sharding_constraint(x, sharding)

    @partial(jit, static_argnums=(0,2))
    def create_grid_mask(self, solid_halo_voxels, level=0):
        """
        This function creates a mask for the background grid that accounts for the location of the boundaries.

        Parameters
        ----------
            solid_halo_voxels: A numpy array representing the voxels in the halo of the solid object.
            level: The multigrid level for which to create the grid mask (default: 0).

        Returns
        -------
            A JAX array representing the grid mask of the grid.
        """
        # Halo width (hw_x is different to accommodate the domain sharding per XLA device)
        hw_x = self.nDevices
        hw_y = hw_z = 1
        if self.dim == 2:
            nx, ny = self.grid_size(level)
            grid_mask = self.distributed_array_init((nx + 2 * hw_x, ny + 2 * hw_y, self.lattice.q), jnp.bool_, init_val=True)
            grid_mask = grid_mask.at[(slice(hw_x, -hw_x), slice(hw_y, -hw_y), slice(None))].set(False)
            if solid_halo_voxels is not None:
                solid_halo_voxels = solid_halo_voxels.at[:, 0].add(hw_x)
                solid_halo_voxels = solid_halo_voxels.at[:, 1].add(hw_y)
                grid_mask = grid_mask.at[tuple(solid_halo_voxels.T)].set(True)

            grid_mask = self.streaming(grid_mask)
            return lax.with_sharding_constraint(grid_mask, self.sharding)

        elif self.dim == 3:
            nx, ny, nz = self.grid_size(level)
            grid_mask = self.distributed_array_init((nx + 2 * hw_x, ny + 2 * hw_y, nz + 2 * hw_z, self.lattice.q), jnp.bool_, init_val=True)
            grid_mask = grid_mask.at[(slice(hw_x, -hw_x), slice(hw_y, -hw_y), slice(hw_z, -hw_z), slice(None))].set(False)
            if solid_halo_voxels is not None:
                solid_halo_voxels = solid_halo_voxels.at[:, 0].add(hw_x)
                solid_halo_voxels = solid_halo_voxels.at[:, 1].add(hw_y)
                solid_halo_voxels = solid_halo_voxels.at[:, 2].add(hw_z)
                grid_mask = grid_mask.at[tuple(solid_halo_voxels.T)].set(True)
            grid_mask = self.streaming(grid_mask)
            return lax.with_sharding_constraint(grid_mask, self.sharding)

    def grid_size(self, level=0):
        """
        This function calculates the size of the grid at a given multigrid level.

        Parameters
        ----------
            level: int
                The multigrid level (default: 0).

        Returns
        -------
            tuple: A tuple containing the number of grid points in the x, y, and z directions at the specified multigrid level.
        """
        if level == 0:
            if self.dim == 2:
                return self.nx, self.ny
            else:
                return self.nx, self.ny, self.nz
        else:
            nx_BC = int(self.mg_BCs["left"]) + int(self.mg_BCs["right"])
            nx = (self.nx - nx_BC) // 2**level + nx_BC
            ny_BC = int(self.mg_BCs["bottom"]) + int(self.mg_BCs["top"])
            ny = (self.ny - ny_BC) // 2**level + ny_BC
            if self.dim == 2:
                return nx, ny
            else:
                nz_BC = int(self.mg_BCs["front"]) + int(self.mg_BCs["back"])
                nz = (self.nz - nz_BC) // 2**level + nz_BC
                return nx, ny, nz

    def grid_info(self, level=0):
        """
        This function returns the grid information at a given multigrid level.

        Parameters
        ----------
            level: int
                The multigrid level (default: 0).

        Returns
        -------
            dict: A dictionary containing the grid information at the specified multigrid level.
        """
        if self.dim == 2:
            nx, ny = self.grid_size(level)
            return {
                "nx": nx,
                "ny": ny,
                "nz": self.nz,
                "dim": self.lattice.d,
                "lattice": self.lattice
            }
        else:
            nx, ny, nz = self.grid_size(level)
            return {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "dim": self.lattice.d,
                "lattice": self.lattice
            }

    def bounding_box_indices(self, level=0):
        """
        This function calculates the indices of the bounding box of a 2D or 3D grid.
        The bounding box is defined as the set of grid points on the outer edge of the grid.

        Parameters
        ----------
        level: int
            The multigrid level (default: 0).

        Returns
        -------
            boundingBox (dict): A dictionary where keys are the names of the bounding box faces
            ("bottom", "top", "left", "right" for 2D; additional "front", "back" for 3D), and values
            are numpy arrays of indices corresponding to each face.
        """
        if self.dim == 2:
            nx, ny = self.grid_size(level)
            # For a 2D grid, the bounding box consists of four edges: bottom, top, left, and right.
            # Each edge is represented as an array of indices. For example, the bottom edge includes
            # all points where the y-coordinate is 0, so its indices are [[i, 0] for i in range(nx)].
            bounding_box = {"bottom": np.array([[i, 0] for i in range(nx)], dtype=int),
                           "top": np.array([[i, ny - 1] for i in range(nx)], dtype=int),
                           "left": np.array([[0, i] for i in range(ny)], dtype=int),
                           "right": np.array([[nx - 1, i] for i in range(ny)], dtype=int)}

            return bounding_box

        elif self.dim == 3:
            nx, ny, nz = self.grid_size(level)
            # For a 3D grid, the bounding box consists of six faces: bottom, top, left, right, front, and back.
            # Each face is represented as an array of indices. For example, the bottom face includes all points
            # where the z-coordinate is 0, so its indices are [[i, j, 0] for i in range(self.nx) for j in range(self.ny)].
            bounding_box = {
                "bottom": np.array([[i, j, 0] for i in range(nx) for j in range(ny)], dtype=int),
                "top": np.array([[i, j, nz - 1] for i in range(nx) for j in range(ny)],dtype=int),
                "left": np.array([[0, j, k] for j in range(ny) for k in range(nz)], dtype=int),
                "right": np.array([[nx - 1, j, k] for j in range(ny) for k in range(nz)], dtype=int),
                "front": np.array([[i, 0, k] for i in range(nx) for k in range(nz)], dtype=int),
                "back": np.array([[i, ny - 1, k] for i in range(nx) for k in range(nz)], dtype=int)}

            return bounding_box

    def set_precisions(self, precision):
        """
        This function sets the precision of the computations. The precision is defined by a pair of values,
        representing the precision of the computation and the precision of the storage, respectively.

        Parameters
        ----------
            precision (str): A string representing the desired precision. The string should be in the format
            "computation/storage", where "computation" and "storage" are either "f64", "f32", or "f16",
            representing 64-bit, 32-bit, or 16-bit floating point numbers, respectively.

        Returns
        -------
            tuple: A pair of jax.numpy data types representing the computation and storage precisions, respectively.
            If the input string does not match any of the predefined options, the function defaults to (jnp.float32, jnp.float32).
        """
        return {
            "f64/f64": (jnp.float64, jnp.float64),
            "f32/f32": (jnp.float32, jnp.float32),
            "f32/f16": (jnp.float32, jnp.float16),
            "f16/f16": (jnp.float16, jnp.float16),
            "f64/f32": (jnp.float64, jnp.float32),
            "f64/f16": (jnp.float64, jnp.float16),
        }.get(precision, (jnp.float32, jnp.float32))

    def initialize_macroscopic_fields(self):
        """
        This function initializes the macroscopic fields (displacement and stress) to their default values.
        The default displacement is 0 and the default stress is 0.

        Note: This function is a placeholder and should be overridden in a subclass or in an instance of the class
        to provide specific initial conditions.

        Returns
        -------
            None, None: The default density and velocity, both None. This indicates that the actual values should be set elsewhere.
        """
        print("WARNING: Default initial conditions assumed: displacement = 0, stress = 0")
        print("         To set explicit initial displacement and stress, use self.initialize_macroscopic_fields.")
        return None, None

    def assign_fields_sharded(self):
        """
        This function is used to initialize the simulation by assigning the macroscopic fields and populations.

        The function first initializes the macroscopic fields, which are the displacement (eta0) and stress (sigma0).
        Depending on the dimension of the simulation (2D or 3D), it then sets the shape of the array that will hold the
        distribution functions (f).

        If the displacement or stress are not provided, the function initializes the distribution functions with a default
        value (self.w), representing displacement = 0 and stress = 0. Otherwise, it uses the provided displacement and
        stress to initialize the populations.

        Parameters
        ----------
        None

        Returns
        -------
        f: a distributed JAX array of shape (nx, ny, nz, q) or (nx, ny, q) holding the distribution functions for the simulation.
        """
        eta0, sigma0 = self.initialize_macroscopic_fields()

        if self.dim == 2:
            shape = (self.nx, self.ny, self.lattice.q)
        if self.dim == 3:
            shape = (self.nx, self.ny, self.nz, self.lattice.q)

        if eta0 is None or sigma0 is None:
            f = self.distributed_array_init(shape, self.precisionPolicy.output_dtype, init_val=0.0)
        else:
            f = self.initialize_populations(eta0, sigma0)

        return f

    def initialize_populations(self, eta0, sigma0):
        """
        This function initializes the populations (distribution functions) for the simulation.
        It uses the equilibrium distribution function, which is a function of the macroscopic
        displacement and stress.

        Parameters
        ----------
        eta0: jax.numpy.ndarray
            The initial displacement field.
        sigma0: jax.numpy.ndarray
            The initial stress field.

        Returns
        -------
        f: jax.numpy.ndarray
            The array holding the initialized distribution functions for the simulation.
        """
        return self.equilibrium(eta0, sigma0)

    def send_right(self, x, axis_name):
        """
        This function sends the data to the right neighboring process in a parallel computing environment.
        It uses a permutation operation provided by the LAX library.

        Parameters
        ----------
        x: jax.numpy.ndarray
            The data to be sent.
        axis_name: str
            The name of the axis along which the data is sent.

        Returns
        -------
        jax.numpy.ndarray
            The data after being sent to the right neighboring process.
        """
        return lax.ppermute(x, perm=self.rightPerm, axis_name=axis_name)

    def send_left(self, x, axis_name):
        """
        This function sends the data to the left neighboring process in a parallel computing environment.
        It uses a permutation operation provided by the LAX library.

        Parameters
        ----------
        x: jax.numpy.ndarray
            The data to be sent.
        axis_name: str
            The name of the axis along which the data is sent.

        Returns
        -------
            The data after being sent to the left neighboring process.
        """
        return lax.ppermute(x, perm=self.leftPerm, axis_name=axis_name)

    def streaming_m(self, f):
        """
        This function performs the streaming step in the Lattice Boltzmann Method, which is
        the propagation of the distribution functions in the lattice.

        To enable multi-GPU/TPU functionality, it extracts the left and right boundary slices of the
        distribution functions that need to be communicated to the neighboring processes.

        The function then sends the left boundary slice to the right neighboring process and the right
        boundary slice to the left neighboring process. The received data is then set to the
        corresponding indices in the receiving domain.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The array holding the distribution functions for the simulation.

        Returns
        -------
        jax.numpy.ndarray
            The distribution functions after the streaming operation.
        """
        f = self.streaming_p(f)
        left_comm, right_comm = f[:1, ..., self.lattice.right_indices], f[-1:, ..., self.lattice.left_indices]

        left_comm, right_comm = self.send_right(left_comm, 'x'), self.send_left(right_comm, 'x')
        f = f.at[:1, ..., self.lattice.right_indices].set(left_comm)
        f = f.at[-1:, ..., self.lattice.left_indices].set(right_comm)
        return f

    @partial(jit, static_argnums=(0,))
    def streaming_p(self, f):
        """
        Perform streaming operation on a partitioned (in the x-direction) distribution function.

        The function uses the vmap operation provided by the JAX library to vectorize the computation
        over all lattice directions.

        Parameters
        ----------
            f: The distribution function.

        Returns
        -------
            The updated distribution function after streaming.
        """
        def streaming_i(f, c):
            """
            Perform individual streaming operation in a direction.

            Parameters
            ----------
                f: The distribution function.
                c: The streaming direction vector.

            Returns
            -------
                jax.numpy.ndarray
                The updated distribution function after streaming.
            """
            if self.dim == 2:
                return jnp.roll(f, (c[0], c[1]), axis=(0, 1))
            elif self.dim == 3:
                return jnp.roll(f, (c[0], c[1], c[2]), axis=(0, 1, 2))

        return vmap(streaming_i, in_axes=(-1, 0), out_axes=-1)(f, self.c.T)

    def _create_transfer_matrices(self):
        """
        This function computes the transfer matrices from distribution functions to moments and vice versa.
        """
        # remove moments m_21, m12 and m_f (last moments) which are fully equilibrated each step
        g = 1.0 / (12.0*(self.lamda + self.mu) - 4.0)
        if(jnp.abs(g) > 100):
            print("Warning: scaling value is too large. Reset last moment to default. Boundary conditions may fail.")
            g = 0.0
        #t = 1 + 2 * g
        #self.f_to_m = jnp.array([[1, 0,-1, 0, 1,-1,-1, 1],
        #                         [0, 1, 0,-1, 1, 1,-1,-1],
        #                         [1, 1, 1, 1, 2, 2, 2, 2],
        #                         [1,-1, 1,-1, 0, 0, 0, 0],
        #                         [0, 0, 0, 0, 1,-1, 1,-1],
        #                         [0, 0, 0, 0, 1,-1,-1, 1],
        #                         [0, 0, 0, 0, 1, 1,-1,-1],
        #                         [g, g, g, g, t, t, t, t]], dtype=self.precisionPolicy.compute_dtype).T
        self.f_to_m = jnp.array([[1, 0,-1, 0, 1,-1,-1, 1],
                                 [0, 1, 0,-1, 1, 1,-1,-1],
                                 [1, 1, 1, 1, 2, 2, 2, 2],
                                 [1,-1, 1,-1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1,-1, 1,-1]], dtype=self.precisionPolicy.compute_dtype).T

        h = 0.25 + 0.5 * g
        n = -0.25 * g
        # remove moment m_f (last moment) which equilibrates to zero
        # Instead, multiply the moment m_s accodingly to account for m_22
        #self.m_to_f = jnp.array([[ 0.5,   0, h, 0.25,    0, -0.5,    0, -0.5],
        #                         [   0, 0.5, h,-0.25,    0,    0, -0.5, -0.5],
        #                         [-0.5,   0, h, 0.25,    0,  0.5,    0, -0.5],
        #                         [   0,-0.5, h,-0.25,    0,    0,  0.5, -0.5],
        #                         [   0,   0, n,    0, 0.25, 0.25, 0.25, 0.25],
        #                         [   0,   0, n,    0,-0.25,-0.25, 0.25, 0.25],
        #                         [   0,   0, n,    0, 0.25,-0.25,-0.25, 0.25],
        #                         [   0,   0, n,    0,-0.25, 0.25,-0.25, 0.25]], dtype=self.precisionPolicy.compute_dtype).T
        self.m_to_f = jnp.array([[ 0.5,   0, h, 0.25,    0, -0.5,    0],
                                 [   0, 0.5, h,-0.25,    0,    0, -0.5],
                                 [-0.5,   0, h, 0.25,    0,  0.5,    0],
                                 [   0,-0.5, h,-0.25,    0,    0,  0.5],
                                 [   0,   0, n,    0, 0.25, 0.25, 0.25],
                                 [   0,   0, n,    0,-0.25,-0.25, 0.25],
                                 [   0,   0, n,    0, 0.25,-0.25,-0.25],
                                 [   0,   0, n,    0,-0.25, 0.25,-0.25]], dtype=self.precisionPolicy.compute_dtype).T

    @partial(jit, static_argnums=(0,))
    def _restrict(self, f):
        """
        MG restriction from f to half size, i.e., assumes the shape to be (2 nx, 2 ny, 2 nz, q)
        """
        x_min = int(self.mg_BCs["left"])
        x_max = f.shape[0] - int(self.mg_BCs["right"])
        y_min = int(self.mg_BCs["bottom"])
        y_max = f.shape[1] - int(self.mg_BCs["top"])
        if self.dim == 2:
            kernel = jnp.full((2,2, 1,1), 1, dtype=self.precisionPolicy.compute_dtype)
            dn = lax.conv_dimension_numbers((1,) + f.shape, kernel.shape, ('CHWN', 'HWIO', 'CHWN'))
            return lax.conv_general_dilated(f[jnp.newaxis, x_min:x_max, y_min:y_max, :], kernel,  # lhs, rhs
                                            (2,2),   # window strides
                                            'VALID', # padding mode
                                            dimension_numbers=dn)[0]
        elif self.dim == 3:
            z_min = int(self.mg_BCs["front"])
            z_max = f.shape[2] - int(self.mg_BCs["back"])
            kernel = jnp.full((2,2,2, 1,1), 0.5, dtype=self.precisionPolicy.compute_dtype)
            dn = lax.conv_dimension_numbers((1,) + f.shape, kernel.shape, ('CHWDN', 'HWDIO', 'CHWDN'))
            return lax.conv_general_dilated(f[jnp.newaxis, x_min:x_max, y_min:y_max, z_min:z_max, :], kernel,  # lhs, rhs
                                            (2,2,2), # window strides
                                            'VALID', # padding mode
                                            dimension_numbers=dn)[0]
        else:
            raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def _prolong(self, f):
        """
        MG prolongation from f to double size.
        """
        x_min = int(self.mg_BCs["left"])
        x_max = f.shape[0] - int(self.mg_BCs["right"])
        y_min = int(self.mg_BCs["bottom"])
        y_max = f.shape[1] - int(self.mg_BCs["top"])
        if self.dim == 2:
            kernel = jnp.full((2,2, 1,1), 1, dtype=self.precisionPolicy.compute_dtype)
            dn = lax.conv_dimension_numbers((1,) + f.shape, kernel.shape, ('CHWN', 'HWIO', 'CHWN'))
            return lax.conv_general_dilated(f[jnp.newaxis, x_min:x_max, y_min:y_max, :], kernel,  # lhs, rhs
                                            (1,1),   # window strides
                                            ((1,1), (1,1)), # padding
                                            (2,2), (1,1), # dilation in lhs, rhs
                                            dimension_numbers=dn)[0]
        elif self.dim == 3:
            z_min = int(self.mg_BCs["front"])
            z_max = f.shape[2] - int(self.mg_BCs["back"])
            kernel = jnp.full((2,2,2, 1,1), 1, dtype=self.precisionPolicy.compute_dtype)
            dn = lax.conv_dimension_numbers((1,) + f.shape, kernel.shape, ('CHWDN', 'HWDIO', 'CHWDN'))
            return lax.conv_general_dilated(f[jnp.newaxis, x_min:x_max, y_min:y_max, z_min:z_max, :], kernel,  # lhs, rhs
                                            (1,1,1), # window strides
                                            ((1,1), (1,1), (1,1)), # padding
                                            (2,2,2), (1,1,1), # dilation in lhs, rhs
                                            dimension_numbers=dn)[0]
        else:
            raise NotImplementedError

    @partial(jit, static_argnums=(0, 3), inline=True)
    def equilibrium(self, eta, sigma, cast_output=True, external_force=None):
        """
        This function computes the equilibrium distribution function in the Lattice Boltzmann Method.
        The equilibrium distribution function is a function of the macroscopic displacement and stress.

        The function first casts the displacement and stress to the compute precision if the cast_output flag is True.
        The function finally casts the equilibrium distribution function to the output precision if the cast_output
        flag is True.

        Parameters
        ----------
        eta: jax.numpy.ndarray
            The macroscopic displacement.
        sigma: jax.numpy.ndarray
            The macroscopic stress.
        cast_output: bool, optional
            A flag indicating whether to cast the displacement, stress, and equilibrium distribution function to the
            compute and output precisions. Default is True.

        Returns
        -------
        feq: ja.numpy.ndarray
            The equilibrium distribution function.
        """
        # Cast the displacement and stress to the compute precision if the cast_output flag is True
        if cast_output:
            eta, sigma = self.precisionPolicy.cast_to_compute((eta, sigma))

        # inverted version of function update_macroscopic
        if self.force is not None:
            eta = eta - 0.5 * self.force
        if external_force is not None:
            eta = eta - 0.5 * external_force

        cs = -1.0 - 1 / (3*(self.lamda+self.mu))
        ms = (sigma[..., 0:1] + sigma[..., 2:3]) * cs
        cd = -1.0 - 1 / (6*self.mu)
        md = (sigma[..., 0:1] - sigma[..., 2:3]) * cd
        m11 = sigma[..., 1:2] * cd

        # remove moment m_f (last moment) which equilibrates to zero
        #m = jnp.concatenate((eta, ms, md, m11, eta/3.0, jnp.zeros(ms.shape)), axis=-1)
        m = jnp.concatenate((eta, ms, md, m11, eta/3.0), axis=-1)
        feq = jnp.matmul(m, self.m_to_f)

        if cast_output:
            return self.precisionPolicy.cast_to_output(feq)
        else:
            return feq

    @partial(jit, static_argnums=(0,3), donate_argnums=(1,))
    def collision(self, f, external_force, use_force=True):
        """
        Elasticity collision step for lattice.

        The collision step is where the main physics of the LBM is applied. Here, the moments of
        the distribution function are relaxed towards the equilibrium moments.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        external_force = self.precisionPolicy.cast_to_compute(external_force)
        m = jnp.matmul(f, self.f_to_m)

        # update m12 and m21 based on m10 and m01 and the force
        # and relax ms, md and m11
        cs = (3*(self.lamda+self.mu) - 1) / (3*(self.lamda+self.mu) + 1)
        cd = (6*self.mu - 1) / (6*self.mu + 1)

        if use_force and self.force is not None and external_force is not None:
            force = self.force + external_force
            m = jnp.concatenate([m[...,0:2] + force,
                                m[...,2:3] * cs,
                                m[...,3:5] * cd,
                                m[...,0:2]/3.0 + force/6.0], axis=-1)
        elif external_force is not None:
            m = jnp.concatenate([m[...,0:2] + external_force,
                                m[...,2:3] * cs,
                                m[...,3:5] * cd,
                                m[...,0:2]/3.0 + external_force/6.0], axis=-1)
        elif use_force and self.force is not None:
            m = jnp.concatenate([m[...,0:2] + self.force,
                                m[...,2:3] * cs,
                                m[...,3:5] * cd,
                                m[...,0:2]/3.0 + self.force/6.0], axis=-1)
        else:
            m = jnp.concatenate([m[...,0:2],
                                m[...,2:3] * cs,
                                m[...,3:5] * cd,
                                m[...,0:2]/3.0], axis=-1)

        fout = jnp.matmul(m, self.m_to_f)

        return self.precisionPolicy.cast_to_output(fout)

    @partial(jit, static_argnums=(0,), inline=True)
    def update_macroscopic(self, f, external_force=None):
        """
        This function computes the macroscopic variables (displacement and stress) based on the
        distribution functions (f).

        Both quantities are computed from the relaxed moments assuming f is in precollision state.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The distribution functions.

        Returns
        -------
        eta: jax.numpy.ndarray
            Computed displacement.
        sigma: jax.numpy.ndarray
            Computed stress.
        """
        m = jnp.matmul(f, self.f_to_m)

        # first compute eta based on m10 and m01 and the force
        eta = m[...,0:2]
        if self.force is not None:
            eta = eta + 0.5 * self.force
        if external_force is not None:
            eta = eta + 0.5 * external_force

        # then compute sigma based on ms, md and m11
        cs = -1.5*(self.lamda+self.mu) / (3*(self.lamda+self.mu) + 1)
        ms = m[..., 2] * cs
        cd = -3*self.mu / (6*self.mu + 1)
        md = m[..., 3] * cd
        sigma = jnp.stack((ms + md, (2*cd) * m[..., 4], ms - md), axis=-1)

        return eta, sigma

    @partial(jit, static_argnums=(0,), inline=True)
    def update_divergence(self, f):
        """
        This function computes the macroscopic divergence based on the distribution functions (f).

        The quantity is computed from the relaxed moments assuming f is in precollision state.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The distribution functions.

        Returns
        -------
        div: jax.numpy.ndarray
            Computed divergence.
        """
        # only compute ms and half-relax to get the divergence
        return (-1.5 / (3*(self.lamda+self.mu) + 1)) * jnp.matmul(f, self.f_to_m[:, 2:3])

    @partial(jit, static_argnums=(0, 4), inline=True)
    def apply_bc(self, fout, fin, timestep, implementation_step):
        """
        This function applies the boundary conditions to the distribution functions.

        It iterates over all boundary conditions (BCs) and checks if the implementation step of the
        boundary condition matches the provided implementation step. If it does, it applies the
        boundary condition to the post-streaming distribution functions (fout).

        Parameters
        ----------
        fout: jax.numpy.ndarray
            The post-collision distribution functions.
        fin: jax.numpy.ndarray
            The post-streaming distribution functions.
        implementation_step: str
            The implementation step at which the boundary conditions should be applied.

        Returns
        -------
        ja.numpy.ndarray
            The output distribution functions after applying the boundary conditions.
        """
        for bc in self.BCs:
            fout = bc.prepare_populations(fout, fin, implementation_step)
            if bc.implementationStep == implementation_step:
                if bc.isDynamic:
                    fout = bc.apply(fout, fin, timestep)
                else:
                    fout = fout.at[bc.indices].set(bc.apply(fout, fin))

        return fout

    @partial(jit, static_argnums=(0, 4, 5), inline=True)
    def apply_mg_bc(self, fout, fin, timestep, implementation_step, level):
        """
        This function applies the boundary conditions to the distribution functions within the multigrid step.

        It iterates over all boundary conditions (BCs) and checks if the implementation step of the
        boundary condition matches the provided implementation step. If it does, it applies the
        boundary condition to the post-streaming distribution functions (fout).

        Parameters
        ----------
        fout: jax.numpy.ndarray
            The post-collision distribution functions.
        fin: jax.numpy.ndarray
            The post-streaming distribution functions.
        implementation_step: str
            The implementation step at which the boundary conditions should be applied.
        level: int
            The multigrid level at which the boundary conditions should be applied.

        Returns
        -------
        ja.numpy.ndarray
            The output distribution functions after applying the boundary conditions.
        """
        for bc in self.BCs_MG[level]:
            fout = bc.prepare_populations(fout, fin, implementation_step)
            if bc.implementationStep == implementation_step:
                if bc.isDynamic:
                    fout = bc.apply(fout, fin, timestep)
                else:
                    fout = fout.at[bc.indices].set(bc.apply(fout, fin))

        return fout

    def step(self, f_poststreaming, timestep, return_fpost=False, external_force=None, nonlinear_force=None):
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
        external_force: jax.numpy.ndarray, optional
            The external force to apply in the current timestep of the simulation.
        nonlinear_force: callable[jax.numpy.ndarray, jax.numpy.ndarray], optional
            A nonlinear force depending on the solution, to apply in the current timestep of the simulation.

        Returns
        -------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions after the simulation step.
        f_postcollision: jax.numpy.ndarray or None
            The post-collision distribution functions after the simulation step, or None if
            return_fpost is False.
        """

        if self.levels == 1:
            if callable(nonlinear_force):
                external_force += nonlinear_force(f_poststreaming)
            return self._step(f_poststreaming, timestep, return_fpost, external_force)
        else:
            return self._mg_step(f_poststreaming, timestep, external_force=external_force, nonlinear_force=nonlinear_force)

    @partial(jit, static_argnums=(0, 3, 5), donate_argnums=(1,))
    def _step(self, f_poststreaming, timestep, return_fpost=False, external_force=None, level=0):
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
        external_force: jax.numpy.ndarray, optional
            The external force to apply in the current timestep of the simulation.
        residual: jax.numpy.ndarray, optional
            The residual to apply coming from the finer level.
        level: int, optional
            The multigrid level.

        Returns
        -------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions after the simulation step.
        f_postcollision: jax.numpy.ndarray or None
            The post-collision distribution functions after the simulation step, or None if
            return_fpost is False.
        """
        f_postcollision = self.collision(f_poststreaming, external_force, level==0)
        if level > 0:
            f_postcollision = self.apply_mg_bc(f_postcollision, f_poststreaming, timestep, "PostCollision", level)
        else:
            f_postcollision = self.apply_bc(f_postcollision, f_poststreaming, timestep, "PostCollision")
        f_poststreaming = self.streaming(f_postcollision)
        if level > 0:
            f_poststreaming = self.apply_mg_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming", level)
        else:
            f_poststreaming = self.apply_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming")

        if return_fpost:
            return f_poststreaming, f_postcollision
        else:
            return f_poststreaming, None

    #@partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def _mg_step(self, f_poststreaming, timestep, level=0, residual=None, external_force=None, nonlinear_force=None):
        """
        This function performs a single step of the LBM simulation with multigrid.

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
        level: int
            The multigrid level.
        residual: jax.numpy.ndarray, optional
            The residual to apply coming from the finer level.
        external_force: jax.numpy.ndarray, optional
            The external force to apply in the current timestep of the simulation.
        nonlinear_force: callable[jax.numpy.ndarray, jax.numpy.ndarray], optional
            A nonlinear force depending on the solution, to apply in the current timestep of the simulation.

        Returns
        -------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions after the simulation step.
        f_postcollision: jax.numpy.ndarray or None
            The post-collision distribution functions after the simulation step, or None if
            return_fpost is False.
        """
        # pre smoothing
        for _ in range(self.k_smooth):
            if callable(nonlinear_force):
                if external_force is not None:
                    ext_force = external_force + nonlinear_force(f_poststreaming)
                else:
                    ext_force = nonlinear_force(f_poststreaming)
            else:
                ext_force = external_force
            f_poststreaming = (1-self.gamma) * f_poststreaming \
                + self.gamma * self._step(self.substract_residual(f_poststreaming, residual), timestep, False, ext_force, level)[0]

        if level < self.levels-1:
            x_min = int(self.mg_BCs["left"])
            x_max = f_poststreaming.shape[0] - int(self.mg_BCs["right"])
            y_min = int(self.mg_BCs["bottom"])
            y_max = f_poststreaming.shape[1] - int(self.mg_BCs["top"])
            if self.dim == 3:
                z_min = int(self.mg_BCs["front"])
                z_max = f_poststreaming.shape[2] - int(self.mg_BCs["back"])

            if callable(nonlinear_force):
                if external_force is not None:
                    ext_force = external_force + nonlinear_force(f_poststreaming)
                else:
                    ext_force = nonlinear_force(f_poststreaming)
            else:
                ext_force = external_force
            res = f_poststreaming - self._step(self.substract_residual(f_poststreaming.copy(), residual),
                                               timestep, False, ext_force, level)[0]
            res_coarse = self._restrict(res)
            f_coarse = jnp.zeros((*self.grid_size(level+1), self.lattice.q), dtype=self.precisionPolicy.compute_dtype)
            f_coarse = self._mg_step(f_coarse, timestep, level+1, res_coarse, nonlinear_force=nonlinear_force)
            if self.dim == 2:
                f_poststreaming = f_poststreaming.at[x_min:x_max, y_min:y_max].add(self._prolong(f_coarse))
            else:
                f_poststreaming = f_poststreaming.at[x_min:x_max, y_min:y_max, z_min:z_max].add(self._prolong(f_coarse))

        # post smoothing
        for _ in range(self.k_smooth):
            if callable(nonlinear_force):
                if external_force is not None:
                    ext_force = external_force + nonlinear_force(f_poststreaming)
                else:
                    ext_force = nonlinear_force(f_poststreaming)
            else:
                ext_force = external_force
            f_poststreaming = (1-self.gamma) * f_poststreaming \
                + self.gamma * self._step(self.substract_residual(f_poststreaming, residual), timestep, False, ext_force, level)[0]

        if level == 0:
            if self.dim == 2:
                return f_poststreaming, jnp.linalg.norm(res[x_min:x_max, y_min:y_max])
            else:
                return f_poststreaming, jnp.linalg.norm(res[x_min:x_max, y_min:y_max, z_min:z_max])
        return f_poststreaming

    @partial(jit, static_argnums=(0,), donate_argnums=(1,), inline=True)
    def substract_residual(self, f, residual):
        """
        This function subtracts the residual from the distribution functions (f) in the interior of the domain.

        Parameters
        ----------
        f: jax.numpy.ndarray
            The distribution functions from which the residual should be subtracted.
        residual: jax.numpy.ndarray
            The residual to be subtracted from the distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The distribution functions after subtracting the residual in the interior of the domain.
        """
        if residual is None:
            return f

        x_min = int(self.mg_BCs["left"])
        x_max = f.shape[0] - int(self.mg_BCs["right"])
        y_min = int(self.mg_BCs["bottom"])
        y_max = f.shape[1] - int(self.mg_BCs["top"])
        if self.dim == 2:
            f = f.at[x_min:x_max, y_min:y_max].add(-residual)
        if self.dim == 3:
            z_min = int(self.mg_BCs["front"])
            z_max = f.shape[2] - int(self.mg_BCs["back"])
            f = f.at[x_min:x_max, y_min:y_max, z_min:z_max].add(-residual)
        return f

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

        # Loop over all time steps
        for timestep in range(start_step, t_max + 1):
            io_flag = self.ioRate > 0 and (timestep % self.ioRate == 0 or timestep == t_max)
            print_iter_flag = self.printInfoRate> 0 and timestep % self.printInfoRate== 0

            if io_flag:
                # Update the macroscopic variables and save the previous values (for error computation)
                eta_prev, sigma_prev = self.update_macroscopic(f)
                eta_prev = downsample_field(eta_prev, self.downsamplingFactor)
                sigma_prev = downsample_field(sigma_prev, self.downsamplingFactor)
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                eta_prev = process_allgather(eta_prev)
                sigma_prev = process_allgather(sigma_prev)


            # Perform one time-step (collision, streaming, and boundary conditions)
            f, fstar = self.step(f, timestep, return_fpost=self.returnFpost)
            # Print the progress of the simulation
            if print_iter_flag:
                print(colored("Timestep ", 'blue') + colored(f"{timestep}", 'green') + colored(" of ", 'blue') + colored(f"{t_max}", 'green') + colored(" completed", 'blue'))

            if io_flag:
                # Save the simulation data
                print(f"Saving data at timestep {timestep}/{t_max}")
                eta, sigma = self.update_macroscopic(f)
                eta = downsample_field(eta, self.downsamplingFactor)
                sigma = downsample_field(sigma, self.downsamplingFactor)

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                eta = process_allgather(eta)
                sigma = process_allgather(sigma)

                # Save the data
                self.handle_io_timestep(timestep, f, fstar, eta, sigma, eta_prev, sigma_prev)

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

    def handle_io_timestep(self, timestep, f, fstar, eta, sigma, eta_prev, sigma_prev):
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
        eta: jax.numpy.ndarray
            The displacement field at the current time step.
        sigma: jax.numpy.ndarray
            The stress field at the current time step.
        """
        kwargs = {
            "timestep": timestep,
            "eta": eta,
            "eta_prev": eta_prev,
            "sigma": sigma,
            "sigma_prev": sigma_prev,
            "f_poststreaming": f,
            "f_postcollision": fstar
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

    def set_boundary_conditions(self):
        """
        This function sets the boundary conditions for the simulation.

        It is intended to be overwritten by the user to specify the boundary conditions according to
        the specific problem being solved.

        By default, it does nothing. When overwritten, it could set periodic boundaries, no-slip
        boundaries, inflow/outflow boundaries, etc.
        """
        pass

    def get_force(self):
        """
        This function computes the force to be applied to the solid in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the force according to the specific
        problem being solved.

        By default, it does nothing and returns None. When overwritten, it could implement a constant
        force term.

        Returns
        -------
        force: jax.numpy.ndarray
            The force to be applied to the solid.
        """
        pass

class ElasticityBounceBackFullway(BoundaryCondition):
    """
    Elasticity bounce-back (Dirichlet) boundary condition for a lattice Boltzmann method simulation.

    This class implements a bounce-back boundary condition, where particles hitting the boundary are reflected
    back in the direction they came from, with an additional displacement at the boundary. The boundary
    condition is applied after the collision step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "ElasticityBounceBackFullway".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostCollision".
    eta : array-like
        The prescribed value of the displacement vector for the boundary condition. No-slip BC is assumed if vel=None (default).

    """
    def __init__(self, indices, gridInfo, precision_policy, eta=None):
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "ElasticityBounceBackFullway"
        self.implementationStep = "PostCollision"
        self.eta = eta

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the moving bounce-back boundary condition.

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
        if self.eta is None:
            return fin[self.indices][..., self.lattice.opp_indices]
        else:
            c = jnp.array(self.lattice.c, dtype=self.precisionPolicy.compute_dtype)
            cu = 6.0 * self.lattice.w * jnp.dot(self.eta, c)
            return fin[self.indices][..., self.lattice.opp_indices] + cu

class ElasticityBounceBack(BoundaryCondition):
    """
    Halfway bounce-back (Dirichlet) boundary condition for an elastic LBM simulation.

    This class implements a halfway bounce-back boundary condition. The boundary condition is applied after
    the streaming step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "ElasticityBounceBack".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostStreaming".
    needsExtraConfiguration : bool
        Whether the boundary condition needs extra configuration before it can be applied. For this class, it is True.
    isSolid : bool
        Whether the boundary condition represents a solid boundary. For this class, it is True.
    eta : array-like
        The prescribed value of the displacement vector for the boundary condition. No-slip BC is assumed if vel=None (default).
    """
    def __init__(self, indices, gridInfo, precision_policy, eta=None):
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "ElasticityBounceBack"
        self.implementationStep = "PostStreaming"
        self.needsExtraConfiguration = True
        self.isSolid = True
        self.eta = eta

    def configure(self, boundaryMask):
        """
        Configures the boundary condition.

        Parameters
        ----------
        boundaryMask : array-like
            The grid mask for the boundary voxels.

        Notes
        -----
        This method performs an index shift for the halfway bounce-back boundary condition. It updates the indices of
        the boundary nodes to be the indices of fluid nodes adjacent of the solid nodes.
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
        if (nbd_orig != nbd_modified) and self.eta is not None:
            if self.eta.shape == (self.lattice.d,):
                self.eta = jnp.tile(self.eta, (nbd_modified, 1))
            else:
                print("WARNING: Assuming a correct displacement vector to impose at the BC cells!")
                self.eta = self.eta[array_indices]
        elif self.eta is not None and self.eta.shape == (self.lattice.d,):
            self.eta = jnp.tile(self.eta, (nbd_modified, 1))

    @partial(jit, static_argnums=(0,))
    def impose_boundary_displacement(self, fbd, bindex):
        c = jnp.array(self.lattice.c, dtype=self.precisionPolicy.compute_dtype)
        cu = 6.0 * self.lattice.w * jnp.dot(self.eta, c)
        # TODO: check sign (+ in paper)
        fbd = fbd.at[bindex, self.imissing].add(cu[bindex, self.imissing])
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

        fbd = fbd.at[bindex, self.imissing].set(fin[self.indices][bindex, self.iknown])
        if self.eta is not None:
            fbd = self.impose_boundary_displacement(fbd, bindex)
        return fbd

class ElasticityAntiBounceBack(BoundaryCondition):
    """
    Halfway anti-bounce-back (Neumann) boundary condition for an elastic LBM simulation.

    This class implements a halfway anti-bounce-back boundary condition. The boundary condition is applied after
    the streaming step.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "ElasticityAntiBounceBack".
    implementationStep : str
        The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
        it is "PostStreaming".
    needsExtraConfiguration : bool
        Whether the boundary condition needs extra configuration before it can be applied. For this class, it is True.
    isSolid : bool
        Whether the boundary condition represents a solid boundary. For this class, it is True.
    T : array-like
        The prescribed value of the normal stress vector for the boundary condition. No-stress BC is assumed if T=None (default).
    """
    def __init__(self, indices, gridInfo, precision_policy, T=None):
        super().__init__(indices, gridInfo, precision_policy)
        self.name = "ElasticityAntiBounceBack"
        self.implementationStep = "PostStreaming"
        self.needsExtraConfiguration = True
        self.isSolid = True
        self.T = T

    def configure(self, boundaryMask):
        """
        Configures the boundary condition.

        Parameters
        ----------
        boundaryMask : array-like
            The grid mask for the boundary voxels.

        Notes
        -----
        This method performs an index shift for the halfway bounce-back boundary condition. It updates the indices of
        the boundary nodes to be the indices of fluid nodes adjacent of the solid nodes.
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
        if (nbd_orig != nbd_modified) and self.T is not None:
            if self.T.shape == (self.lattice.d,):
                self.T = jnp.tile(self.T, (nbd_modified, 1))
            else:
                print("WARNING: Assuming a correct normal stress vector to impose at the BC cells!")
                self.T = self.T[array_indices]
        elif self.T is not None and self.T.shape == (self.lattice.d,):
            self.T = jnp.tile(self.T, (nbd_modified, 1))

    def get_missing_mask(self, boundaryMask):
        if self.T is not None:
            normals = self.get_normals(boundaryMask)
            if jnp.all(normals[:, 0] == -1): # left
                self.i_update = (0, 4, 7)
                self.T_update = jnp.stack((self.T[:, 0], 0.5*self.T[:, 1], -0.5*self.T[:, 1]), axis=-1)
            elif jnp.all(normals[:, 0] == 1): # right
                self.i_update = (2, 5, 6)
                self.T_update = jnp.stack((-self.T[:, 0], 0.5*self.T[:, 1], -0.5*self.T[:, 1]), axis=-1)
            elif jnp.all(normals[:, 1] == -1): # bottom
                self.i_update = (1, 4, 5)
                self.T_update = jnp.stack((self.T[:, 1], 0.5*self.T[:, 0], -0.5*self.T[:, 0]), axis=-1)
            elif jnp.all(normals[:, 1] == 1): # top
                self.i_update = (3, 6, 7)
                self.T_update = jnp.stack((-self.T[:, 1], -0.5*self.T[:, 0], 0.5*self.T[:, 0]), axis=-1)
            else:
                raise NotImplementedError
        return super().get_missing_mask(boundaryMask)

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
        fbd = fout[self.indices]
        fbd = fbd.at[self.imissingMask].set(-fin[self.indices][self.iknownMask])
        if self.T is not None:
            fbd = fbd.at[:, self.i_update].add(self.T_update)
        return fbd
