import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from simwave.kernel.backend.compiler import Compiler


class Middleware:
    """
    Communication interface between frontend and backend.

    Parameters
    ----------
    compiler : Compiler
        Compiler object.
    """
    def __init__(self, compiler):
        if compiler is None:
            self._compiler = Compiler()
        else:
            self._compiler = compiler

    @property
    def compiler(self):
        return self._compiler

    def library(self, dtype, operator, c_file):
        """Load and return the C library."""

        # convert dtype to C compiling value
        # that define float precision
        type = {
            'float32': '-DFLOAT',
            'float64': '-DDOUBLE'
        }
        
        # compile the code
        shared_object = self.compiler.compile(           
            float_precision=type[str(dtype)],            
            program_path=c_file
        )

        # load the library
        return ctypes.cdll.LoadLibrary(shared_object)

    def exec(self, operator, c_file, **kwargs):
        """
        Run an operator.

        Parameters
        ----------
        operator : str
            operator to be executed.
        kwargs : dict
            List of keyword arguments.

        Returns
        ----------
        tuple
            The operation results
        """
       
        # get the grid shape
        grid_shape = kwargs.get('velocity_model').shape
        
        nz, nx, ny = grid_shape
        kwargs.update({'nz': nz, 'nx': nx, 'ny': ny})

        # unpack the grid spacing
        grid_spacing = kwargs.get('grid_spacing')
        kwargs.pop('grid_spacing')

        dz, dx, dy = grid_spacing
        kwargs.update({'dz': dz, 'dx': dx, 'dy': dy})

        # run the forward operator
        if operator == 'forward-saving' or operator == 'forward-no-saving':
            return self._exec_forward(operator, c_file, **kwargs)

        # run the gradient operator
        if operator == 'gradient':
            return self._exec_gradient(c_file, **kwargs)

    def _exec_forward(self, operator, c_file, **kwargs):
        """
        Run the forward operator.

        Parameters
        ----------
        kwargs : dict
            Dictonary of keyword arguments.

        Returns
        ----------
        ndarray
            Full wavefield after timestepping.
        ndarray
            Shot record after timestepping.
        """

        # load the C library
        lib = self.library(            
            dtype=kwargs.get('velocity_model').dtype,
            operator="forward",
            c_file=c_file
        )

        # get the argtype for each arg key
        types = self._argtypes(**kwargs)

        # get the all possible keys in order
        ordered_keys = self._keys_in_order

        # list of argtypes
        argtypes = []

        # list of args to pass to C function
        args = []

        # compose the list of args and arg types
        for key in ordered_keys:
            if kwargs.get(key) is not None:
                argtypes.append(types.get(key))
                args.append(kwargs.get(key))

        forward = lib.forward
        forward.restype = ctypes.c_double
        forward.argtypes = argtypes

        # run the C forward function
        exec_time = forward(*args)

        print('Run forward in %f seconds.' % exec_time)

        if operator == 'forward-saving':
            return kwargs.get('snapshots_prev'), kwargs.get('snapshots_current'), kwargs.get('shot_record')
        elif operator == 'forward-no-saving':
            return kwargs.get('u'), kwargs.get('shot_record')
        else:
            raise Exception("Operator not supported.")

    def _exec_gradient(self, c_file, **kwargs):
        """
        Run the adjoint and gradient operator.

        Parameters
        ----------
        kwargs : dict
            Dictonary of keyword arguments.

        Returns
        ----------
        ndarray
            Gradient array.
        """

        # load the C library
        lib = self.library(            
            dtype=kwargs.get('velocity_model').dtype,
            operator="gradient",
            c_file=c_file
        )

        # get the argtype for each arg key
        types = self._argtypes(**kwargs)

        # get the all possible keys in order
        ordered_keys = self._keys_in_order

        # list of argtypes
        argtypes = []

        # list of args to pass to C function
        args = []

        # compose the list of args and arg types
        for key in ordered_keys:
            if kwargs.get(key) is not None:
                argtypes.append(types.get(key))
                args.append(kwargs.get(key))

        gradient = lib.gradient
        gradient.restype = ctypes.c_double
        gradient.argtypes = argtypes

        # run the C gradient function
        exec_time = gradient(*args)

        print('Run gradient in %f seconds.' % exec_time)

        return kwargs.get('grad')

    @property
    def _keys_in_order(self):
        """
        Return all possible arg keys in the expected order in the C function.
        """

        key_order = [
            'u',            
            'v',
            'grad',
            'velocity_model',            
            'damping_mask',
            'wavelet',
            'wavelet_size',
            'wavelet_count',
            'wavelet_adjoint',
            'wavelet_adjoint_size',
            'wavelet_adjoint_count',
            'second_order_fd_coefficients',
            'src_points_interval',            
            'src_points_interval_size',
            'src_points_values',
            'src_points_values_size',
            'src_points_values_offset',
            'rec_points_interval',
            'rec_points_interval_size',
            'rec_points_values',
            'rec_points_values_size',
            'rec_points_values_offset',
            'shot_record',
            'num_sources',
            'num_receivers',
            'nz',
            'nx',
            'ny',
            'dz',
            'dx',
            'dy',
            'saving_stride',            
            'dt',
            'begin_timestep',
            'end_timestep',
            'space_order',
            'num_snapshots'
        ]

        return key_order

    def _argtypes(self, **kwargs):
        """
        Get the ctypes argtypes of the keyword arguments.

        Parameters
        ----------
        kwargs : dict
            Dictonary of keyword arguments.

        Returns
        ----------
        dict
            Dictonary of argtypes with the same keys.
        """
        types = {}

        for key, value in kwargs.items():

            if isinstance(value, np.ndarray):
                types[key] = self._convert_type_to_ctypes(
                    'np({})'.format(str(value.dtype))
                )
            else:
                types[key] = self._convert_type_to_ctypes(
                    type(value).__name__
                )

        return types    

    def _convert_type_to_ctypes(self, type):
        """
        Convert a given type in python to a ctypes.argtypes format.

        Parameters
        ----------
        type : str
            Native type in python or numpy dtype.

        Returns
        ----------
        object
            Argtype in ctypes format.
        """
        argtype = {
            'int': ctypes.c_size_t,
            'float': ctypes.c_float,
            'float32': ctypes.c_float,
            'float64': ctypes.c_double,
            'np(uint64)': ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),
            'np(float32)': ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            'np(float64)': ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
        }

        return argtype[type]
