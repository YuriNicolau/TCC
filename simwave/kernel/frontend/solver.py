import numpy as np
from simwave.kernel.backend.middleware import Middleware
from simwave.kernel.frontend.source import MultiWavelet


class Solver:
    """
    Acoustic solver for the simulation.

    Parameters
    ----------
    space_model : SpaceModel
        Space model object.
    time_model: TimeModel
        Time model object.
    sources : Source
        Source object.
    receivers : Receiver
        Receiver object.
    wavelet : Wavelet
        Wavelet object.
    compiler : Compiler
        Backend compiler object.
    """
    def __init__(self, space_model, time_model, sources,
                 receivers, wavelet, compiler=None):

        self._space_model = space_model
        self._time_model = time_model
        self._sources = sources
        self._receivers = receivers
        self._wavelet = wavelet
        self._compiler = compiler

        # create a middleware to communicate with backend
        self._middleware = Middleware(compiler=self.compiler)

    @property
    def space_model(self):
        """Space model object."""
        return self._space_model

    @property
    def time_model(self):
        """Time model object."""
        return self._time_model

    @property
    def sources(self):
        """Source object."""
        return self._sources

    @property
    def receivers(self):
        """Receiver object."""
        return self._receivers

    @property
    def wavelet(self):
        """Wavelet object."""
        return self._wavelet

    @property
    def compiler(self):
        """Compiler object."""
        return self._compiler
    
    def snapshot_indexes(self, saving_stride):
        """List of snapshot indexes (wavefields to be saved)."""
        
        """This list starts on index 1, not 0"""
        
        # if saving_stride is 0, only saves the last timestep
        if saving_stride == 0:
            return np.int64([self.time_model.time_indexes[-1]+1])

        snap_indexes = list(
            range(
                self.time_model.time_indexes[1],
                self.time_model.timesteps+1,
                saving_stride
            )
        )

        return np.uint(snap_indexes)

    @property
    def shot_record(self):
        """Return the shot record array."""
        u_recv = np.zeros(
            shape=(self.time_model.timesteps, self.receivers.count),
            dtype=self.space_model.dtype
        )

        return u_recv
 
    def u(self):
        """Return u grid (3, nz. nx [, ny])."""
        
        # define the final shape (prev,current,next + domain)
        shape = (3,) + self.space_model.extended_shape

        return np.zeros(shape, dtype=self.space_model.dtype)
    
    def checkpoints(self, num_checkpoints):
        """Return the checkpoints/snapshots array."""
        
        # define the final shape (prev,current,next + domain)
        shape = (num_checkpoints,) + self.space_model.extended_shape

        return np.zeros(shape, dtype=self.space_model.dtype)
    
    def v(self):
        """Return the adjoint grid (3, nz. nx [, ny])."""

        # add 2 halo snapshots (second order in time)
        snapshots = 3

        # define the final shape (snapshots + domain)
        shape = (snapshots,) + self.space_model.extended_shape

        return np.zeros(shape, dtype=self.space_model.dtype)

    def grad(self):
        """Return the the gradient array."""

        # define the final shape (domain)
        shape = self.space_model.extended_shape

        return np.zeros(shape, dtype=self.space_model.dtype)
    
    def calc_saving_stride(self, num_checkpoints):
        # calculate saving stride
        if num_checkpoints == 0:
            saving_stride = 0
        else:
            #saving_stride = self.time_model.timesteps // num_checkpoints
            saving_stride = int( np.ceil(self.time_model.timesteps / num_checkpoints) )
            
        return saving_stride
    
    def forward_no_saving(self, c_file):
        """
        Run the forward propagator without checkpoints saving.
        
        Parameters
        ----------
        c_file: str
            Path to c_file
        
        Returns
        ----------
        ndarray
            Last wavefield.
        ndarray
            Shot record.
        """
        
        print("Model shape (with halo):", self.space_model.extended_shape)
        print("Model shape:", self.space_model.shape)        
        print("Number of timesteps:", self.time_model.timesteps)       
                
        src_points, src_values, src_offsets = \
            self.sources.interpolated_points_and_values
        rec_points, rec_values, rec_offsets = \
            self.receivers.interpolated_points_and_values

        u_last, recv = self._middleware.exec(
            operator='forward-no-saving',
            c_file=c_file,
            u=self.u(),            
            velocity_model=self.space_model.extended_velocity_model,            
            damping_mask=self.space_model.damping_mask,
            wavelet=self.wavelet.values,
            wavelet_size=self.wavelet.timesteps,
            wavelet_count=self.wavelet.num_sources,
            second_order_fd_coefficients=self.space_model.fd_coefficients(2),        
            src_points_interval=src_points,
            src_points_interval_size=len(src_points),
            src_points_values=src_values,
            src_points_values_offset=src_offsets,
            src_points_values_size=len(src_values),
            rec_points_interval=rec_points,
            rec_points_interval_size=len(rec_points),
            rec_points_values=rec_values,
            rec_points_values_offset=rec_offsets,
            rec_points_values_size=len(rec_values),
            shot_record=self.shot_record,
            num_sources=self.sources.count,
            num_receivers=self.receivers.count,
            grid_spacing=self.space_model.grid_spacing,            
            dt=self.time_model.dt,
            begin_timestep=1,
            end_timestep=self.time_model.timesteps,
            space_order=self.space_model.space_order,            
        )        

        # remove spatial halo region
        #u_last = self.space_model.remove_halo_region(u_last)
        
        # remove time halo region
        u_last = self.time_model.remove_time_halo_region(u_last)

        return u_last, recv

    def gradient(self, num_checkpoints, residual, c_file):
        """
        Run the the adjoint and gradient.

        Parameters
        ----------
        snapshots : ndarray
            Full wavefield checkpoints.
        residual : ndarray
            Difference between observed and predicted data.
        c_file: str
            Path to c_file

        Returns
        ----------
        ndarray
            Gradient array
        """

        saving_stride = self.calc_saving_stride(num_checkpoints)
        
        print("Calculating Checkpoints - Forward")
        
        print("Model shape (with halo):", self.space_model.extended_shape)
        print("Model shape:", self.space_model.shape)        
        print("Number of timesteps:", self.time_model.timesteps)
        print("Number of checkpoints:", num_checkpoints)
        print("Snapshot stride:", saving_stride)        
        print(f"Snapshot indexes ({len(self.snapshot_indexes(saving_stride))}):", self.snapshot_indexes(saving_stride) )

        src_points, src_values, src_offsets = \
            self.sources.interpolated_points_and_values
        rec_points, rec_values, rec_offsets = \
            self.receivers.interpolated_points_and_values

        # encapsulate the residual in a multi wavelet object
        adjoint_wavelet = MultiWavelet(
            values=residual,
            time_model=self.time_model
        )

        grad = self._middleware.exec(
            operator='gradient',
            c_file=c_file,            
            v=self.v(),
            grad=self.grad(),
            velocity_model=self.space_model.extended_velocity_model,            
            damping_mask=self.space_model.damping_mask,            
            wavelet=self.wavelet.values,
            wavelet_size=self.wavelet.timesteps,
            wavelet_count=self.wavelet.num_sources,            
            wavelet_adjoint=adjoint_wavelet.values,
            wavelet_adjoint_size=adjoint_wavelet.timesteps,
            wavelet_adjoint_count=adjoint_wavelet.num_sources,            
            second_order_fd_coefficients=self.space_model.fd_coefficients(2),        
            src_points_interval=src_points,
            src_points_interval_size=len(src_points),
            src_points_values=src_values,
            src_points_values_size=len(src_values),
            src_points_values_offset=src_offsets,
            rec_points_interval=rec_points,
            rec_points_interval_size=len(rec_points),
            rec_points_values=rec_values,            
            rec_points_values_size=len(rec_values),
            rec_points_values_offset=rec_offsets,            
            num_sources=self.sources.count,
            num_receivers=self.receivers.count,
            grid_spacing=self.space_model.grid_spacing,
            saving_stride=saving_stride,
            dt=self.time_model.dt,
            begin_timestep=1,
            end_timestep=self.time_model.timesteps,
            space_order=self.space_model.space_order,
            num_snapshots=num_checkpoints
        ) 
           
        # remove spatial halo region from gradient
        grad = self.space_model.remove_halo_region_from_gradient(grad)

        return grad
