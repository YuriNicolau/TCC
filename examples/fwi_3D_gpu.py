"""Run FWI for Overthrust 3D model"""


import os
import tarfile
import zipfile
import re
import json
import time
import argparse
import urllib.request

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, zoom
from distributed import LocalCluster, Client, wait

from simwave import (
    SpaceModel, TimeModel, RickerWavelet, Solver, Compiler,
    Receiver, Source, plot_shotrecord, plot_velocity_model
)


# URL of the model
url = "https://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_models_cd/Salt_Model_3D.tar.gz"


def download_progress_hook(count, block_size, total_size):
    """A hook to report the progress of a download."""
    progress = count * block_size
    percent = int(progress * 100 / total_size)
    print(f"\rDownloading... {percent}%", end="")

def load_velocity_model_url(url):
    filename = os.path.basename(url)
    tar_filename = filename.split(".")[0] + ".tar.gz"
    zip_filename = "SALTF.ZIP"
    binary_filename = "Saltf@@"
    extract_folder = "ExtractedModel"

    # Download the file if it doesn't exist
    if not os.path.exists(tar_filename):
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, filename=tar_filename, reporthook=download_progress_hook)
        print()

    # Extract tar.gz file if the extraction hasn't already been done
    if not os.path.exists(extract_folder):
        print(f"Extracting {tar_filename}")
        with tarfile.open(tar_filename) as tar:
            tar.extractall(path=extract_folder)

    # Extract zip file if the extraction hasn't already been done
    if not os.path.exists(binary_filename):
        print(f"Extracting {zip_filename}")
        with zipfile.ZipFile(os.path.join(extract_folder, "Salt_Model_3D", "3-D_Salt_Model", "VEL_GRIDS", zip_filename), 'r') as zip_ref:
            zip_ref.extractall()

    # Read the binary file and convert to numpy array
    nx, ny, nz = 676, 676, 210
    with open(binary_filename, 'r') as file:
        vel = np.fromfile(file, dtype=np.dtype('float32').newbyteorder('>'))
        vel = vel.reshape(nx, ny, nz, order='F')

        # Cast type
        vel = np.asarray(vel, dtype=float)

    return vel

def load_velocity_model_npy(filename):
    if filename.endswith(".hdf5"):
        with h5py.File(filename) as f:
            vel = np.flip(np.asarray(f['velocity_model']), 0)
        return vel
    elif filename.endswith(".npy"):
        return np.load(filename)

def create_space_model(velocity_model, space_options):
    new_space_options = space_options.copy()
    if new_space_options["dtype"] in ["np.float32"]:
        new_space_options["dtype"] = np.float32
    else:
        new_space_options["dtype"] = None
    return SpaceModel(velocity_model=velocity_model, **new_space_options)

def create_time_model(space_model, time_options):
    return TimeModel(space_model=space_model, **time_options)

def create_sources(space_model, source_options, wr=4):
    return Source(space_model, window_radius=wr, **source_options)

def create_source(space_model, location, wr=4):
    return Source(space_model, window_radius=wr, coordinates=location)

def create_receivers(space_model, receiver_options, wr=4):
    return Receiver(space_model, window_radius=wr, **receiver_options)

def create_wavelet(time_model, wavelet_options):
    return RickerWavelet(time_model=time_model, **wavelet_options)

def create_solver(velocity_model, source_location, options):

    compiler_options = options["compiler_options"]
    space_options = options["space_options"]
    time_options = options["time_options"]
    boundary_condition_options = options["boundary_condition_options"]
    source_options = options["source_options"]
    receiver_options = options["receiver_options"]
    wavelet_options = options["wavelet_options"]

    space_model = create_space_model(velocity_model, space_options)
    space_model.config_boundary(**boundary_condition_options)
    time_model = create_time_model(space_model, time_options)
    source = create_source(space_model, source_location, wr=4)
    receivers = create_receivers(space_model, receiver_options, wr=4)
    wavelet = create_wavelet(time_model, wavelet_options)
    compiler = Compiler(cc=selected_compiler['cc'], cflags=selected_compiler['cflags'])

    return Solver(space_model,time_model,source,receivers,wavelet,compiler)

def forward_no_saving(velocity_model, source_location, options):
    fwd_file = options["solver_options"]["fwd_file"]
    solver = create_solver(velocity_model, source_location, options)
    return solver.forward_no_saving(c_file=fwd_file)

def generate_synthetic_serial(velocity_model, options):
    print("Generating synthetic data")
    recv = []
    for location in options["source_options"]["coordinates"]:
        recv.append(forward_no_saving(velocity_model, location, options)[1])
    return recv

def generate_synthetic_parallel(velocity_model, options):
    futures = []
    big_future = client.scatter(velocity_model)
    for location in options["source_options"]["coordinates"]:
        futures.append(client.submit(forward_no_saving, big_future, location, options))
    # wait(futures)
    return [future.result()[1] for future in futures]

def obj_fun(velocity_model, source_location, recv_true, options):
    fwd_file = options["solver_options"]["fwd_file"]
    num_checkpoints = options["solver_options"]["num_checkpoints"]
    solver = create_solver(velocity_model, source_location, options)
    # run the forward computation
    print("Evaluating forward problem")
    _, recv = solver.forward_no_saving(c_file=fwd_file)
    res = recv - recv_true
    f_obj = 0.5 * np.linalg.norm(res) ** 2
    # gradient compute the gradient
    if args.forward:
        grad = 0
    else:
        print("Evaluating gradient")
        grad = solver.gradient(num_checkpoints=num_checkpoints, residual=res, c_file=selected_compiler['path'])

    return f_obj, grad

def eval_misfit_and_gradient_parallel(velocity_model, recv_all, options):
    futures = []
    big_future = client.scatter(velocity_model)
    locations = options["source_options"]["coordinates"]
    for location, recv in zip(locations, recv_all):
        futures.append(
            client.submit(
                obj_fun, big_future, location, recv, options
            )
        )
    # wait(futures)
    f_obj, grad = 0, 0
    for future in futures:
        f_obj += future.result()[0]
        grad += future.result()[1]
    return f_obj, crop(grad, options)

def eval_misfit_and_gradient_serial(velocity_model, recv_all, options):
    f_obj, grad = 0, 0
    locations = options["source_options"]["coordinates"]
    for location, recv in zip(locations, recv_all):
        f, g = obj_fun(velocity_model, location, recv, options)
        f_obj += f
        grad += g
    if args.forward:
        return f_obj, grad
    else:
        return f_obj, crop(grad, options)

def crop(u, options):
    grid_spacing = options["space_options"]["grid_spacing"]
    damping_length = options["boundary_condition_options"]["damping_length"]
    return u[
        damping_length[0]//grid_spacing[0]:-damping_length[1]//grid_spacing[0],
        damping_length[2]//grid_spacing[1]:-damping_length[3]//grid_spacing[1],
        damping_length[4]//grid_spacing[2]:-damping_length[5]//grid_spacing[2]]

def fun(velocity_model, recv_all, options):
    space_options = options["space_options"]
    velocity_model = velocity_model.reshape(eval_shape(space_options))
    mode = options["solver_options"]["mode"]

    start_time = time.time()
    if mode in ["parallel"]:
        f, g = eval_misfit_and_gradient_parallel(velocity_model, recv_all, options)
    if mode in ["serial"]:
        f, g = eval_misfit_and_gradient_serial(velocity_model, recv_all, options)
    else:
        print("mode has to be either 'parallel' or 'serial'")
    total_time = time.time() - start_time

    dirname = options["plot_options"]["dirname"]
    slice_pos = options["plot_options"]["slice_pos"]
    model_name = options["plot_options"]["name"] + "_" + get_filename(dirname, "model")
    grad_name = options["plot_options"]["name"] + "_" + get_filename(dirname, "grad")
    if not args.forward:
        plot_velocity_model(g[slice_pos,:,:], file_name=grad_name, dirname=dirname)
    plot_velocity_model(velocity_model[slice_pos,:,:], file_name=model_name, dirname=dirname)

    dump_internal_loop(f, dirname)
    dump_time(total_time, dirname)
    if args.forward or args.gradient:
        exit()

    if args.forward:
        return f, options["solver_options"]["mul"] * g
    else:
        return f, options["solver_options"]["mul"] * g.flatten().astype(np.float64)

def dump_internal_loop(f, dirname):
    f_his_internal.append(f)
    filename = dirname+"/obj_fun_internal_history.npy"
    np.save(filename, f_his_internal)
    with open(filename.replace('.npy','.log'), "a") as fl:
        fl.write(f"{f}\n")
        pass

def dump_time(t, dirname):
    t = f"{t // 60} min {t % 60} sec"
    t_his.append(t)
    filename = dirname+"/time_history.npy"
    np.save(filename, t_his)
    with open(filename.replace('.npy','.log'), "a") as fl:
        fl.write(f"Evaluation time: {t} \n")
        pass
def dump_internal_loop(f, dirname):
    f_his_internal.append(f)
    filename = dirname+"/obj_fun_internal_history.npy"
    np.save(filename, f_his_internal)
    with open(filename.replace('.npy','.log'), "a") as fl:
        fl.write(f"{f}\n")
        pass
def dump_external_loop(f, dirname):
    f_his_external.append(f)
    filename = dirname+"/obj_fun_external_history.npy"
    np.save(filename, f_his_external)
    with open(filename.replace('.npy','.log'), "a") as fl:
        fl.write(f"{f}\n")
        pass

def callback(xk):
    dump_external_loop(f_his_internal[-1], dirname)
    np.save(dirname+"/xk_result.npy", xk)
    plt.plot(f_his_external)
    plt.xlabel("Iterations")
    plt.ylabel("Misfit")
    plt.savefig(dirname+"/obj_fun")

def eval_shape(space_opts):
    h = space_opts["grid_spacing"]
    bbox = space_opts["bounding_box"]
    nz = (bbox[1] - bbox[0]) / h[0] + 1
    nx = (bbox[3] - bbox[2]) / h[1] + 1
    ny = (bbox[5] - bbox[4]) / h[2] + 1
    return (int(nz), int(nx), int(ny))

def create_sphere_velocity_model(space_options):
    shape = eval_shape(space_options)
    if space_options["dtype"] in ["np.float32"]:
        dtype = np.float32
    else:
        dtype = None
    vel = 2500 * np.ones(shape, dtype=dtype)
    a, b, c = shape[0] / 2, shape[1] / 2, shape[2] / 2
    z, x, y = np.ogrid[-a:shape[0]-a, -b:shape[1]-b, -c:shape[2]-c]
    vel[z*z + x*x + y*y <= 15*15] = 3000
    return vel

def check_file(cur_dir, file_pattern):
    files = [] 
    for file_name in os.listdir(cur_dir):
        if file_pattern.match(file_name):
            files.append(file_name)
    return files

def max_file(dir_list):
    maximum_dir = max(dir_list, key=lambda x: int(x.rstrip('.png').split('_')[-1]))
    return int(maximum_dir.rstrip('.png').split('_')[-1])

def get_filename(dirname, name_pattern):
    file_pattern = re.compile(r".*_" + name_pattern + r"_.*")
    files = check_file(dirname, file_pattern)

    if files:
        new_file = name_pattern + "_" + str(max_file(files) + 1)
    else:
        new_file = name_pattern + "_" + str(0)

    return new_file

def check_dir(dir_pattern):
    dirs = [] 
    for dir_name in os.listdir():
        if os.path.isdir(dir_name):
            if dir_pattern.match(dir_name):
                dirs.append(dir_name)
    return dirs

def max_dir(dir_list):
    maximum_dir = max(dir_list, key=lambda x: int(x.split('_')[-1]))
    return int(maximum_dir.split('_')[-1])

def create_out_dir(dirname):
    dir_pattern = re.compile(dirname+r'_\d+')
    dirs = check_dir(dir_pattern)
    if dirs:
        new_dir = dirname + "_" + str(max_dir(dirs) + 1)
    else:
        new_dir = dirname + "_" + str(0)
    return new_dir
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""forward/fwi example script,
        accepts optional parameters for specifying problem.
        Run 'python fwi_3D_GPU.py help for details.""")
    parser.add_argument("-C", default="no-compression", help="compression option")
    parser.add_argument("--forward", action="store_true", help="if true, runs forward problem once")
    parser.add_argument("--gradient", action="store_true", help="if true, runs forward and gradient evaluation once")
    parser.add_argument("--fwi", action="store_true", help="if true, runs inversion")
    parser.add_argument("--num-shots-x", default=6, help="number of shots along x direction")
    parser.add_argument("--num-shots-y", default=6, help="number of shots along y direction")
    parser.add_argument("-n", default=2000, help="number of checkpoints")
    parser.add_argument("-dx", default="80", help="grid spacing in x direction")
    parser.add_argument("-dy", default="80", help="grid spacing in y direction")
    parser.add_argument("-dz", default="60", help="grid spacing in z direction")
    args = parser.parse_args()

    dirname = create_out_dir('experiment_salt_3D')
    #dirname = "fwi_checkpoints_"+args.n+"_"+args.C+"_dx"+args.nx+"_dy"+args.ny+"_dz"+args.nz

    compiler_options = {

    'compression': {
        'cc': 'clang++',
        
        'cflags': '-O3 -fPIC -ffast-math -fopenmp -std=c++11\
                   -shared -fopenmp-targets=nvptx64-nvidia-cuda \
                   -Xopenmp-target -march=sm_70 \
                   -DGPU_OPENMP -DDEVICEID=3 \
                   -L/usr/local/cuda/lib64 -lcudart \
                   -I/usr/local/cuda/targets/x86_64-linux/include \
                   -isystem ../simwave/kernel/backend/include \
                   -rpath ../simwave/kernel/backend/lib \
                    ../simwave/kernel/backend/lib/libnvcomp.so',
                    
        'path': '../simwave/kernel/backend/c_code/compression/gradient.cpp'
    },
    
    'no-compression': {
        'cc': 'clang++',
        
        'cflags': '-O3 -fPIC -ffast-math -fopenmp -std=c++11\
                   -shared -fopenmp-targets=nvptx64-nvidia-cuda \
                   -Xopenmp-target -march=sm_70 \
                   -DGPU_OPENMP -DDEVICEID=3',
                   
        'path': '../simwave/kernel/backend/c_code/normal/gradient.cpp'
    },
    }
    space_options = {
        "bounding_box": (0, 13500, 0, 13500, 0, 4180),
        "grid_spacing": (int(args.dx), int(args.dy), int(args.dz)),
        "space_order": 4,
        "dtype": "np.float32"
    }
    time_options = {
        "tf": 10.0,
        "dt": 0.001
    }
    boundary_condition_options = {
        "damping_length": (480, 480, 480, 480, 480, 480),
        "damping_polynomial_degree":3,
        "damping_alpha":0.002
    }
    source_options = {
        "coordinates": [(i, j, 4180) for i in np.linspace(0, 13500, int(args.num_shots_x)) 
                                     for j in np.linspace(0, 13500, int(args.num_shots_y))]
    }
    receiver_options = {
	"coordinates": [(i, j, 4080) for i in np.linspace(0, 13500, 101) for j in np.linspace(0, 13500, 101)],
    }
    wavelet_options = {
        "peak_frequency": 2,
        "amplitude": 1
    }
    solver_options = {
        "fwd_file": "../simwave/kernel/backend/c_code/forward.cpp",
        "num_checkpoints": int(args.n),
        "mode": "serial",
        "mul": -1e-9
    }
    plot_options = {
        "slice_pos": 84,
        "dirname": dirname,
        "name": "EAGE"
    }
    options = {
        "compiler_options": compiler_options, 
        "space_options": space_options, 
        "time_options": time_options, 
        "boundary_condition_options": boundary_condition_options, 
        "source_options": source_options, 
        "receiver_options": receiver_options, 
        "wavelet_options": wavelet_options, 
        "solver_options": solver_options, 
        "plot_options": plot_options
    }

    # choose compression
    selected_compiler = compiler_options[args.C]

    print(f"number of checkpoints n = {args.n}")
    print(f"compression type C = {args.C}")
    print(f"dirname = {dirname}")
    print(f"number of shots = {int(args.num_shots_x) * int(args.num_shots_y)}")
    if args.forward:
        print("Running forward execution once and then quiting")
    elif args.gradient:
        print("Running gradient execution once and then quiting")
    else:
        print("Run inversion")

    # velocity_models
    # vp = load_velocity_model_npy("EAGE_3D_SALT.npy")
    vp = load_velocity_model_url(url)
    vp = resize(vp, eval_shape(space_options))
    vp_guess = gaussian_filter(vp, sigma=5)

    # plot initial guess and reference model
    slice_pos = options["plot_options"]["slice_pos"]
    plot_velocity_model(vp[slice_pos,:,:], file_name="ref", dirname=dirname)
    plot_velocity_model(vp_guess[slice_pos,:,:], file_name="init", dirname=dirname)
    # save options as dict
    dict_name = plot_options["name"]
    with open(os.path.join(dirname, dict_name)+".json", "w") as f:
        json.dump(options, f, indent=2)

    if options["solver_options"]["mode"] in ["parallel"]:
        cluster = LocalCluster(n_workers=4, threads_per_worker=9, death_timeout=600)
        client = Client(cluster)

    recv_all = generate_synthetic_serial(vp, options)

    # Optimization
    f_his_internal = []
    f_his_external = []
    t_his = []
    res = minimize(
        fun,
        vp_guess.flatten(),
        method="L-BFGS-B",
        args=(recv_all, options),
        jac=True,
        callback=callback,
        bounds=[(1500, 4482) for _ in vp_guess.flatten()],
        options={
            "disp": True,
            "iprint": 1,
            "maxiter": 0,
            "maxls": 1
        }
    )
    
