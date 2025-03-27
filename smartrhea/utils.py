import numpy as np
import os
import shutil
from smartsim.log import get_logger

logger = get_logger(__name__)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def n_points(fname):
    """
    Return number of witness/control points
    Assumption: number of lines in fname is equal to the number of witness/control points (ignore lines with only whitespaces)
    """
    logger.debug(f"[utils:n_points] Filename: {fname}")
    with open(fname, "r") as file:
        n_points_ = int(sum(1 for l in file if l.strip))
    logger.debug(f"Number of witness/control points: {n_points_}")
    return n_points_


def n_cubes(fname):
    """
    Return number of control cubes
    Assumption: num. cubes is stored in the first line of fname
    """
    with open(fname, "r") as file:
        n_cubes_ = int(file.readline().strip())
    logger.debug(f"Number of cubes: {n_cubes_}")
    return n_cubes_


def check_points_xyz(fname):
    """
    Make sure the points are written in such that the first moving coordinate is z, then x, and last y.
    Arguments:
        fname (str): witness/control points filename
    """
    logger.debug(f"[utils:check_points_xyz] Filename: {fname}")
    points_list = np.loadtxt(fname)
    prev_point = points_list[0]
    ### Check if z changes first, then x, then y
    for point in points_list[1:]:
        # Ensure z changes first (x,y fixed)
        if point[0] == prev_point[0] and point[1] == prev_point[1]:
            assert point[2] >= prev_point[2], "Error: z should increase first, then y, then x"
        # If z is reset (with different combination (x,y)), x should change next (y fixed)
        elif point[1] == prev_point[1]:
            assert point[0] >= prev_point[0], "Error: x should increase after z, then y"
        # If both x and z are reset, y should change last
        else:
            assert point[1] >= prev_point[1], "Error: y should increase after x and z."
        # Update previous point
        prev_point = point
    logger.debug("Successfull witness/control points check: points are written in such that the first moving coordinate is z, then x, and last y.")


def get_points_xyz(fname):
    """
    Gets the number of points in the (x, y, x) directions
    Arguments:
        fname (str): witness points filename
    """
    check_points_xyz(fname)
    points_list = np.loadtxt(fname)
    unique_x = np.unique(points_list[:,0])
    unique_y = np.unique(points_list[:,1])
    unique_z = np.unique(points_list[:,2])
    points_xyz = [len(unique_x), len(unique_y), len(unique_z)]
    logger.debug(f"Witness/control points in the (x, y, z) directions: \n{points_xyz}")
    return points_xyz


def print_params(params, title=None):
    print("")
    if title: print(f"{bcolors.OKGREEN}{title}{bcolors.ENDC}")
    print(params_str(params))
    

def params_str(params, title=None):
    my_str = ""
    if title:
        my_str += "title\n"
    for k,v in params.items():
        my_str += f"\n{k}: {v}"
    my_str += "\n"
    return my_str

def params_html_table(params, title=None):
    my_str = "<table>"
    if title:
        my_str += f"<tr><th colspan='2'>{title}</th></tr>"
    for k, v in params.items():
        my_str += f"<tr><td>{k}</td><td>{v}</td></tr>"
    my_str += "</table>"
    return my_str

def numpy_str(a, precision=2):
    return np.array2string(a, precision=precision, floatmode='fixed')

# Delete all files inside the directory
def delete_all_files_in_dir(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and all its contents
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


# --- TensorFlow utils ---
def deactivate_tf_gpus():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'


