import numpy as np
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


def n_witness_points(fname):
    """
    Return number of witness points
    Assumption: number of lines in fname is equal to the number of witness points (ignore lines with only whitespaces)
    """
    with open(fname, "r") as file:
        n_witness_points_ = int(sum(1 for l in file if l.strip))
    logger.debug(f"Number of witness points: {n_witness_points_}")
    return n_witness_points_


def n_cubes(fname):
    """
    Return number of control cubes
    Assumption: num. cubes is stored in the first line of fname
    """
    with open(fname, "r") as file:
        n_cubes_ = int(file.readline().strip())
    logger.debug(f"Number of cubes: {n_cubes_}")
    return n_cubes_


def check_witness_xyz(fname):
    """
    Make sure the witness points are written in such that the first moving coordinate is x, then y, and last z.
    Arguments:
        fname (str): witness points filename
    """
    witness_points = np.loadtxt(fname)
    # Check if x changes first, then y, then z
    prev_point = witness_points[0]
    for point in witness_points[1:]:
        logger.debug(f"prev_point: {prev_point}, point: {point}")
        # Ensure x changes first (y,z fixed)
        if point[2] == prev_point[2] and point[1] == prev_point[1]:
            assert point[0] >= prev_point[0], "Error: x should increase first, then y, then z"
        # If x is reset (with different combination (y,z)), y should change next (z fixed)
        elif point[2] == prev_point[2]:
            assert point[1] >= prev_point[1], "Error: y should increase after x, then z"
        # If both x and y are reset, z should change last
        else:
            assert point[2] >= prev_point[2], "Error: z should increase after x and y."
        # Update previous point
        prev_point = point
    logger.debug("Successfull witness points check: witness points are written in such that the first moving coordinate is x, then y, and last z.")


def get_witness_xyz(fname):
    """
    Gets the number of witness points in the (x, y, x) directions
    Arguments:
        fname (str): witness points filename
    """
    check_witness_xyz(fname)
    witness_points = np.loadtxt(fname)
    unique_x = np.unique(witness_points[:,0])
    unique_y = np.unique(witness_points[:,1])
    unique_z = np.unique(witness_points[:,2])
    witness_xyz = [len(unique_x), len(unique_y), len(unique_z)]
    logger.debug(f"Number of witness points in the (x, y, z) directions: {witness_xyz}")
    return witness_xyz


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


# --- TensorFlow utils ---
def deactivate_tf_gpus():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'

