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


def n_rectangles(fname):
    """
    Return number of rectangles
    Assumption: num. rectangles is stored in the first line of fname
    """
    with open(fname, "r") as file:
        n_rectangles_ = int(file.readline().strip())
    logger.debug(f"Number of rectangles: {n_rectangles_}")
    return n_rectangles_


def print_params(params, title=None):
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

