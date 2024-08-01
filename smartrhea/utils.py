import numpy as np

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
    return sum([1 for l in open(fname,"r").readlines() if l.strip()])  # num. witness points = num. lines of fname


def n_rectangles(fname):
    return int(open(fname,"r").readline())  # first line of fname contains info num. rectangles


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


def numpy_str(a, precision=2):
    return np.array2string(a, precision=precision, floatmode='fixed')


# --- TensorFlow utils ---
def deactivate_tf_gpus():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'

