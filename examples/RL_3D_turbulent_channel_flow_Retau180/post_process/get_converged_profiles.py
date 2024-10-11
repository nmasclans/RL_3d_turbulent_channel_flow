#!/home/jofre/miniconda3/envs/smartrhea-env-v2/bin/python3

import sys
import os
import glob
import numpy as np
import h5py    
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib import rc,rcParams
import matplotlib.colors as colors
from matplotlib import ticker
import matplotlib.cm as cm

# --- Get Converged Reference data ---

print("\nImporting reference data...")
filename = "reference_data/3d_turbulent_channel_flow_reference.h5"
with h5py.File( filename, 'r' ) as data_file:
    averaging_time = data_file.attrs["AveragingTime"][0]
    time           = data_file.attrs["Time"][0]
    y_data         = data_file['y'][:,:,:]
    rmsf_u_data    = data_file['rmsf_u'][:,:,:]
    rmsf_v_data    = data_file['rmsf_v'][:,:,:]
    rmsf_w_data    = data_file['rmsf_w'][:,:,:]
    num_points_x   = rmsf_u_data[0,0,:].size
    num_points_y   = rmsf_u_data[0,:,0].size
    num_points_z   = rmsf_u_data[:,0,0].size
    num_points_xz  = num_points_x * num_points_z
print(f"Data imported from filename: '{filename}', with:")
print(f"Averaging Time: {averaging_time}")
print(f"Time: {time}")

print("\nAveraging variables in space...")
y = np.zeros([num_points_y])
rmsf_u = np.zeros([num_points_y])
rmsf_v = np.zeros([num_points_y])
rmsf_w = np.zeros([num_points_y])
for j in range( 0, num_points_y ):
    y[j] = y_data[0,j,0]
    for i in range( 0, num_points_x ):
        for k in range( 0, num_points_z ):
            rmsf_u[j] += (1/num_points_xz) * rmsf_u_data[k,j,i]
            rmsf_v[j] += (1/num_points_xz) * rmsf_v_data[k,j,i]
            rmsf_w[j] += (1/num_points_xz) * rmsf_w_data[k,j,i]
            assert y[j] == y_data[k,j,i], f"Not uniform grid, with y[j]={y[j]} != y_data[k,j,i]={y_data[k,j,i]}"
tke = 0.5 * (rmsf_u**2 + rmsf_v**2 + rmsf_w**2)

print("\nRESULT: Calculated profiles:")
print("\ny-coordinate", y)
print("\nrmsf_u:", rmsf_u)
print("\nrmsf_v:", rmsf_v)
print("\nrmsf_w:", rmsf_w)
print("\ntke:", tke)
