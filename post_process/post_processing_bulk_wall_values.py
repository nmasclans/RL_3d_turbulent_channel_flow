#!/usr/bin/python

import sys
import os
import numpy as np
import h5py    
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib import rc,rcParams
import matplotlib.colors as colors
from matplotlib import ticker
import matplotlib.cm as cm

try :
    iteration  = sys.argv[1]
except : 
    exit(0)


### Open data file
data_file = h5py.File( f'RL_3d_turbulent_channel_flow_{iteration}.h5', 'r' )
#list( data_file.keys() )
x_data        = data_file['x'][:,:,:]
y_data        = data_file['y'][:,:,:]
z_data        = data_file['z'][:,:,:]
u_data        = data_file['u'][:,:,:]
avg_u_data    = data_file['avg_u'][:,:,:]
num_points_x  = u_data[0,0,:].size
num_points_y  = u_data[0,:,0].size
num_points_z  = u_data[:,0,0].size


### Reference parameters
rho_0  = 1.0				# Reference density [kg/m3]
u_tau  = 1.0				# Friction velocity [m/s]
delta  = 1.0				# Channel half-height [m]
Re_tau = 100.0				# Friction Reynolds number [-]
mu_ref = rho_0*u_tau*delta/Re_tau	# Dynamic viscosity [Pa s]
nu_ref = mu_ref/rho_0			# Kinematic viscosity [m2/s]
L_x = 4.0*np.pi*delta


### Average variables in space
sum_avg_u_volume = 0.0
sum_volume = 0.0
for i in range( 1, num_points_x-1 ):
    for j in range( 1, num_points_y-1 ):
        for k in range( 1, num_points_z-1 ):
            delta_x = 0.5*( x_data[k,j,i+1] - x_data[k,j,i-1] )
            delta_y = 0.5*( y_data[k,j+1,i] - y_data[k,j-1,i] )
            delta_z = 0.5*( z_data[k+1,j,i] - z_data[k-1,j,i] )
            volume = delta_x*delta_y*delta_z
            sum_avg_u_volume += volume*avg_u_data[k,j,i]
            sum_volume += volume
U_b = sum_avg_u_volume/sum_volume
print( "Numerical U_bulk:", U_b )
print( "Numerical L_x / U_bulk:", L_x/U_b )

### Average variables in space
sum_avg_u_inner = 0.0
sum_avg_u_boundary = 0.0
sum_surface = 0.0
for i in range( 1, num_points_x-1 ):
    for k in range( 1, num_points_z-1 ):
        # Bottom wall
        j = 0
        delta_x = 0.5*( x_data[k,j,i+1] - x_data[k,j,i-1] )
        delta_z = 0.5*( z_data[k+1,j,i] - z_data[k-1,j,i] )
        delta_surface = delta_x*delta_z
        sum_surface += delta_surface
        sum_avg_u_inner += avg_u_data[k,j+1,i] * delta_surface
        sum_avg_u_boundary += avg_u_data[k,j,i] * delta_surface
        # Top wall
        j = num_points_y - 1
        delta_x = 0.5*( x_data[k,j,i+1] - x_data[k,j,i-1] )
        delta_z = 0.5*( z_data[k+1,j,i] - z_data[k-1,j,i] )
        delta_surface = delta_x*delta_z
        sum_surface += delta_surface
        sum_avg_u_inner += avg_u_data[k,j-1,i] * delta_surface
        sum_avg_u_boundary += avg_u_data[k,j,i] * delta_surface
avg_u_inner = sum_avg_u_inner / sum_surface
avg_u_boundary = sum_avg_u_boundary/ sum_surface
tau_w_num = mu_ref * (avg_u_inner - avg_u_boundary) / (y_data[0,1,0] - y_data[0,0,0])
u_tau_num = np.sqrt(tau_w_num / rho_0) 
print("Numerical tau_w:", tau_w_num)
print("Numerical u_tau:", u_tau_num)
