import h5py
import numpy as np
import math
import os

#-----------------------------------------------------------------------------------------
#           Anisotropy tensor, eigen-decomposition, mapping to barycentric map 
#-----------------------------------------------------------------------------------------

def check_realizability_conditions(Rxx, Ryy, Rzz, Rxy, Rxz, Ryz, verbose=False):

    #------------ Realizability conditions ---------------

    # help: .all() ensures the condition is satisfied in all grid points

    # COND 1: Rii >= 0, for i = 1,2,3

    cond0_0 = ( Rxx >= 0 ).all()    # i = 1
    cond0_1 = ( Ryy >= 0 ).all()    # i = 2
    cond0_2 = ( Rzz >= 0 ).all()    # i = 3
    cond0   = cond0_0 and cond0_1 and cond0_2

    # COND 2: Rij^2 <= Rii*Rjj, for i!=j

    cond1_0 = ( Rxy**2 <= Rxx * Ryy ).all()     # i = 0, j = 1
    cond1_1 = ( Rxz**2 <= Rxx * Rzz ).all()     # i = 0, j = 2
    cond1_2 = ( Ryz**2 <= Ryy * Rzz ).all()     # i = 1, j = 2
    cond1   = cond1_0 and cond1_1 and cond1_2

    # COND 3: det(Rij) >= 0
    detR  = Rxx * Ryy * Rzz + 2 * Rxy * Rxz * Ryz - (Rxx * Ryz * Ryz + Ryy * Rxz * Rxz + Rzz * Rxy * Rxy) # np.linalg.det(R_ij), length #num_points
    # detR  = np.linalg.det(R_ij)
    # enforce detR == 0 if -eps < detR < 0 due to computational error  
    for i in range(len(detR)):
        if detR[i] > -1e-12 and detR[i] < 0.0:
            detR[i] = 0.0
    cond2 = ( detR >= 0 ).all()

    if cond0 and cond1 and cond2:
        if verbose:
            print("\nCONGRATULATIONS, the reynolds stress tensor satisfies REALIZABILITY CONDITIONS.")
    else:
        message = f"ATTENTION!!! \nThe reynolds stress tensor does not satisfy REALIZABILITY CONDITIONS: cond0 = {cond0}, cond1 = {cond1}, cond2 = {cond2}"
        print(message)
        # raise Exception(message)

    
def compute_reynolds_stress_dof(Rxx, Rxy, Rxz, Ryy, Ryz, Rzz, 
                                tensor_kk_tolerance   = 1.0e-8, 
                                eigenvalues_tolerance = 1.0e-8, 
                                verbose = False,
                                x1c = np.array( [ 1.0 , 0.0 ] ),
                                x2c = np.array( [ 0.0 , 0.0 ] ),
                                x3c = np.array( [ 0.5 , math.sqrt(3.0)/2.0 ] )):

    check_realizability_conditions(Rxx, Ryy, Rzz, Rxy, Rxz, Ryz, verbose)
    
    # Computed for each point of the grid
    # If the trace of the reynolds stress tensor (2 * TKE) is too small, the corresponding 
    # datapoint is omitted, because the anisotropy tensor would -> infinity, as its equation
    # contains the multiplier ( 1 / (2*TKE) )
    
    # initialize arrays
    num_points = len(Rxx)
    Rkk     = np.zeros(num_points)
    lambda1 = np.zeros(num_points)
    lambda2 = np.zeros(num_points)
    lambda3 = np.zeros(num_points)
    xmap1   = np.zeros(num_points)
    xmap2   = np.zeros(num_points)

    for p in range(num_points):

        #------------ Reynolds stress tensor ---------------

        R_ij      = np.zeros([3, 3])
        R_ij[0,0] = Rxx[p]
        R_ij[0,1] = Rxy[p]
        R_ij[0,2] = Rxz[p]
        R_ij[1,0] = Rxy[p]
        R_ij[1,1] = Ryy[p]
        R_ij[1,2] = Ryz[p]
        R_ij[2,0] = Rxz[p]
        R_ij[2,1] = Ryz[p]
        R_ij[2,2] = Rzz[p]

        #------------ Anisotropy Tensor ------------

        # identity tensor
        delta_ij = np.eye(3)                                        # shape: [3,3]

        # calculate trace -> 2 * (Turbulent kinetic energy)
        Rkk[p] = R_ij[0,0] + R_ij[1,1] + R_ij[2,2]
        TKE = 0.5 * Rkk[p] #  -> same formula!                      # shape: scalar
        ###TKE = 0.5 * (urmsf[p]**2 + vrmsf[p]**2 + wrmsf[p]**2)    # shape: scalar

        # omit grid point if reynolds stress tensor trace (2 * TKE) is too small
        discarded_points = []
        if np.abs(Rkk[p]) < tensor_kk_tolerance:
            discarded_points.append(p)
            continue
        if len(discarded_points)>0:
            print(f"Discarded points id:", discarded_points)

        # construct anisotropy tensor
        a_ij = (1.0 / (2*TKE)) * R_ij - (1.0 / 3.0) * delta_ij   # shape: [3,3]

        #------------ eigen-decomposition of the SYMMETRIC TRACE-FREE anisotropy tensor ------------

        # ensure a_ij is trace-free
        # -> calculate trace
        a_kk = a_ij[0,0] + a_ij[1,1] + a_ij[2,2]
        # -> substract the trace
        a_ij[0,0] -= a_kk/3.0
        a_ij[1,1] -= a_kk/3.0
        a_ij[2,2] -= a_kk/3.0

        # Calculate the eigenvalues and eigenvectors
        eigenvalues_a_ij, eigenvectors_a_ij = np.linalg.eigh( a_ij )
        eigenvalues_a_ij_sum = sum(eigenvalues_a_ij)
        assert eigenvalues_a_ij_sum < eigenvalues_tolerance, f"ERROR: The sum of the anisotropy tensor eigenvalues should be 0; in point #{p} the sum is = {eigenvalues_a_ij_sum}"

        # Sort eigenvalues and eigenvectors in decreasing order, so that eigval_1 >= eigval_2 >= eigval_3
        idx = eigenvalues_a_ij.argsort()[::-1]   
        eigenvalues_a_ij  = eigenvalues_a_ij[idx]
        eigenvectors_a_ij = eigenvectors_a_ij[:,idx]
        (lambda1[p], lambda2[p], lambda3[p]) = eigenvalues_a_ij

        if verbose:
            inspected_eigenvalue = (-Rxx[p]+Ryy[p]-3*Ryz[p])/(3*Rxx[p]+6*Ryy[p])
            print(f"\nPoint p = {p}")
            print(f"3rd eigenvalue lambda_2 = {eigenvalues_a_ij[2]}")
            print(f"3rd eigenvector v_2     = {eigenvectors_a_ij[:,2]}")
            print(f"(expected from equations) \lambda_2 = (-R_00+R_11-3R_12)/(3R_00+6R_11) = {inspected_eigenvalue}")
            print(f"(expected from equations) v_2 = (0, -1, 1)$, not normalized")
            print(f"R_11 = {Ryy[p]:.5f}, R_12 = {Ryz[p]:.5f}")

        # Calculate Barycentric map point
        # where eigenvalues_a_ij[0] >= eigenvalues_a_ij[1] >= eigenvalues_a_ij[2] (eigval in decreasing order)
        bar_map_xy =   x1c * (     eigenvalues_a_ij[0] -     eigenvalues_a_ij[1]) \
                     + x2c * ( 2 * eigenvalues_a_ij[1] - 2 * eigenvalues_a_ij[2]) \
                     + x3c * ( 3 * eigenvalues_a_ij[2] + 1)
        xmap1[p]   = bar_map_xy[0]
        xmap2[p]   = bar_map_xy[1]

    return (Rkk, lambda1, lambda2, lambda3, xmap1, xmap2)

#-----------------------------------------------------------------------------------------
#           Temporal probes at (y,z) coordinate pair along x-direction 
#-----------------------------------------------------------------------------------------


def build_probelines_csv_from_snapshot_h5(
    input_h5_filepath, file_details, output_csv_directory, num_grid_x, num_grid_y, dt_dx, n_probes, probes_zy_desired
):

    # Get data: x, y, z, u, v, w; attributes: time
    with h5py.File(input_h5_filepath, 'r') as file:
        t0     = file.attrs['Time']
        x_data = file['x'][1:-1,1:-1,1:-1]    # 1:-1 to take only inner grid points
        y_data = file['y'][1:-1,1:-1,1:-1]
        z_data = file['z'][1:-1,1:-1,1:-1]
        u_data = file['u'][1:-1,1:-1,1:-1]
        v_data = file['v'][1:-1,1:-1,1:-1]
        w_data = file['w'][1:-1,1:-1,1:-1]
    num_points_z, num_points_y, num_points_x = u_data.shape
    assert num_points_x == num_grid_x & num_points_y == num_grid_y
    
    # Convert dx -> dt --> rebuild 'x_data' and 'time_data' to translate spatial advancement to temporal advancement
    x0 = x_data[0,0,0]
    dx_data = x_data - x0             # shape [num_points_z, num_points_y, num_points_x]
    t_data  = t0 + dt_dx * dx_data    # shape [num_points_z, num_points_y, num_points_x]
    x_data  = x0 * np.ones([num_points_z, num_points_y, num_points_x])

    # Find probes (z,y) coordinates closest to chosen 'zy_probes' pairs of coordinates
    # ASSUMPTION: regular grid!
    probes_zy = np.zeros([n_probes,2])                  # (z,y) coordinates pairs
    probes_kj = np.zeros([n_probes,2], dtype='int64')   # (k,j) indexes paris (on z,y-directions) 
    y_coords = y_data[0,:,0]
    z_coords = z_data[:,0,0] 
    for i_probe in range(n_probes):
        [z_i, y_i] = probes_zy_desired[i_probe,:]
        k_i = np.argmin(np.abs(z_coords - z_i));    z_i = z_coords[k_i]
        j_i = np.argmin(np.abs(y_coords - y_i));    y_i = y_coords[j_i]
        probes_kj[i_probe,:] = [k_i, j_i];          probes_zy[i_probe,:] = [z_i, y_i]
    print(f"\nDesired probes (z,y)-coordinates: \n{probes_zy_desired}")
    print(f"\nFound probes (z,y)-coordinates: \n{probes_zy}")
    print(f"\nFound probes (k,z)-index for (z,y)-coordinates: \n{probes_kj}")
    
    # Get probes data and store in csv file
    print("\nSaving probes data...")
    for i_probe in range(n_probes):
        k,j = probes_kj[i_probe,:]
        t_ = t_data[k,j,:]    # 1-D array, length num_points_x (= num_points_t)
        x_ = x_data[k,j,:]
        y_ = y_data[k,j,:]
        z_ = z_data[k,j,:]
        u_ = u_data[k,j,:]
        v_ = v_data[k,j,:]
        w_ = w_data[k,j,:]
        fname_ = os.path.join(output_csv_directory, f'probeline{i_probe}_k{k}_j{j}_{file_details}.csv')
        np.savetxt(fname_, 
                   X=np.array([t_, x_, y_, z_, u_, v_, w_]).T,
                   header='t[s],          x[m],            y[m],            z[m],            u[m/s],          v[m/s],           w[m/s]',
                   delimiter=',',
                   fmt='%.10e',
        )
        print(f"Probe {i_probe} in '{fname_}'")