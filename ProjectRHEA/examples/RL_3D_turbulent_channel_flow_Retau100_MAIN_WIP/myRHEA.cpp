#include "myRHEA.hpp"

#ifdef _OPENACC
#include <openacc.h>
#endif
#include <numeric>
#include <algorithm>
#include <cmath>        // std::pow
#include <math.h>       // M_PI constant

using namespace std;

////////// COMPILATION DIRECTIVES //////////
/// TODO: move these env parameters to an .h to be included by SmartRedisManager.cpp & .h files 
#define _FEEDBACK_LOOP_BODY_FORCE_ 1				/// Activate feedback loop for the body force moving the flow
#define _CORRECT_U_BULK_ 1                          /// Activate correction of u_bulk
#define _FIXED_TIME_STEP_ 1                         /// Activate fixed time step
#define _ACTIVE_CONTROL_BODY_FORCE_ 1               /// Activate active control for the body force
#define _RL_CONTROL_IS_SUPERVISED_ 1
#define _SPATIAL_SMOOTHING_RL_ACTION_ 1
#define _TEMPORAL_SMOOTHING_RL_ACTION_ 1
#define _WITNESS_XZ_SLICES_ 1
#define _INCLUDE_YCOORD_INTO_RL_STATE_ 1

/// Pi number
//const double pi = 2.0*asin( 1.0 );

/// PROBLEM PARAMETERS ///
//const double R_specific = 287.058;				/// Specific gas constant
const double gamma_0    = 1.4;					    /// Heat capacity ratio
//const double c_p        = gamma_0*R_specific/( gamma_0 - 1.0 );	/// Isobaric heat capacity
const double delta      = 1.0;					    /// Channel half-height
const double Re_tau     = 100.0;				    /// Friction Reynolds number
const double Ma         = 3.0e-1;				    /// Mach number
//const double Pr         = 0.71;					/// Prandtl number
const double rho_0      = 1.0;					    /// Reference density	
const double u_tau      = 1.0;					    /// Friction velocity
const double tau_w      = rho_0*u_tau*u_tau;		/// Wall shear stress
const double mu         = rho_0*u_tau*delta/Re_tau;	/// Dynamic viscosity	
const double nu         = u_tau*delta/Re_tau;		/// Kinematic viscosity	
//const double kappa      = c_p*mu/Pr;				/// Thermal conductivity	
const double Re_b       = pow( Re_tau/0.09, 1.0/0.88 );		/// Bulk (approximated) Reynolds number
const double u_b        = nu*Re_b/( 2.0*delta );		    /// Bulk (approximated) velocity
const double P_0        = rho_0*u_b*u_b/( gamma_0*Ma*Ma );	/// Reference pressure
//const double T_0        = P_0/( rho_0*R_specific );		/// Reference temperature
//const double L_x        = 4.0*pi*delta;			/// Streamwise length
//const double L_y        = 2.0*delta;				/// Wall-normal height
//const double L_z        = 4.0*pi*delta/3.0;		/// Spanwise width
const double kappa_vK   = 0.41;                     /// von Kármán constant
const double y_0        = nu/( 9.0*u_tau );         /// Smooth-wall roughness
const double u_0        = ( u_tau/kappa_vK )*( log( delta/y_0 ) + ( y_0/delta ) - 1.0 );    /// Volume average of a log-law velocity profile
const double alpha_u    = 0.5;                      /// Magnitude of velocity perturbations
const double alpha_P    = 0.1;                      /// Magnitude of pressure perturbations

const double fixed_time_step = 1.0e-4;              /// Time step value [s]
const int cout_precision = 10;		                /// Output precision (fixed) 

#if _CORRECT_U_BULK_
const double u_bulk_reference = 14.665;
#endif
#if _FEEDBACK_LOOP_BODY_FORCE_
/// Estimated uniform body force to drive the flow
double controller_output = tau_w/delta;			    /// Initialize controller output
double controller_error  = 0.0;			        	/// Initialize controller error
double controller_K_p    = 1.0e-1;		        	/// Controller proportional gain
#endif

#if _ACTIVE_CONTROL_BODY_FORCE_

#ifndef RL_CASE_PATH
    #error "RL_CASE_PATH is not defined! Please define it during compilation."
#endif
const char* rl_case_path = RL_CASE_PATH;  // Use compile-time constant value

int action_dim = 6;
/// eigen-values barycentric map coordinates - corners of realizable region
const double EPS     = numeric_limits<double>::epsilon();
/* Baricentric map coordinates, source: https://en.wikipedia.org/wiki/Barycentric_coordinate_system
 * Transform xmap -> lambda: lambda0,1 = Tinv * (xmap0,1 - t); lambda2 = 1 - lambda0 - lambda1
 * Transform lambda -> xmap: xmap0,1   = T * lambda0,1 + t   
 *                                     = lambda0 * x1c + lambda1 * x2c + lambda2 * x3c
 * Realizability Condition (lambda): 0<=lambda_i<=1, sum(lambda_i)=1
 * Realizability Condition (xmap):   xmap coord inside barycentric map triangle, defined by x1c, x2c, x3c */
const vector<double> x1c = {1.0, 0.0};                // corner x1c
const vector<double> x2c = {0.0, 0.0};                // corner x2c
const vector<double> x3c = {0.5, sqrt(3.0) / 2.0};    // corner x3c
const vector<double> t   = {x3c[0], x3c[1]};
const vector<vector<double>> T = {
    {x1c[0] - x3c[0], x2c[0] - x3c[0] },        // row 1, T[0][:]
    {x1c[1] - x3c[1], x2c[1] - x3c[1]},         // row 2, T[1][:]
};
const double Tdet = T[0][0] * T[1][1] - T[0][1] * T[1][0];
const vector<vector<double>> Tinv = {
    {  T[1][1] / Tdet, - T[0][1] / Tdet},       // row 1, Tinv[0][:]
    {- T[1][0] / Tdet,   T[0][0] / Tdet},       // row 2, Tinv[1][:]
};
/// Kronecker delta
const vector<vector<double>> Deltaij = {
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},
};

#if _RL_CONTROL_IS_SUPERVISED_
// Reference profiles (_ALL_ coordinates, not including boundaries)
double avg_u_reference_profile[] = {   /// only inner points
    1.56250681,  4.53208987,  7.1319341,   9.21610755,  10.80941713, 12.01061986,
    12.92394768, 13.63084145, 14.19101597, 14.64527649, 15.02198375, 15.34038925,
    15.61407404, 15.8526443,  16.06328415, 16.25131431, 16.42061089, 16.57378995,
    16.71267302, 16.83863397, 16.95284802, 17.05627761, 17.1496196,  17.23328929,
    17.3074833,  17.37228527, 17.42777158, 17.47402154, 17.51107657, 17.53892986,
    17.55755075, 17.56688639, 17.56688639, 17.55755075, 17.53892986, 17.51107657,
    17.47402154, 17.42777158, 17.37228527, 17.3074833,  17.23328929, 17.1496196,
    17.05627761, 16.95284802, 16.83863397, 16.71267302, 16.57378995, 16.42061089,
    16.25131431, 16.06328415, 15.8526443,  15.61407404, 15.34038925, 15.02198375,
    14.64527649, 14.19101597, 13.63084145, 12.92394768, 12.01061986, 10.80941713,
    9.21610755,  7.1319341,   4.53208987,  1.56250681,
};
double rmsf_u_reference_profile[] = {   /// only inner points
    0.51420372, 1.46461524, 2.12115121, 2.44822476, 2.55641146, 2.54775172,
    2.48329431, 2.39547355, 2.30102961, 2.20835085, 2.12122626, 2.04096227,
    1.9678679,  1.90159865, 1.84149707, 1.78657356, 1.73595965, 1.68921476,
    1.64614762, 1.60641594, 1.56960547, 1.53545274, 1.5038238,  1.47474421,
    1.4483624,  1.42476202, 1.40400397, 1.38624215, 1.37171104, 1.36065011,
    1.35321591, 1.34948453, 1.34948453, 1.35321591, 1.36065011, 1.37171104,
    1.38624215, 1.40400397, 1.42476202, 1.4483624,  1.47474421, 1.5038238,
    1.53545274, 1.56960547, 1.60641594, 1.64614762, 1.68921476, 1.73595965,
    1.78657356, 1.84149707, 1.90159865, 1.9678679,  2.04096227, 2.12122626,
    2.20835085, 2.30102961, 2.39547355, 2.48329431, 2.54775172, 2.55641146,
    2.44822476, 2.12115121, 1.46461524, 0.51420372,
};
double rmsf_v_reference_profile[] = {   /// only inner points
    0.01665036, 0.0819584,  0.16730256, 0.24795548, 0.32427953, 0.39101187,
    0.44971311, 0.49887803, 0.53992747, 0.57274725, 0.59858818, 0.61780168,
    0.63145149, 0.64002939, 0.64441306, 0.64516155, 0.6430155,  0.6384253,
    0.63195878, 0.62403478, 0.6151404,  0.60561876, 0.5958134,  0.58594603,
    0.57625823, 0.56695901, 0.55830965, 0.55056118, 0.54398771, 0.53882548,
    0.53527766, 0.53347188, 0.53347188, 0.53527766, 0.53882548, 0.54398771,
    0.55056118, 0.55830965, 0.56695901, 0.57625823, 0.58594603, 0.5958134,
    0.60561876, 0.6151404,  0.62403478, 0.63195878, 0.6384253,  0.6430155,
    0.64516155, 0.64441306, 0.64002939, 0.63145149, 0.61780168, 0.59858818,
    0.57274725, 0.53992747, 0.49887803, 0.44971311, 0.39101187, 0.32427953,
    0.24795548, 0.16730256, 0.0819584,  0.01665036,
};
double rmsf_w_reference_profile[] = {   /// only inner points
    0.20548671, 0.41458449, 0.54589934, 0.63491356, 0.69937532, 0.74364254,
    0.77419641, 0.79515881, 0.80928574, 0.81800294, 0.82219749, 0.8225368,
    0.81933529, 0.81306419, 0.80409932, 0.79300213, 0.77993032, 0.76505711,
    0.74868062, 0.73128271, 0.71331081, 0.69522915, 0.67742764, 0.66026295,
    0.6440551,  0.62919172, 0.61603816, 0.60484136, 0.59569842, 0.58866505,
    0.58383073, 0.58135072, 0.58135072, 0.58383073, 0.58866505, 0.59569842,
    0.60484136, 0.61603816, 0.62919172, 0.6440551,  0.66026295, 0.67742764,
    0.69522915, 0.71331081, 0.73128271, 0.74868062, 0.76505711, 0.77993032,
    0.79300213, 0.80409932, 0.81306419, 0.81933529, 0.8225368,  0.82219749,
    0.81800294, 0.80928574, 0.79515881, 0.77419641, 0.74364254, 0.69937532,
    0.63491356, 0.54589934, 0.41458449, 0.20548671,
};
#endif  /// of _RL_CONTROL_IS_SUPERVISED_
#endif  /// of _ACTIVE_CONTROL_BODY_FORCE_

////////// myRHEA CLASS //////////

myRHEA::myRHEA(const string name_configuration_file, const string tag, const string restart_data_file, const string t_action, const string t_episode, const string t_begin_control, const string db_clustered, const string global_step) : FlowSolverRHEA(name_configuration_file) {

#if _ACTIVE_CONTROL_BODY_FORCE_
    rl_f_rhou_field.setTopology(topo, "rl_f_rhou_field");
    rl_f_rhov_field.setTopology(topo, "rl_f_rhov_field");
    rl_f_rhow_field.setTopology(topo, "rl_f_rhow_field");
#if _SPATIAL_SMOOTHING_RL_ACTION_
    rl_f_rhou_field_aux.setTopology(topo, "rl_f_rhou_field_aux");
    rl_f_rhov_field_aux.setTopology(topo, "rl_f_rhov_field_aux");
    rl_f_rhow_field_aux.setTopology(topo, "rl_f_rhow_field_aux");
#endif  /// of _SPATIAL_SMOOTHING_RL_ACTION_
#if _TEMPORAL_SMOOTHING_RL_ACTION_
    rl_f_rhou_field_prev_step.setTopology(topo, "rl_f_rhou_field_prev_step");
    rl_f_rhov_field_prev_step.setTopology(topo, "rl_f_rhov_field_prev_step");
    rl_f_rhow_field_prev_step.setTopology(topo, "rl_f_rhow_field_prev_step");
    rl_f_rhou_field_curr_step.setTopology(topo, "rl_f_rhou_field_curr_step");
    rl_f_rhov_field_curr_step.setTopology(topo, "rl_f_rhov_field_curr_step");
    rl_f_rhow_field_curr_step.setTopology(topo, "rl_f_rhow_field_curr_step");
#endif  /// of _TEMPORAL_SMOOTHING_RL_ACTION_
    DeltaRxx_field.setTopology(topo, "DeltaRxx");
    DeltaRxy_field.setTopology(topo, "DeltaRxy");
    DeltaRxz_field.setTopology(topo, "DeltaRxz");
    DeltaRyy_field.setTopology(topo, "DeltaRyy");
    DeltaRyz_field.setTopology(topo, "DeltaRyz");
    DeltaRzz_field.setTopology(topo, "DeltaRzz");
    action_mask.setTopology(topo, "action_mask");
    timers->createTimer( "rl_smartredis_communications" );
    timers->createTimer( "rl_update_DeltaRij" );
    timers->createTimer( "rl_update_control_term" );
#if _RL_CONTROL_IS_SUPERVISED_
    avg_u_reference_field.setTopology(topo,  "avg_u_reference_field");
    rmsf_u_reference_field.setTopology(topo, "rmsf_u_reference_field");
    rmsf_v_reference_field.setTopology(topo, "rmsf_v_reference_field");
    rmsf_w_reference_field.setTopology(topo, "rmsf_w_reference_field");
#else
    avg_u_previous_field.setTopology(topo,  "avg_u_previous_field");
    rmsf_u_previous_field.setTopology(topo, "rmsf_u_previous_field");
    rmsf_v_previous_field.setTopology(topo, "rmsf_v_previous_field");
    rmsf_w_previous_field.setTopology(topo, "rmsf_w_previous_field");
    ///rmsf_v_previous_field.setTopology(topo, "rmsf_v_previous_field");
    ///rmsf_w_previous_field.setTopology(topo, "rmsf_w_previous_field");
#endif  /// of _RL_CONTROL_IS_SUPERVISED_
    initRLParams(tag, restart_data_file, t_action, t_episode, t_begin_control, db_clustered, global_step);
    initSmartRedis();
#endif  /// of _ACTIVE_CONTROL_BODY_FORCE_

};


void myRHEA::initRLParams(const string &tag, const string &restart_data_file, const string &t_action, const string &t_episode, const string &t_begin_control, const string &db_clustered, const string &global_step) {

    /// Logging
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
        cout << "\nInitializing RL parameters..." << endl;
        cout << "RL simulation params:" << endl;
        cout << "--tag: " << tag << endl;
        cout << "--restart_data_file: " << restart_data_file << endl;
        cout << "--t_action: " << t_action << endl;
        cout << "--t_episode: " << t_episode << endl;
        cout << "--t_begin_control: " << t_begin_control << endl; 
        cout << "--db_clustered: " << db_clustered << endl;
        cout << "--global_step: " << global_step << endl;
    }

    /// String arguments
    this->tag                = tag;
    this->restart_data_file  = restart_data_file;            // updated variable from previously defined value in FlowSolverRHEA::readConfigurationFile
    this->global_step        = global_step;
    /// Double arguments
    try {
        // Set actuation attributes, from string to double
        this->actuation_period        = std::stod(t_action);         
        this->begin_actuation_time    = std::stod(t_begin_control);
        this->previous_actuation_time = 0.0;
        this->final_time              = std::stod(t_episode);        // updated variable from previously defined value in FlowSolverRHEA::readConfigurationFile
        if (my_rank == 0) {
            cout << "[myRHEA::initRLParams] " 
                 << "actuation_period = " << scientific << this->actuation_period
                 << ", begin_actuation_time = " << scientific << this->begin_actuation_time
                 << ", previous_actuation_time = " << scientific << this->previous_actuation_time
                 << ", final_time = " << scientific << this->final_time << endl;
        }
    } catch (const invalid_argument& e) {
        cerr << "Invalid numeric argument: " << e.what() << endl;
        cerr << "for t_action = " << t_action << ", t_episode = " << t_episode << ", t_begin_control = " << t_begin_control << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    } catch (const out_of_range& e) {
        cerr << "Numeric argument out of range: " << e.what() << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /// Bool arguments
    if (db_clustered == "true" || db_clustered == "True" || db_clustered == "1") {
        this->db_clustered = true;
    } else if (db_clustered == "false" || db_clustered == "False" || db_clustered == "0") {
        this->db_clustered = false;
    } else {
        cerr << "Invalid boolean argument for db_clustered = " << db_clustered << endl; 
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /// Additional arguments, defined here /// TODO: include this in some configuration file
#if _WITNESS_XZ_SLICES_
    this->witness_file = string(rl_case_path) + "/config_control_witness/witnessXZSlices8.txt";
#else
    this->witness_file = string(rl_case_path) + "/config_control_witness/witnessPoints8.txt";
#endif
    this->control_cubes_file = string(rl_case_path) + "/config_control_witness/cubeControl8.txt";
    this->time_key      = "ensemble_" + tag + ".time";
    this->step_type_key = "ensemble_" + tag + ".step_type";
    this->state_key     = "ensemble_" + tag + ".state";
    this->action_key    = "ensemble_" + tag + ".action";
    this->reward_key    = "ensemble_" + tag + ".reward";

    /// Witness points
    readWitnessPoints(); 
    preproceWitnessPoints();        // updates attribute 'state_local_size2'

    /// Control cubic regions
    readControlCubes();
    getControlCubes();              // updates 'action_global_size2', 'n_rl_envs', 'action_mask'

    /// Allocate action data
    /// Annotation: State is stored in arrays of different sizes on each MPI rank.
    ///             Actions is a global array living in all processes.
    action_global.resize(action_global_size2);
    state_local.resize(state_local_size2);
    std::fill(action_global.begin(), action_global.end(), 0.0);
    std::fill(state_local.begin(), state_local.end(), 0.0);

    /// Auxiliary variable for reward calculation
#if _RL_CONTROL_IS_SUPERVISED_
    /// -------------- Build rmsf_u,v,w_reference_field --------------
    // Accessing the global domain decomposition data
    int global_startY, global_j;
    int globalNy    = topo->getMesh()->getGNy();    // Total number of y-cells in the global domain
    int localNy     = topo->getlNy()-2;             // Local number of y-cells for this process (-2 for inner points only)
    int divy        = globalNy % np_y;              // Remainder for non-uniform decomposition
    // Calculate the rank's position in the y-dimension based on rank and grid layout
    int plane_rank = my_rank % (np_x * np_y);
    int rank_in_y  = plane_rank / np_x;             // Rank's position in the y-dimension (row in the global grid)
    // Calculate the global start index for this rank's local slice in the y-dimension
    if (rank_in_y < divy) {
        global_startY = rank_in_y * (localNy + 1);  // Extra cell for some ranks
    } else {
        global_startY = rank_in_y * localNy + divy; // Regular distribution for remaining ranks
    }
    // Debugging
    cout << "Rank " << my_rank << ": npx=" << np_x << ", npy=" << np_y 
         << ", localNy=" << localNy << ", divy=" << divy 
         << ", globalNy=" << globalNy << ", rank_in_y=" << rank_in_y 
         << ", global_startY=" << global_startY << endl;
    // Fill global profile data into local field data
    for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
        global_j = global_startY + (j - topo->iter_common[_INNER_][_INIY_]);
        for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                avg_u_reference_field[I1D(i,j,k)]  = avg_u_reference_profile[global_j];
                rmsf_u_reference_field[I1D(i,j,k)] = rmsf_u_reference_profile[global_j];
                rmsf_v_reference_field[I1D(i,j,k)] = rmsf_v_reference_profile[global_j];
                rmsf_w_reference_field[I1D(i,j,k)] = rmsf_w_reference_profile[global_j];
            }
        }
        /// Debugging
        cout << "Rank " << my_rank << ": j=" << j << " (local), global_j=" << global_j 
             << ", Reference value of avg_u: " << avg_u_reference_profile[global_j]
             << ", rmsf_u: " << rmsf_u_reference_profile[global_j]
             << ", rmsf_v: " << rmsf_v_reference_profile[global_j]
             << ", rmsf_w: " << rmsf_w_reference_profile[global_j] 
             << endl;
    }
#endif

    /// Initialize additional attribute members
#if _RL_CONTROL_IS_SUPERVISED_
    first_actuation_time_done   = true;
    first_actuation_period_done = true;
#else
    first_actuation_time_done   = false;
    first_actuation_period_done = false;
#endif
};


/// Based on subroutine mod_smartredis::init_smartredis
void myRHEA::initSmartRedis() {
    /// TODO: transform this Fortran code of BLMARLFlowSolver_Incomp.f90 to current implementation and C++
    /*
    class(BLMARLFlowSolverIncomp), intent(inout) :: this
    open(unit=443,file="./output_"//trim(adjustl(this%tag))//"/"//"control_action.txt",status='replace')
    open(unit=445,file="./output_"//trim(adjustl(this%tag))//"/"//"control_reward.txt",status='replace')
    */  
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /// Create new smart redis manager
    manager = new SmartRedisManager(state_local_size2, action_dim, action_global_size2, n_rl_envs, tag, db_clustered);

    /// Write step type = 1
    manager->writeStepType(1, step_type_key);

};


void myRHEA::setInitialConditions() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    srand( my_rank );

    /// All (inner, halo, boundary): u, v, w, P and T
    double random_number, y_dist;
    for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                random_number       = 2.0*( (double) rand()/( RAND_MAX ) ) - 1.0;
                y_dist              = min( mesh->y[j], 2.0*delta - mesh->y[j] );
                u_field[I1D(i,j,k)] = ( 2.0*u_0*y_dist/delta ) + alpha_u*u_0*random_number;
                //v_field[I1D(i,j,k)] = 0.0;
                v_field[I1D(i,j,k)] = alpha_u*u_0*random_number;
                //w_field[I1D(i,j,k)] = 0.0;
                w_field[I1D(i,j,k)] = alpha_u*u_0*random_number;
                //P_field[I1D(i,j,k)] = P_0;
                P_field[I1D(i,j,k)] = P_0*( 1.0 + alpha_P*random_number );
                T_field[I1D(i,j,k)] = thermodynamics->calculateTemperatureFromPressureDensity( P_field[I1D(i,j,k)], rho_0 );
            }
        }
    }

    /// Update halo values
    u_field.update();
    v_field.update();
    w_field.update();
    P_field.update();
    T_field.update();

};

void myRHEA::calculateSourceTerms() {

    /// IMPORTANT: This method needs to be modified/overwritten according to the problem under consideration

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    cout << fixed << setprecision(cout_precision);

#if _CORRECT_U_BULK_
    double delta_x, delta_y, delta_z, delta_volume;
    double local_volume = 0.0;
    double local_u_volume = 0.0;
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Geometric stuff
                delta_x = 0.5*( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ); 
                delta_y = 0.5*( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ); 
                delta_z = 0.5*( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
                delta_volume =  delta_x * delta_y * delta_z;
                /// Update values
                local_volume += delta_volume;
                local_u_volume += u_field[I1D(i,j,k)] * delta_volume;
            }
        }
    }
    /// Communicate local values to obtain global & average values
    double global_volume = 0.0;
    double global_u_volume = 0.0;
    MPI_Allreduce(&local_volume, &global_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_u_volume, &global_u_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    /// Calculate u_bulk numeric
    double u_bulk_numeric = global_u_volume / global_volume;
    /// Correct flow flux
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                u_field[I1D(i,j,k)] += (u_bulk_reference - u_bulk_numeric);
            }
        }
    }
#endif /// _CORRECT_U_BULK_

#if _FEEDBACK_LOOP_BODY_FORCE_
    /// Evaluate numerical shear stress at walls

    /// Calculate local values
    double local_sum_u_boundary_w = 0.0;
    double local_sum_u_inner_w    = 0.0;
    double local_number_grid_points_w = 0.0;

    /// South boundary
    for(int i = topo->iter_bound[_SOUTH_][_INIX_]; i <= topo->iter_bound[_SOUTH_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_SOUTH_][_INIY_]; j <= topo->iter_bound[_SOUTH_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_SOUTH_][_INIZ_]; k <= topo->iter_bound[_SOUTH_][_ENDZ_]; k++) {
//		if( abs( avg_u_field[I1D(i,j,k)] ) > 0.0 ) {
//                    /// Sum boundary values
//                    local_sum_u_boundary_w += avg_u_field[I1D(i,j,k)];
//                    /// Sum inner values
//                    local_sum_u_inner_w    += avg_u_field[I1D(i,j+1,k)];
//		} else {
                    /// Sum boundary values
                    local_sum_u_boundary_w += u_field[I1D(i,j,k)];
                    /// Sum inner values
                    local_sum_u_inner_w    += u_field[I1D(i,j+1,k)];
//		}
                /// Sum number grid points
                local_number_grid_points_w += 1.0;
            }
        }
    }

    /// North boundary
    for(int i = topo->iter_bound[_NORTH_][_INIX_]; i <= topo->iter_bound[_NORTH_][_ENDX_]; i++) {
        for(int j = topo->iter_bound[_NORTH_][_INIY_]; j <= topo->iter_bound[_NORTH_][_ENDY_]; j++) {
            for(int k = topo->iter_bound[_NORTH_][_INIZ_]; k <= topo->iter_bound[_NORTH_][_ENDZ_]; k++) {
//		if( abs( avg_u_field[I1D(i,j,k)] ) > 0.0 ) {
//                    /// Sum boundary values
//                    local_sum_u_boundary_w += avg_u_field[I1D(i,j,k)];
//                    /// Sum inner values
//                    local_sum_u_inner_w    += avg_u_field[I1D(i,j-1,k)];
//		} else {
                    /// Sum boundary values
                    local_sum_u_boundary_w += u_field[I1D(i,j,k)];
                    /// Sum inner values
                    local_sum_u_inner_w    += u_field[I1D(i,j-1,k)];
//		}
                /// Sum number grid points
                local_number_grid_points_w += 1.0;
            }
        }
    }   

    /// Communicate local values to obtain global & average values
    double global_sum_u_boundary_w;
    MPI_Allreduce(&local_sum_u_boundary_w, &global_sum_u_boundary_w, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_sum_u_inner_w;
    MPI_Allreduce(&local_sum_u_inner_w, &global_sum_u_inner_w, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_number_grid_points_w;
    MPI_Allreduce(&local_number_grid_points_w, &global_number_grid_points_w, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double global_avg_u_boundary_w = global_sum_u_boundary_w/global_number_grid_points_w;
    double global_avg_u_inner_w    = global_sum_u_inner_w/global_number_grid_points_w;   

    /// Calculate delta_y
    double delta_y_wall = mesh->getGloby(1) - mesh->getGloby(0);

    /// Calculate tau_wall_numerical
    double tau_w_numerical = mu*( global_avg_u_inner_w - global_avg_u_boundary_w )/delta_y_wall;
    
    /// Update controller variables
    controller_error   = ( tau_w - tau_w_numerical )/delta;
    controller_output += controller_K_p*controller_error;

    //int my_rank, world_size;
    //MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    //MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //if( my_rank == 0 ) cout << tau_w << "  " << tau_w_numerical << "  " << controller_output << "  " << controller_error << endl;    
#endif

    /// Inner points: f_rhou, f_rhov, f_rhow and f_rhoE
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
#if _FEEDBACK_LOOP_BODY_FORCE_
		        f_rhou_field[I1D(i,j,k)] = controller_output;
#else
		        f_rhou_field[I1D(i,j,k)] = tau_w/delta;
#endif
                f_rhov_field[I1D(i,j,k)] = 0.0;
                f_rhow_field[I1D(i,j,k)] = 0.0;
                f_rhoE_field[I1D(i,j,k)] = 0.0;
            }
        }
    }

#if _ACTIVE_CONTROL_BODY_FORCE_

    if (rk_time_stage == 1) {    /// only recalculate f_rl_rhoi_field once per Runge-Kutta loop (1st rk iter.), if new action is needed

        /// If needed, get and post-process new action
        if (current_time > begin_actuation_time) {

            if ( !first_actuation_time_done ) {     /// executed just once
                calculateReward();                  /// initializing 'rmsf_u_field_local_previous', 'rmsf_v_field_local_previous', 'rmsf_w_field_local_previous' (unsupervised)
                updateState();
                first_actuation_time_done = true;
                if (my_rank == 0) {
                    cout << endl << endl << "[myRHEA::calculateSourceTerms] Initializing 'rmsf_u_field_local_previous', 'rmsf_v_field_local_previous', 'rmsf_w_field_local_previous'" << endl;
                }
            } 

            /// Check if new action is needed -> update F_pert(x,t_p) using updated RL actions a_k^p
            if (current_time - previous_actuation_time >= actuation_period) {

                /// Store previous actuation time
                previous_actuation_time = current_time;
                
#if _TEMPORAL_SMOOTHING_RL_ACTION_
                /// Store previous value of RL perturbation load term at previous RL step, F_pert(x,t_{p-1})  
                for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                    for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                        for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                            rl_f_rhou_field_prev_step[I1D(i,j,k)] = rl_f_rhou_field_curr_step[I1D(i,j,k)];
                            rl_f_rhov_field_prev_step[I1D(i,j,k)] = rl_f_rhov_field_curr_step[I1D(i,j,k)];
                            rl_f_rhow_field_prev_step[I1D(i,j,k)] = rl_f_rhow_field_curr_step[I1D(i,j,k)];
                        }
                    }
                }
#endif  /// of _TEMPORAL_SMOOTHING_RL_ACTION_

                if ( !first_actuation_period_done ) {

                    /// At the begining of the first actuation step, reward and state are updated to collect previous fields values, but no action is applied,
                    /// This is considered a previous step necessary before activating perturbations
                    calculateReward();      /// update avg_u,v,w_previous_field, rmsf_u,v,w_previous_field for next reward calculation
                    updateState();
                    first_actuation_period_done = true;
                    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
#if _TEMPORAL_SMOOTHING_RL_ACTION_
                                rl_f_rhou_field_curr_step[I1D(i,j,k)] = 0.0;
                                rl_f_rhov_field_curr_step[I1D(i,j,k)] = 0.0;
                                rl_f_rhow_field_curr_step[I1D(i,j,k)] = 0.0;
#else 
                                rl_f_rhou_field[I1D(i,j,k)] = 0.0;
                                rl_f_rhov_field[I1D(i,j,k)] = 0.0;
                                rl_f_rhow_field[I1D(i,j,k)] = 0.0;
#endif  /// of _TEMPORAL_SMOOTHING_RL_ACTION_
                            }
                        }
                    }
                    if (my_rank == 0) {
                        cout << endl << "[myRHEA::calculateSourceTerms] RL control is activated at current time (" << scientific << current_time << "), perturbation actions will be applied in the next RL step." << endl;
                    }

                } else {

                    /// -------------------------------------------------------------------
                    ///             COMMUNICATIONS BETWEEN RL <-> SIMULATION ENV. 
                    ///
                    /// RL -> simulation env. : action
                    /// Simulation env. -> RL : state, reward
                    /// -------------------------------------------------------------------

                    MPI_Barrier(MPI_COMM_WORLD);
                    timers->start( "rl_smartredis_communications" );

                    /// Logging
                    if (my_rank == 0) {
                        cout << endl << "[myRHEA::calculateSourceTerms] Performing SmartRedis communications (state, action, reward) at time instant " << current_time << endl;
                    }
                    
                    // SmartRedis communications 
                    // Writing state, reward, time
                    updateState();
                    manager->writeState(state_local, state_key);
                    calculateReward();                              /// update 'reward_local' attribute
                    manager->writeReward(reward_local, reward_key);
                    manager->writeTime(current_time, time_key);
                    // Reading new action...
                    manager->readAction(action_key);
                    action_global = manager->getActionGlobal();     /// action_global: vector<double> of size action_global_size2 = action_dim * n_rl_envs (currently only 1 action variable per rl env.)
                    /// action_local  = manager->getActionLocal();  /// action_local not used
                    /// Update & Write step size (from 1) to 0 if the next time that we require actuation value is the last one
                    if (current_time + 2.0 * actuation_period > final_time) {
                        manager->writeStepType(0, step_type_key);
                    }

                    MPI_Barrier(MPI_COMM_WORLD);
                    timers->stop( "rl_smartredis_communications" );

                    /// -------------------------------------------------------------------
                    ///             CALCULATE F_pert(x,t_p) FROM ACTION a_k^p
                    /// -------------------------------------------------------------------

                    MPI_Barrier(MPI_COMM_WORLD);
                    timers->start( "rl_update_DeltaRij" );

                    /// Initialize variables
                    double Rkk, phi1, phi2, phi3, xmap1, xmap2; 
                    double DeltaRkk = 0.0, DeltaPhi1 = 0.0, DeltaPhi2 = 0.0, DeltaPhi3 = 0.0, DeltaXmap1 = 0.0, DeltaXmap2 = 0.0; 
                    double Rkk_inv, Akk;
                    bool   isNegligibleAction, isNegligibleRkk;
                    size_t actuation_idx;
                    vector<vector<double>> Aij(3, vector<double>(3, 0.0));
                    vector<vector<double>> Dij(3, vector<double>(3, 0.0));
                    vector<vector<double>> Qij(3, vector<double>(3, 0.0));
                    vector<vector<double>> RijPert(3, vector<double>(3, 0.0));
                    
                    /// Calculate DeltaRij = Rij_perturbated - Rij_original
                    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {

                                /// If [i,j,k] is a control point (non-zero action_mask) -> introduce action
                                if (action_mask[I1D(i,j,k)] != 0.0) {

                                    /// Get perturbation values from RL agent
                                    actuation_idx = static_cast<size_t>(action_mask[I1D(i,j,k)]) - 1;   /// type size_t
                                    if (action_dim == 1) {
                                        DeltaRkk   = action_global[actuation_idx * action_dim + 0];
                                        DeltaPhi1  = 0.0;
                                        DeltaPhi2  = 0.0;
                                        DeltaPhi3  = 0.0;
                                        DeltaXmap1 = 0.0;
                                        DeltaXmap2 = 0.0;
                                    } else if (action_dim == 2) {
                                        DeltaRkk   = 0.0;
                                        DeltaPhi1  = 0.0;
                                        DeltaPhi2  = 0.0;
                                        DeltaPhi3  = 0.0;
                                        DeltaXmap1 = action_global[actuation_idx * action_dim + 0];
                                        DeltaXmap2 = action_global[actuation_idx * action_dim + 1];
                                    } else if (action_dim == 3) {
                                        DeltaRkk   = 0.0;
                                        DeltaPhi1  = action_global[actuation_idx * action_dim + 0];
                                        DeltaPhi2  = action_global[actuation_idx * action_dim + 1];
                                        DeltaPhi3  = action_global[actuation_idx * action_dim + 2];
                                        DeltaXmap1 = 0.0;
                                        DeltaXmap2 = 0.0;
                                    } else if (action_dim == 5) {
                                        DeltaRkk   = 0.0;
                                        DeltaPhi1  = action_global[actuation_idx * action_dim + 0];
                                        DeltaPhi2  = action_global[actuation_idx * action_dim + 1];
                                        DeltaPhi3  = action_global[actuation_idx * action_dim + 2];
                                        DeltaXmap1 = action_global[actuation_idx * action_dim + 3];
                                        DeltaXmap2 = action_global[actuation_idx * action_dim + 4];
                                    } else if (action_dim == 6) {
                                        DeltaRkk   = action_global[actuation_idx * action_dim + 0];
                                        DeltaPhi1  = action_global[actuation_idx * action_dim + 1];
                                        DeltaPhi2  = action_global[actuation_idx * action_dim + 2];
                                        DeltaPhi3  = action_global[actuation_idx * action_dim + 3];
                                        DeltaXmap1 = action_global[actuation_idx * action_dim + 4];
                                        DeltaXmap2 = action_global[actuation_idx * action_dim + 5];
                                    } else {
                                        cerr << "[myRHEA::calculateSourceTerms] _ACTIVE_CONTROL_BODY_FORCE_=1 new action calculation only implemented for action_dim == 1,2,3,5 and 6, but action_dim = " << action_dim << endl;
                                        MPI_Abort( MPI_COMM_WORLD, 1);
                                    }
                                    
                                    /// Calculate DeltaRij_field from DeltaRij d.o.f. (action), if action is not negligible 
                                    isNegligibleAction = (abs(DeltaRkk) < EPS && abs(DeltaPhi1) < EPS && abs(DeltaPhi2) < EPS && abs(DeltaPhi3) < EPS && abs(DeltaXmap1) < EPS && abs(DeltaXmap2) < EPS);
                                    Rkk                = favre_uffuff_field[I1D(i,j,k)] + favre_vffvff_field[I1D(i,j,k)] + favre_wffwff_field[I1D(i,j,k)];
                                    isNegligibleRkk    = (abs(Rkk) < EPS);
                                    if (isNegligibleAction || isNegligibleRkk) {
                                        DeltaRxx_field[I1D(i,j,k)] = 0.0;
                                        DeltaRyy_field[I1D(i,j,k)] = 0.0;
                                        DeltaRzz_field[I1D(i,j,k)] = 0.0;
                                        DeltaRxy_field[I1D(i,j,k)] = 0.0;
                                        DeltaRxz_field[I1D(i,j,k)] = 0.0;
                                        DeltaRyz_field[I1D(i,j,k)] = 0.0;
                                    } else {

                                        /// Rkk is Rij trace (dof #1)
                                        Rkk_inv = 1.0 / Rkk;

                                        /// Build anisotropy tensor (symmetric, trace-free)
                                        Aij[0][0]  = Rkk_inv * favre_uffuff_field[I1D(i,j,k)] - 1.0/3.0;
                                        Aij[1][1]  = Rkk_inv * favre_vffvff_field[I1D(i,j,k)] - 1.0/3.0;
                                        Aij[2][2]  = Rkk_inv * favre_wffwff_field[I1D(i,j,k)] - 1.0/3.0;
                                        Aij[0][1]  = Rkk_inv * favre_uffvff_field[I1D(i,j,k)];
                                        Aij[0][2]  = Rkk_inv * favre_uffwff_field[I1D(i,j,k)];
                                        Aij[1][2]  = Rkk_inv * favre_vffwff_field[I1D(i,j,k)];
                                        Aij[1][0]  = Aij[0][1];
                                        Aij[2][0]  = Aij[0][2];
                                        Aij[2][1]  = Aij[1][2];

                                        /// Ensure a_ij is trace-free (previous calc. introduces computational errors)
                                        Akk        = Aij[0][0] + Aij[1][1] + Aij[2][2];
                                        Aij[0][0] -= Akk / 3.0;
                                        Aij[1][1] -= Akk / 3.0;
                                        Aij[2][2] -= Akk / 3.0;

                                        /// Aij eigen-decomposition
                                        symmetricDiagonalize(Aij, Qij, Dij);                   // update Qij, Qij
                                        sortEigenDecomposition(Qij, Dij);                      // update Qij, Dij s.t. eigenvalues in decreasing order

                                        /// Eigen-vectors Euler ZXZ rotation angles (dof #2-4)
                                        eigVect2eulerAngles(Qij, phi1, phi2, phi3);      // update phi1, phi2, phi3

                                        /// Eigen-values Barycentric coordinates (dof #5-6)
                                        eigValMatrix2barycentricCoord(Dij, xmap1, xmap2);      // update xmap1, xmap2

                                        /// Build perturbed Rij d.o.f. -> x_new = x_old + Delta_x * x_old
                                        /// Delta_* are standarized values between 'action_bounds' RL parameter
                                        Rkk    = Rkk   * (1 + DeltaRkk);
                                        phi1   = phi1  * (1 + DeltaPhi1);
                                        phi2   = phi2  * (1 + DeltaPhi2);
                                        phi3   = phi3  * (1 + DeltaPhi3);
                                        xmap1  = xmap1 * (1 + DeltaXmap1);
                                        xmap2  = xmap2 * (1 + DeltaXmap2);

                                        /// Enforce realizability to perturbed Rij d.o.f
                                        enforceRealizability(Rkk, phi1, phi2, phi3, xmap1, xmap2);    // update Rkk, phi1, phi2, phi3, xmap1, xmap2, if necessary

                                        /// Calculate perturbed & realizable Rij
                                        eulerAngles2eigVect(phi1, phi2, phi3, Qij);                   // update Qij
                                        barycentricCoord2eigValMatrix(xmap1, xmap2, Dij);                   // update Dij
                                        sortEigenDecomposition(Qij, Dij);                                   // update Qij & Dij, if necessary
                                        Rijdof2matrix(Rkk, Dij, Qij, RijPert);                              // update RijPert

                                        /// Calculate perturbed & realizable DeltaRij
                                        DeltaRxx_field[I1D(i,j,k)] = RijPert[0][0] - favre_uffuff_field[I1D(i,j,k)];
                                        DeltaRyy_field[I1D(i,j,k)] = RijPert[1][1] - favre_vffvff_field[I1D(i,j,k)];
                                        DeltaRzz_field[I1D(i,j,k)] = RijPert[2][2] - favre_wffwff_field[I1D(i,j,k)];
                                        DeltaRxy_field[I1D(i,j,k)] = RijPert[0][1] - favre_uffvff_field[I1D(i,j,k)];
                                        DeltaRxz_field[I1D(i,j,k)] = RijPert[0][2] - favre_uffwff_field[I1D(i,j,k)];
                                        DeltaRyz_field[I1D(i,j,k)] = RijPert[1][2] - favre_vffwff_field[I1D(i,j,k)];

                                    }
                                }
                            }
                        }
                    }

                    MPI_Barrier(MPI_COMM_WORLD);
                    timers->stop( "rl_update_DeltaRij" );

                    MPI_Barrier(MPI_COMM_WORLD);
                    timers->start( "rl_update_control_term" );

                    /// Initialize variables
                    double d_DeltaRxx_x, d_DeltaRxy_x, d_DeltaRxz_x, d_DeltaRxy_y, d_DeltaRyy_y, d_DeltaRyz_y, d_DeltaRxz_z, d_DeltaRyz_z, d_DeltaRzz_z;
                    double delta_x, delta_y, delta_z;

                    /// Calculate and incorporate perturbation load F = \partial DeltaRij / \partial xj
                    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                                
                                /// Geometric stuff
                                delta_x = 0.5*( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ); 
                                delta_y = 0.5*( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ); 
                                delta_z = 0.5*( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
                                
                                /// Calculate DeltaRij derivatives
                                d_DeltaRxx_x = ( DeltaRxx_field[I1D(i+1,j,k)] - DeltaRxx_field[I1D(i-1,j,k)] ) / ( 2.0 * delta_x );
                                d_DeltaRxy_x = ( DeltaRxy_field[I1D(i+1,j,k)] - DeltaRxy_field[I1D(i-1,j,k)] ) / ( 2.0 * delta_x );
                                d_DeltaRxz_x = ( DeltaRxz_field[I1D(i+1,j,k)] - DeltaRxz_field[I1D(i-1,j,k)] ) / ( 2.0 * delta_x );
                                d_DeltaRxy_y = ( DeltaRxy_field[I1D(i,j+1,k)] - DeltaRxy_field[I1D(i,j-1,k)] ) / ( 2.0 * delta_y );
                                d_DeltaRyy_y = ( DeltaRyy_field[I1D(i,j+1,k)] - DeltaRyy_field[I1D(i,j-1,k)] ) / ( 2.0 * delta_y );
                                d_DeltaRyz_y = ( DeltaRyz_field[I1D(i,j+1,k)] - DeltaRyz_field[I1D(i,j-1,k)] ) / ( 2.0 * delta_y );
                                d_DeltaRxz_z = ( DeltaRxz_field[I1D(i,j,k+1)] - DeltaRxz_field[I1D(i,j,k-1)] ) / ( 2.0 * delta_z );
                                d_DeltaRyz_z = ( DeltaRyz_field[I1D(i,j,k+1)] - DeltaRyz_field[I1D(i,j,k-1)] ) / ( 2.0 * delta_z );
                                d_DeltaRzz_z = ( DeltaRzz_field[I1D(i,j,k+1)] - DeltaRzz_field[I1D(i,j,k-1)] ) / ( 2.0 * delta_z );

                                /// Apply perturbation load (\partial DeltaRij / \partial xj) into ui momentum equation
                                rl_f_rhou_field[I1D(i,j,k)] = ( -1.0 ) * rho_field[I1D(i,j,k)] * ( d_DeltaRxx_x + d_DeltaRxy_y + d_DeltaRxz_z );
                                rl_f_rhov_field[I1D(i,j,k)] = ( -1.0 ) * rho_field[I1D(i,j,k)] * ( d_DeltaRxy_x + d_DeltaRyy_y + d_DeltaRyz_z );
                                rl_f_rhow_field[I1D(i,j,k)] = ( -1.0 ) * rho_field[I1D(i,j,k)] * ( d_DeltaRxz_x + d_DeltaRyz_y + d_DeltaRzz_z );
#if _SPATIAL_SMOOTHING_RL_ACTION_
                                /// TODO: not necessary if averaging window size 3
                                rl_f_rhou_field_aux[I1D(i,j,k)] = rl_f_rhou_field[I1D(i,j,k)];
                                rl_f_rhov_field_aux[I1D(i,j,k)] = rl_f_rhov_field[I1D(i,j,k)];
                                rl_f_rhow_field_aux[I1D(i,j,k)] = rl_f_rhow_field[I1D(i,j,k)];
#endif  /// of _SPATIAL_SMOOTHING_RL_ACTION_
                            }
                        }
                    }

                    /// -------------------------------------------------------------------
                    ///                 SPATIAL SMOOTHING OF F_pert(x,t_p)
                    /// -------------------------------------------------------------------

#if _SPATIAL_SMOOTHING_RL_ACTION_
                    /// Calculate rl_f_rhou_field from rl_f_rhou_field_aux (idem. for v, w) + perform f_rl space averaging on bottom/top y boundaries of each mpi process 
                    /// Allocate memory for sending/receiving boundary data slices
                    int idx;
                    int i_start = topo->iter_common[_INNER_][_INIX_]; int i_end = topo->iter_common[_INNER_][_ENDX_];
                    int j_start = topo->iter_common[_INNER_][_INIY_]; int j_end = topo->iter_common[_INNER_][_ENDY_];
                    int k_start = topo->iter_common[_INNER_][_INIZ_]; int k_end = topo->iter_common[_INNER_][_ENDZ_];
                    int xz_slice_size = (i_end - i_start + 1) * (k_end - k_start + 1);
                    /// > 1st xz-slice from pseudo-environments / mpi processes (top & bottom) boundaries
                    vector<double> send_xz_slice_ymin_rl_f_rhou(xz_slice_size);  vector<double> send_xz_slice_ymin_rl_f_rhov(xz_slice_size);  vector<double> send_xz_slice_ymin_rl_f_rhow(xz_slice_size); 
                    vector<double> send_xz_slice_ymax_rl_f_rhou(xz_slice_size);  vector<double> send_xz_slice_ymax_rl_f_rhov(xz_slice_size);  vector<double> send_xz_slice_ymax_rl_f_rhow(xz_slice_size); 
                    vector<double> recv_xz_slice_ymin_rl_f_rhou(xz_slice_size);  vector<double> recv_xz_slice_ymin_rl_f_rhov(xz_slice_size);  vector<double> recv_xz_slice_ymin_rl_f_rhow(xz_slice_size); 
                    vector<double> recv_xz_slice_ymax_rl_f_rhou(xz_slice_size);  vector<double> recv_xz_slice_ymax_rl_f_rhov(xz_slice_size);  vector<double> recv_xz_slice_ymax_rl_f_rhow(xz_slice_size); 
                    /// > 2nd xz-slice from pseudo-environments / mpi processes (top & bottom) boundaries
                    vector<double> send_xz_slice_ymin2_rl_f_rhou(xz_slice_size); vector<double> send_xz_slice_ymin2_rl_f_rhov(xz_slice_size); vector<double> send_xz_slice_ymin2_rl_f_rhow(xz_slice_size); 
                    vector<double> send_xz_slice_ymax2_rl_f_rhou(xz_slice_size); vector<double> send_xz_slice_ymax2_rl_f_rhov(xz_slice_size); vector<double> send_xz_slice_ymax2_rl_f_rhow(xz_slice_size); 
                    vector<double> recv_xz_slice_ymin2_rl_f_rhou(xz_slice_size); vector<double> recv_xz_slice_ymin2_rl_f_rhov(xz_slice_size); vector<double> recv_xz_slice_ymin2_rl_f_rhow(xz_slice_size); 
                    vector<double> recv_xz_slice_ymax2_rl_f_rhou(xz_slice_size); vector<double> recv_xz_slice_ymax2_rl_f_rhov(xz_slice_size); vector<double> recv_xz_slice_ymax2_rl_f_rhow(xz_slice_size); 
                    /// > 2nd xz-slice from pseudo-environments / mpi processes (top & bottom) boundaries
                    vector<double> send_xz_slice_ymin3_rl_f_rhou(xz_slice_size); vector<double> send_xz_slice_ymin3_rl_f_rhov(xz_slice_size); vector<double> send_xz_slice_ymin3_rl_f_rhow(xz_slice_size); 
                    vector<double> send_xz_slice_ymax3_rl_f_rhou(xz_slice_size); vector<double> send_xz_slice_ymax3_rl_f_rhov(xz_slice_size); vector<double> send_xz_slice_ymax3_rl_f_rhow(xz_slice_size); 
                    vector<double> recv_xz_slice_ymin3_rl_f_rhou(xz_slice_size); vector<double> recv_xz_slice_ymin3_rl_f_rhov(xz_slice_size); vector<double> recv_xz_slice_ymin3_rl_f_rhow(xz_slice_size); 
                    vector<double> recv_xz_slice_ymax3_rl_f_rhou(xz_slice_size); vector<double> recv_xz_slice_ymax3_rl_f_rhov(xz_slice_size); vector<double> recv_xz_slice_ymax3_rl_f_rhow(xz_slice_size); 

                    /// Populate send_xz_slice_ymin_..., send_xz_slice_ymax_..., send_xz_slice_ymin2_... and send_xz_slice_ymax2_...  with boundary data
                    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                        for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                            idx = (i - i_start) * (k_end - k_start + 1) + (k - k_start);
                            /// > 1st xz-slice from pseudo-environments / mpi processes (top & bottom) boundaries
                            send_xz_slice_ymin_rl_f_rhou[idx]  = rl_f_rhou_field[I1D(i, j_start, k)]; 
                            send_xz_slice_ymin_rl_f_rhov[idx]  = rl_f_rhov_field[I1D(i, j_start, k)]; 
                            send_xz_slice_ymin_rl_f_rhow[idx]  = rl_f_rhow_field[I1D(i, j_start, k)];     
                            send_xz_slice_ymax_rl_f_rhou[idx]  = rl_f_rhou_field[I1D(i, j_end, k)];   
                            send_xz_slice_ymax_rl_f_rhov[idx]  = rl_f_rhov_field[I1D(i, j_end, k)];   
                            send_xz_slice_ymax_rl_f_rhow[idx]  = rl_f_rhow_field[I1D(i, j_end, k)]; 
                            /// > 2nd xz-slice from pseudo-environments / mpi processes (top & bottom) boundaries
                            send_xz_slice_ymin2_rl_f_rhou[idx] = rl_f_rhou_field[I1D(i, j_start+1, k)]; 
                            send_xz_slice_ymin2_rl_f_rhov[idx] = rl_f_rhov_field[I1D(i, j_start+1, k)]; 
                            send_xz_slice_ymin2_rl_f_rhow[idx] = rl_f_rhow_field[I1D(i, j_start+1, k)];     
                            send_xz_slice_ymax2_rl_f_rhou[idx] = rl_f_rhou_field[I1D(i, j_end-1, k)];   
                            send_xz_slice_ymax2_rl_f_rhov[idx] = rl_f_rhov_field[I1D(i, j_end-1, k)];   
                            send_xz_slice_ymax2_rl_f_rhow[idx] = rl_f_rhow_field[I1D(i, j_end-1, k)]; 
                            /// > 3rd xz-slice from pseudo-environments / mpi processes (top & bottom) boundaries
                            send_xz_slice_ymin3_rl_f_rhou[idx] = rl_f_rhou_field[I1D(i, j_start+2, k)]; 
                            send_xz_slice_ymin3_rl_f_rhov[idx] = rl_f_rhov_field[I1D(i, j_start+2, k)]; 
                            send_xz_slice_ymin3_rl_f_rhow[idx] = rl_f_rhow_field[I1D(i, j_start+2, k)];     
                            send_xz_slice_ymax3_rl_f_rhou[idx] = rl_f_rhou_field[I1D(i, j_end-2, k)];   
                            send_xz_slice_ymax3_rl_f_rhov[idx] = rl_f_rhov_field[I1D(i, j_end-2, k)];   
                            send_xz_slice_ymax3_rl_f_rhow[idx] = rl_f_rhow_field[I1D(i, j_end-2, k)]; 
                        }
                    }

                    /// Exchange boundary data with neighboring processes
                    /// Info: np_y == n_rl_envs == number of MPI processes (only distributed along y-coord)
                    /// Arrays to hold MPI request handles
                    MPI_Request requests[36];
                    int req_count = 0;
                    /// > Exchange data between current & previous pseudo-environment / mpi process in y-axis
                    if (my_rank > 0) {
                        /// > 1st xz-slice from pseudo-environments / mpi processes (top & bottom) boundaries
                        MPI_Isend(send_xz_slice_ymin_rl_f_rhou.data(),  xz_slice_size, MPI_DOUBLE, my_rank - 1, 0,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymin_rl_f_rhou.data(),  xz_slice_size, MPI_DOUBLE, my_rank - 1, 1,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Isend(send_xz_slice_ymin_rl_f_rhov.data(),  xz_slice_size, MPI_DOUBLE, my_rank - 1, 2,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymin_rl_f_rhov.data(),  xz_slice_size, MPI_DOUBLE, my_rank - 1, 3,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Isend(send_xz_slice_ymin_rl_f_rhow.data(),  xz_slice_size, MPI_DOUBLE, my_rank - 1, 4,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymin_rl_f_rhow.data(),  xz_slice_size, MPI_DOUBLE, my_rank - 1, 5,  MPI_COMM_WORLD, &requests[req_count++]);
                        /// > 2nd xz-slice from pseudo-environments / mpi processes (top & bottom) boundaries
                        MPI_Isend(send_xz_slice_ymin2_rl_f_rhou.data(), xz_slice_size, MPI_DOUBLE, my_rank - 1, 6,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymin2_rl_f_rhou.data(), xz_slice_size, MPI_DOUBLE, my_rank - 1, 7,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Isend(send_xz_slice_ymin2_rl_f_rhov.data(), xz_slice_size, MPI_DOUBLE, my_rank - 1, 8,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymin2_rl_f_rhov.data(), xz_slice_size, MPI_DOUBLE, my_rank - 1, 9,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Isend(send_xz_slice_ymin2_rl_f_rhow.data(), xz_slice_size, MPI_DOUBLE, my_rank - 1, 10, MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymin2_rl_f_rhow.data(), xz_slice_size, MPI_DOUBLE, my_rank - 1, 11, MPI_COMM_WORLD, &requests[req_count++]);
                        /// > 3rd xz-slice from pseudo-environments / mpi processes (top & bottom) boundaries
                        MPI_Isend(send_xz_slice_ymin3_rl_f_rhou.data(), xz_slice_size, MPI_DOUBLE, my_rank - 1, 12, MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymin3_rl_f_rhou.data(), xz_slice_size, MPI_DOUBLE, my_rank - 1, 13, MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Isend(send_xz_slice_ymin3_rl_f_rhov.data(), xz_slice_size, MPI_DOUBLE, my_rank - 1, 14, MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymin3_rl_f_rhov.data(), xz_slice_size, MPI_DOUBLE, my_rank - 1, 15, MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Isend(send_xz_slice_ymin3_rl_f_rhow.data(), xz_slice_size, MPI_DOUBLE, my_rank - 1, 16, MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymin3_rl_f_rhow.data(), xz_slice_size, MPI_DOUBLE, my_rank - 1, 17, MPI_COMM_WORLD, &requests[req_count++]);
                    }
                    /// > Exchange data between current & next pseudo-environment / mpi process in y-axis
                    if (my_rank < (n_rl_envs - 1)) {
                        /// >> 1st xz-slice from pseudo-environments / mpi processes (top & bottom) boundaries
                        MPI_Isend(send_xz_slice_ymax_rl_f_rhou.data(),  xz_slice_size, MPI_DOUBLE, my_rank + 1, 1,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymax_rl_f_rhou.data(),  xz_slice_size, MPI_DOUBLE, my_rank + 1, 0,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Isend(send_xz_slice_ymax_rl_f_rhov.data(),  xz_slice_size, MPI_DOUBLE, my_rank + 1, 3,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymax_rl_f_rhov.data(),  xz_slice_size, MPI_DOUBLE, my_rank + 1, 2,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Isend(send_xz_slice_ymax_rl_f_rhow.data(),  xz_slice_size, MPI_DOUBLE, my_rank + 1, 5,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymax_rl_f_rhow.data(),  xz_slice_size, MPI_DOUBLE, my_rank + 1, 4,  MPI_COMM_WORLD, &requests[req_count++]);
                        /// >> 2nd xz-slice from pseudo-environments / mpi processes (top & bottom) boundaries
                        MPI_Isend(send_xz_slice_ymax2_rl_f_rhou.data(), xz_slice_size, MPI_DOUBLE, my_rank + 1, 7,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymax2_rl_f_rhou.data(), xz_slice_size, MPI_DOUBLE, my_rank + 1, 6,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Isend(send_xz_slice_ymax2_rl_f_rhov.data(), xz_slice_size, MPI_DOUBLE, my_rank + 1, 9,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymax2_rl_f_rhov.data(), xz_slice_size, MPI_DOUBLE, my_rank + 1, 8,  MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Isend(send_xz_slice_ymax2_rl_f_rhow.data(), xz_slice_size, MPI_DOUBLE, my_rank + 1, 11, MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymax2_rl_f_rhow.data(), xz_slice_size, MPI_DOUBLE, my_rank + 1, 10, MPI_COMM_WORLD, &requests[req_count++]);
                        /// >> 3rd xz-slice from pseudo-environments / mpi processes (top & bottom) boundaries
                        MPI_Isend(send_xz_slice_ymax3_rl_f_rhou.data(), xz_slice_size, MPI_DOUBLE, my_rank + 1, 13, MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymax3_rl_f_rhou.data(), xz_slice_size, MPI_DOUBLE, my_rank + 1, 12, MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Isend(send_xz_slice_ymax3_rl_f_rhov.data(), xz_slice_size, MPI_DOUBLE, my_rank + 1, 15, MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymax3_rl_f_rhov.data(), xz_slice_size, MPI_DOUBLE, my_rank + 1, 14, MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Isend(send_xz_slice_ymax3_rl_f_rhow.data(), xz_slice_size, MPI_DOUBLE, my_rank + 1, 17, MPI_COMM_WORLD, &requests[req_count++]);
                        MPI_Irecv(recv_xz_slice_ymax3_rl_f_rhow.data(), xz_slice_size, MPI_DOUBLE, my_rank + 1, 16, MPI_COMM_WORLD, &requests[req_count++]);
                    }

                    /// Wait for all non-blocking communication to complete
                    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
                    cout << "[myRHEA::calculateSourceTerms] Rank " << my_rank << " performed all data exchange for space-averaging rl action along y-coord on pseudo-environments (mpi processes) boundaries" << endl; 

                    /// Apply action averaging in y-coord at the bottom/top y-coord boundaries of each mpi process
                    /// Averaging window size: 7 
                    /// > Averaging in pseudo-envioronments / mpi processes bottom boundary 
                    if (my_rank > 0) {
                        for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                                idx = (i - i_start) * (k_end - k_start + 1) + (k - k_start);
                                /// >> 1st xz_slice from bottom boundary
                                rl_f_rhou_field[I1D(i,j_start,k)]   = (1.0 / 7.0) * ( recv_xz_slice_ymin3_rl_f_rhou[idx] + recv_xz_slice_ymin2_rl_f_rhou[idx] + recv_xz_slice_ymin_rl_f_rhou[idx] + 4.0 * rl_f_rhou_field_aux[I1D(i,j_start,k)] );
                                rl_f_rhov_field[I1D(i,j_start,k)]   = (1.0 / 7.0) * ( recv_xz_slice_ymin3_rl_f_rhov[idx] + recv_xz_slice_ymin2_rl_f_rhov[idx] + recv_xz_slice_ymin_rl_f_rhov[idx] + 4.0 * rl_f_rhov_field_aux[I1D(i,j_start,k)] );
                                rl_f_rhow_field[I1D(i,j_start,k)]   = (1.0 / 7.0) * ( recv_xz_slice_ymin3_rl_f_rhow[idx] + recv_xz_slice_ymin2_rl_f_rhow[idx] + recv_xz_slice_ymin_rl_f_rhow[idx] + 4.0 * rl_f_rhow_field_aux[I1D(i,j_start,k)] );
                                /// >> 2nd xz_slice from bottom boundary
                                rl_f_rhou_field[I1D(i,j_start+1,k)] = (1.0 / 7.0) * ( recv_xz_slice_ymin2_rl_f_rhou[idx] + recv_xz_slice_ymin_rl_f_rhou[idx] + rl_f_rhou_field_aux[I1D(i,j_start,k)] + 4.0 * rl_f_rhou_field_aux[I1D(i,j_start+1,k)] );
                                rl_f_rhov_field[I1D(i,j_start+1,k)] = (1.0 / 7.0) * ( recv_xz_slice_ymin2_rl_f_rhov[idx] + recv_xz_slice_ymin_rl_f_rhov[idx] + rl_f_rhov_field_aux[I1D(i,j_start,k)] + 4.0 * rl_f_rhov_field_aux[I1D(i,j_start+1,k)] );
                                rl_f_rhow_field[I1D(i,j_start+1,k)] = (1.0 / 7.0) * ( recv_xz_slice_ymin2_rl_f_rhow[idx] + recv_xz_slice_ymin_rl_f_rhow[idx] + rl_f_rhow_field_aux[I1D(i,j_start,k)] + 4.0 * rl_f_rhow_field_aux[I1D(i,j_start+1,k)] );
                                /// >> 3rd xz_slice from bottom boundary
                                rl_f_rhou_field[I1D(i,j_start+2,k)] = (1.0 / 7.0) * ( recv_xz_slice_ymin_rl_f_rhou[idx] + rl_f_rhou_field_aux[I1D(i,j_start,k)] + rl_f_rhou_field_aux[I1D(i,j_start+1,k)] + 4.0 * rl_f_rhou_field_aux[I1D(i,j_start+2,k)] );
                                rl_f_rhov_field[I1D(i,j_start+2,k)] = (1.0 / 7.0) * ( recv_xz_slice_ymin_rl_f_rhov[idx] + rl_f_rhov_field_aux[I1D(i,j_start,k)] + rl_f_rhov_field_aux[I1D(i,j_start+1,k)] + 4.0 * rl_f_rhov_field_aux[I1D(i,j_start+2,k)] );
                                rl_f_rhow_field[I1D(i,j_start+2,k)] = (1.0 / 7.0) * ( recv_xz_slice_ymin_rl_f_rhow[idx] + rl_f_rhow_field_aux[I1D(i,j_start,k)] + rl_f_rhow_field_aux[I1D(i,j_start+1,k)] + 4.0 * rl_f_rhow_field_aux[I1D(i,j_start+2,k)] );
                            }
                        }
                    } 
                    /// > Averaging in pseudo-envioronments / mpi processes top boundary 
                    if (my_rank < (n_rl_envs - 1)) {
                        for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                                idx = (i - i_start) * (k_end - k_start + 1) + (k - k_start);
                                /// >> 1st xz_slice from top boundary
                                rl_f_rhou_field[I1D(i,j_end,k)]   = (1.0 / 7.0) * ( recv_xz_slice_ymax3_rl_f_rhou[idx] + recv_xz_slice_ymax2_rl_f_rhou[idx] + recv_xz_slice_ymax_rl_f_rhou[idx] + 4.0 * rl_f_rhou_field_aux[I1D(i,j_end,k)] );
                                rl_f_rhov_field[I1D(i,j_end,k)]   = (1.0 / 7.0) * ( recv_xz_slice_ymax3_rl_f_rhov[idx] + recv_xz_slice_ymax2_rl_f_rhov[idx] + recv_xz_slice_ymax_rl_f_rhov[idx] + 4.0 * rl_f_rhov_field_aux[I1D(i,j_end,k)] );
                                rl_f_rhow_field[I1D(i,j_end,k)]   = (1.0 / 7.0) * ( recv_xz_slice_ymax3_rl_f_rhow[idx] + recv_xz_slice_ymax2_rl_f_rhow[idx] + recv_xz_slice_ymax_rl_f_rhow[idx] + 4.0 * rl_f_rhow_field_aux[I1D(i,j_end,k)] );
                                /// >> 2nd xz_slice from top boundary
                                rl_f_rhou_field[I1D(i,j_end-1,k)] = (1.0 / 7.0) * ( recv_xz_slice_ymax2_rl_f_rhou[idx] + recv_xz_slice_ymax_rl_f_rhou[idx] + rl_f_rhou_field_aux[I1D(i,j_end,k)] + 4.0 * rl_f_rhou_field_aux[I1D(i,j_end-1,k)] );
                                rl_f_rhov_field[I1D(i,j_end-1,k)] = (1.0 / 7.0) * ( recv_xz_slice_ymax2_rl_f_rhov[idx] + recv_xz_slice_ymax_rl_f_rhov[idx] + rl_f_rhov_field_aux[I1D(i,j_end,k)] + 4.0 * rl_f_rhov_field_aux[I1D(i,j_end-1,k)] );
                                rl_f_rhow_field[I1D(i,j_end-1,k)] = (1.0 / 7.0) * ( recv_xz_slice_ymax2_rl_f_rhow[idx] + recv_xz_slice_ymax_rl_f_rhow[idx] + rl_f_rhow_field_aux[I1D(i,j_end,k)] + 4.0 * rl_f_rhow_field_aux[I1D(i,j_end-1,k)] );
                                /// >> 3rd xz_slice from top boundary
                                rl_f_rhou_field[I1D(i,j_end-2,k)] = (1.0 / 7.0) * ( recv_xz_slice_ymax_rl_f_rhou[idx] + rl_f_rhou_field_aux[I1D(i,j_end,k)] + rl_f_rhou_field_aux[I1D(i,j_end-1,k)] + 4.0 * rl_f_rhou_field_aux[I1D(i,j_end-2,k)] );
                                rl_f_rhov_field[I1D(i,j_end-2,k)] = (1.0 / 7.0) * ( recv_xz_slice_ymax_rl_f_rhov[idx] + rl_f_rhov_field_aux[I1D(i,j_end,k)] + rl_f_rhov_field_aux[I1D(i,j_end-1,k)] + 4.0 * rl_f_rhov_field_aux[I1D(i,j_end-2,k)] );
                                rl_f_rhow_field[I1D(i,j_end-2,k)] = (1.0 / 7.0) * ( recv_xz_slice_ymax_rl_f_rhow[idx] + rl_f_rhow_field_aux[I1D(i,j_end,k)] + rl_f_rhow_field_aux[I1D(i,j_end-1,k)] + 4.0 * rl_f_rhow_field_aux[I1D(i,j_end-2,k)] );
                            }
                        }
                    }
#endif /// of _SPATIAL_SMOOTHING_RL_ACTION_

#if _TEMPORAL_SMOOTHING_RL_ACTION_
                    /// -------------------------------------------------------------------
                    ///                 TEMPORAL SMOOTHING OF F_pert(x,t_p)
                    /// Not performing space-averaging here, just storing updated data
                    /// -------------------------------------------------------------------
                    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                                rl_f_rhou_field_curr_step[I1D(i,j,k)] = rl_f_rhou_field[I1D(i,j,k)];
                                rl_f_rhov_field_curr_step[I1D(i,j,k)] = rl_f_rhov_field[I1D(i,j,k)];
                                rl_f_rhow_field_curr_step[I1D(i,j,k)] = rl_f_rhow_field[I1D(i,j,k)];
                            }
                        }
                    }
#endif  /// of _TEMPORAL_SMOOTHING_RL_ACTION_

                    MPI_Barrier(MPI_COMM_WORLD);
                    timers->stop( "rl_update_control_term" );
            
                }   /// end else ( !first_actuation_period_done )
            }       /// end if (current_time - previous_actuation_time >= actuation_period), new action was required

        } else {    /// current_time <= begin_actuation_time

            for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                    for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
#if _TEMPORAL_SMOOTHING_RL_ACTION_
                        rl_f_rhou_field_curr_step[I1D(i,j,k)] = 0.0;
                        rl_f_rhov_field_curr_step[I1D(i,j,k)] = 0.0;
                        rl_f_rhow_field_curr_step[I1D(i,j,k)] = 0.0;
#else 
                        rl_f_rhou_field[I1D(i,j,k)] = 0.0;
                        rl_f_rhov_field[I1D(i,j,k)] = 0.0;
                        rl_f_rhow_field[I1D(i,j,k)] = 0.0;
#endif  /// of _TEMPORAL_SMOOTHING_RL_ACTION_
                    }
                }
            }
            if (my_rank == 0) {
                cout << "[myRHEA::calculateSourceTerms] RL Control is NOT applied yet, as current time (" << scientific << current_time << ") " << "< time begin control (" << scientific << begin_actuation_time << ")" << endl;
            }

        }           /// end else current_time <= begin_actuation_time
    }               /// end if (rk_time_stage == 1)
    
#endif  /// of _ACTIVE_CONTROL_BODY_FORCE_
    
    /// Update halo values
    f_rhou_field.update();
    f_rhov_field.update();
    f_rhow_field.update();
    f_rhoE_field.update();
#if _ACTIVE_CONTROL_BODY_FORCE_
    rl_f_rhou_field.update();
    rl_f_rhov_field.update();
    rl_f_rhow_field.update();
#if _TEMPORAL_SMOOTHING_RL_ACTION_
    rl_f_rhou_field_prev_step.update();
    rl_f_rhov_field_prev_step.update();
    rl_f_rhow_field_prev_step.update();
    rl_f_rhou_field_curr_step.update();
    rl_f_rhov_field_curr_step.update();
    rl_f_rhow_field_curr_step.update();
#endif  /// of _TEMPORAL_SMOOTHING_RL_ACTION_
#endif  /// of _ACTIVE_CONTROL_BODY_FORCE_

};


void myRHEA::temporalHookFunction() {

    if ( ( print_timers ) && (current_time_iter%print_frequency_iter == 0) ) {
        /// Save data only for ensemble #0 due to memory limitations
        if (tag == "0") {
            /// Print timers information
            char filename_timers[1024];
            sprintf( filename_timers, "%s/rhea_exp/timers_info/timers_information_file_%d_ensemble%s_step%s.txt", 
                     rl_case_path, current_time_iter, tag.c_str(), global_step.c_str() );
            timers->printTimers( filename_timers );
            /// Output current state in RL dedicated directory
            char data_path[1024];
            sprintf( data_path, "%s/rhea_exp/output_data", rl_case_path );
            this->outputCurrentStateDataRL(data_path);
        }
    }

};


void myRHEA::calculateTimeStep() {

#if _FIXED_TIME_STEP_
    /// Set new time step
    delta_t = fixed_time_step;
#else
    FlowSolverRHEA::calculateTimeStep();
#endif

};


void myRHEA::outputCurrentStateDataRL( string path ) {

    /// Write to file current solver state, time, time iteration and averaging time
    writer_reader->setAttribute( "Time", current_time );
    writer_reader->setAttribute( "Iteration", current_time_iter );
    writer_reader->setAttribute( "AveragingTime", averaging_time );
    writer_reader->writeRL( current_time_iter, tag, global_step, path );

};


void myRHEA::timeAdvanceConservedVariables() {

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /// TODO: check, remove lines below
    /// Debugging variables
    double rhou_inv_flux_ratio   = 0.0;
    double rhou_vis_flux_ratio   = 0.0;
    double f_rhou_field_ratio    = 0.0;
    double rl_f_rhou_field_ratio = 0.0;
    int ratio_counter            = 0;
#if _TEMPORAL_SMOOTHING_RL_ACTION_
    double actuation_period_fraction = ( current_time - previous_actuation_time ) / actuation_period;;
    double f1 = exp(-1.0 / actuation_period_fraction);
    double f2 = exp(-1.0 / (1.0 - actuation_period_fraction));
    double f3 = f1 / (f1 + f2);
#endif

    /// Coefficients of explicit Runge-Kutta stages
    double rk_a = 0.0, rk_b = 0.0, rk_c = 0.0;
    runge_kutta_method->setStageCoefficients(rk_a,rk_b,rk_c,rk_time_stage);    

    /// Inner points: rho, rhou, rhov, rhow and rhoE
    double f_rhouvw = 0.0;
    double rho_rhs_flux = 0.0, rhou_rhs_flux = 0.0, rhov_rhs_flux = 0.0, rhow_rhs_flux = 0.0, rhoE_rhs_flux = 0.0;

#if _TEMPORAL_SMOOTHING_RL_ACTION_
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
            /// ------------------------------------------------------------------- 
            ///                 TEMPORAL SMOOTHING OF F_pert(x,t_p)
            /// -------------------------------------------------------------------
            /// Time smoothing of RL perturbation load using pert. loads calculated at (*prev_step) and current (*curr_step) RL steps 
            /// Updates 'rl_f_rho{u,v,w}_field' to ensure smooth transition from rl_f_rho{u,v,w}_field_prev_step 
            /// to rl_f_rho{u,v,w}_field through actuation_period
            rl_f_rhou_field[I1D(i,j,k)] = rl_f_rhou_field_prev_step[I1D(i,j,k)] + f3 * ( rl_f_rhou_field_curr_step[I1D(i,j,k)] - rl_f_rhou_field_prev_step[I1D(i,j,k)] ); 
            rl_f_rhov_field[I1D(i,j,k)] = rl_f_rhov_field_prev_step[I1D(i,j,k)] + f3 * ( rl_f_rhov_field_curr_step[I1D(i,j,k)] - rl_f_rhov_field_prev_step[I1D(i,j,k)] ); 
            rl_f_rhow_field[I1D(i,j,k)] = rl_f_rhow_field_prev_step[I1D(i,j,k)] + f3 * ( rl_f_rhow_field_curr_step[I1D(i,j,k)] - rl_f_rhow_field_prev_step[I1D(i,j,k)] ); 
            }
        }
    }
#endif  /// of _TEMPORAL_SMOOTHING_RL_ACTION_
    
    /// ------------------------------------------------------------------- 
    ///                 PERTURBATION LOAD INTEGRATION 
    /// -------------------------------------------------------------------
    double delta_x, delta_y, delta_z, delta_volume;
    double local_volume = 0.0;
    double local_rl_f_rhou_volume = 0.0;
    double local_rl_f_rhov_volume = 0.0;
    double local_rl_f_rhow_volume = 0.0;
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Geometric stuff
                delta_x = 0.5*( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ); 
                delta_y = 0.5*( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ); 
                delta_z = 0.5*( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
                delta_volume =  delta_x * delta_y * delta_z;
                // Integrate values
                local_rl_f_rhou_volume += rl_f_rhou_field[I1D(i,j,k)] * delta_volume; 
                local_rl_f_rhov_volume += rl_f_rhov_field[I1D(i,j,k)] * delta_volume;
                local_rl_f_rhow_volume += rl_f_rhow_field[I1D(i,j,k)] * delta_volume;
                local_volume += delta_volume;
            }
        }
    }
    /// Communicate local values to obtain global & average values
    double global_volume = 0.0;
    double global_rl_f_rhou_volume = 0.0;
    double global_rl_f_rhov_volume = 0.0;
    double global_rl_f_rhow_volume = 0.0;
    MPI_Allreduce(&local_volume, &global_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_rl_f_rhou_volume, &global_rl_f_rhou_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_rl_f_rhov_volume, &global_rl_f_rhov_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_rl_f_rhow_volume, &global_rl_f_rhow_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    

    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {

                /// Enforce net flux zero of the RL perturbation load
                rl_f_rhou_field[I1D(i,j,k)] -= global_rl_f_rhou_volume / global_volume; 
                rl_f_rhov_field[I1D(i,j,k)] -= global_rl_f_rhov_volume / global_volume;
                rl_f_rhow_field[I1D(i,j,k)] -= global_rl_f_rhow_volume / global_volume;

                /// Work of momentum sources
                f_rhouvw = (f_rhou_field[I1D(i,j,k)] + rl_f_rhou_field[I1D(i,j,k)]) * u_field[I1D(i,j,k)]
                         + (f_rhov_field[I1D(i,j,k)] + rl_f_rhov_field[I1D(i,j,k)]) * v_field[I1D(i,j,k)]
                         + (f_rhow_field[I1D(i,j,k)] + rl_f_rhow_field[I1D(i,j,k)]) * w_field[I1D(i,j,k)];
                
                /// Sum right-hand-side (RHS) fluxes
                rho_rhs_flux  = ( -1.0 ) * rho_inv_flux[I1D(i,j,k)]; 
                rhou_rhs_flux = ( -1.0 ) * rhou_inv_flux[I1D(i,j,k)] + rhou_vis_flux[I1D(i,j,k)] + f_rhou_field[I1D(i,j,k)] + rl_f_rhou_field[I1D(i,j,k)];
                rhov_rhs_flux = ( -1.0 ) * rhov_inv_flux[I1D(i,j,k)] + rhov_vis_flux[I1D(i,j,k)] + f_rhov_field[I1D(i,j,k)] + rl_f_rhov_field[I1D(i,j,k)]; 
                rhow_rhs_flux = ( -1.0 ) * rhow_inv_flux[I1D(i,j,k)] + rhow_vis_flux[I1D(i,j,k)] + f_rhow_field[I1D(i,j,k)] + rl_f_rhow_field[I1D(i,j,k)]; 
                rhoE_rhs_flux = ( -1.0 ) * rhoE_inv_flux[I1D(i,j,k)] + rhoE_vis_flux[I1D(i,j,k)] + f_rhoE_field[I1D(i,j,k)] + f_rhouvw;
                
                /// Runge-Kutta step
                rho_field[I1D(i,j,k)]  = rk_a*rho_0_field[I1D(i,j,k)]  + rk_b*rho_field[I1D(i,j,k)]  + rk_c*delta_t*rho_rhs_flux;
                rhou_field[I1D(i,j,k)] = rk_a*rhou_0_field[I1D(i,j,k)] + rk_b*rhou_field[I1D(i,j,k)] + rk_c*delta_t*rhou_rhs_flux;
                rhov_field[I1D(i,j,k)] = rk_a*rhov_0_field[I1D(i,j,k)] + rk_b*rhov_field[I1D(i,j,k)] + rk_c*delta_t*rhov_rhs_flux;
                rhow_field[I1D(i,j,k)] = rk_a*rhow_0_field[I1D(i,j,k)] + rk_b*rhow_field[I1D(i,j,k)] + rk_c*delta_t*rhow_rhs_flux;
                rhoE_field[I1D(i,j,k)] = rk_a*rhoE_0_field[I1D(i,j,k)] + rk_b*rhoE_field[I1D(i,j,k)] + rk_c*delta_t*rhoE_rhs_flux;

                /// Debugging information
                if (action_mask[I1D(i,j,k)] != 0.0) { // Actuation point    
                    rhou_inv_flux_ratio          += std::abs( rhou_inv_flux[I1D(i,j,k)]   / ( rhou_rhs_flux + EPS ) );
                    rhou_vis_flux_ratio          += std::abs( rhou_vis_flux[I1D(i,j,k)]   / ( rhou_rhs_flux + EPS ) );
                    f_rhou_field_ratio           += std::abs( f_rhou_field[I1D(i,j,k)]    / ( rhou_rhs_flux + EPS ) );
                    rl_f_rhou_field_ratio        += std::abs( rl_f_rhou_field[I1D(i,j,k)] / ( rhou_rhs_flux + EPS) );
                    ratio_counter += 1;
                }
	        }
        }
    }
    rhou_inv_flux_ratio          /= ratio_counter;
    rhou_vis_flux_ratio          /= ratio_counter;
    f_rhou_field_ratio           /= ratio_counter;
    rl_f_rhou_field_ratio        /= ratio_counter;
    
    /// Summarized output
    if ( my_rank < 4) {
        cout << endl << "Rank " << my_rank << " u-RHS Ratios: " << rhou_inv_flux_ratio << ", " << rhou_vis_flux_ratio << ", " << f_rhou_field_ratio << ", " << rl_f_rhou_field_ratio << endl;
    }

#if _OPENACC_MANUAL_DATA_MOVEMENT_
    #pragma acc exit data copyout (rho_field.vector[0:local_size],rhou_field.vector[0:local_size],rhov_field.vector[0:local_size],rhow_field.vector[0:local_size],rhoE_field.vector[0:local_size])
#endif    

    /// Update halo values
    rho_field.update();
    rhou_field.update();
    rhov_field.update();
    rhow_field.update();
    rhoE_field.update();

    if( transport_pressure_scheme ) {
        this->timeAdvancePressure();
    }

};


void myRHEA::outputTemporalPointProbesData() {

    if (tag == "0") {
        FlowSolverRHEA::outputTemporalPointProbesData();
    }

};


///////////////////////////////////////////////////////////////////////////////
/** Symmetric diagonalization of a 3D matrix
 * 
 * The diagonal matrix D = QT * A * Q;  and  A = Q*D*QT
 * obtained from: 
 *       http://stackoverflow.com/questions/4372224/
 *       fast-method-for-computing-3x3-symmetric-matrix-spectral-decomposition
 * 
 * @param A     \input matrix to diagonalize, must be a symmetric matrix
 * @param Q     \output matrix of eigenvectors
 * @param D     \output diagonal matrix of eigenvalues
 */
void myRHEA::symmetricDiagonalize(
    const vector<vector<double>> &A, vector<vector<double>> &Q, vector<vector<double>> &D) {

    const int maxsteps=24;
    int k0, k1, k2;
    double o[3], m[3];
    double q [4] = {0.0,0.0,0.0,1.0};
    double jr[4];
    double sqw, sqx, sqy, sqz;
    double tmp1, tmp2, mq;
    double AQ[3][3];
    double thet, sgn, t, c;
    
    for(int i=0;i < maxsteps;++i) {
        // quat to matrix
        sqx      = q[0]*q[0];
        sqy      = q[1]*q[1];
        sqz      = q[2]*q[2];
        sqw      = q[3]*q[3];
        Q[0][0]  = ( sqx - sqy - sqz + sqw);
        Q[1][1]  = (-sqx + sqy - sqz + sqw);
        Q[2][2]  = (-sqx - sqy + sqz + sqw);
        tmp1     = q[0]*q[1];
        tmp2     = q[2]*q[3];
        Q[1][0]  = 2.0 * (tmp1 + tmp2);
        Q[0][1]  = 2.0 * (tmp1 - tmp2);
        tmp1     = q[0]*q[2];
        tmp2     = q[1]*q[3];
        Q[2][0]  = 2.0 * (tmp1 - tmp2);
        Q[0][2]  = 2.0 * (tmp1 + tmp2);
        tmp1     = q[1]*q[2];
        tmp2     = q[0]*q[3];
        Q[2][1]  = 2.0 * (tmp1 + tmp2);
        Q[1][2]  = 2.0 * (tmp1 - tmp2);

        // AQ = A * Q
        AQ[0][0] = Q[0][0]*A[0][0]+Q[1][0]*A[0][1]+Q[2][0]*A[0][2];
        AQ[0][1] = Q[0][1]*A[0][0]+Q[1][1]*A[0][1]+Q[2][1]*A[0][2];
        AQ[0][2] = Q[0][2]*A[0][0]+Q[1][2]*A[0][1]+Q[2][2]*A[0][2];
        AQ[1][0] = Q[0][0]*A[0][1]+Q[1][0]*A[1][1]+Q[2][0]*A[1][2];
        AQ[1][1] = Q[0][1]*A[0][1]+Q[1][1]*A[1][1]+Q[2][1]*A[1][2];
        AQ[1][2] = Q[0][2]*A[0][1]+Q[1][2]*A[1][1]+Q[2][2]*A[1][2];
        AQ[2][0] = Q[0][0]*A[0][2]+Q[1][0]*A[1][2]+Q[2][0]*A[2][2];
        AQ[2][1] = Q[0][1]*A[0][2]+Q[1][1]*A[1][2]+Q[2][1]*A[2][2];
        AQ[2][2] = Q[0][2]*A[0][2]+Q[1][2]*A[1][2]+Q[2][2]*A[2][2];

        // D = Qt * AQ
        D[0][0] = AQ[0][0]*Q[0][0]+AQ[1][0]*Q[1][0]+AQ[2][0]*Q[2][0];
        D[0][1] = AQ[0][0]*Q[0][1]+AQ[1][0]*Q[1][1]+AQ[2][0]*Q[2][1];
        D[0][2] = AQ[0][0]*Q[0][2]+AQ[1][0]*Q[1][2]+AQ[2][0]*Q[2][2];
        D[1][0] = AQ[0][1]*Q[0][0]+AQ[1][1]*Q[1][0]+AQ[2][1]*Q[2][0];
        D[1][1] = AQ[0][1]*Q[0][1]+AQ[1][1]*Q[1][1]+AQ[2][1]*Q[2][1];
        D[1][2] = AQ[0][1]*Q[0][2]+AQ[1][1]*Q[1][2]+AQ[2][1]*Q[2][2];
        D[2][0] = AQ[0][2]*Q[0][0]+AQ[1][2]*Q[1][0]+AQ[2][2]*Q[2][0];
        D[2][1] = AQ[0][2]*Q[0][1]+AQ[1][2]*Q[1][1]+AQ[2][2]*Q[2][1];
        D[2][2] = AQ[0][2]*Q[0][2]+AQ[1][2]*Q[1][2]+AQ[2][2]*Q[2][2];
        o[0]    = D[1][2];
        o[1]    = D[0][2];
        o[2]    = D[0][1];
        m[0]    = std::abs(o[0]);
        m[1]    = std::abs(o[1]);
        m[2]    = std::abs(o[2]);

        k0      = (m[0] > m[1] && m[0] > m[2])?0: (m[1] > m[2])? 1 : 2; // index of largest element of offdiag
        k1      = (k0+1)%3;
        k2      = (k0+2)%3;
        if (o[k0]==0.0) {
            break;  // diagonal already
        }

        thet    = (D[k2][k2]-D[k1][k1])/(2.0*o[k0]);
        sgn     = (thet > 0.0)?1.0:-1.0;
        thet   *= sgn; // make it positive
        t       = sgn /(thet +((thet < 1.E6)? std::sqrt(thet*thet+1.0):thet)) ; // sign(T)/(|T|+sqrt(T^2+1))
        c       = 1.0/std::sqrt(t*t+1.0); //  c= 1/(t^2+1) , t=s/c 
        if(c==1.0) {
            break;  // no room for improvement - reached machine precision.
        }

        jr[0 ]  = jr[1] = jr[2] = jr[3] = 0.0;
        jr[k0]  = sgn*std::sqrt((1.0-c)/2.0);  // using 1/2 angle identity sin(a/2) = std::sqrt((1-cos(a))/2)  
        jr[k0] *= -1.0; // since our quat-to-matrix convention was for v*M instead of M*v
        jr[3 ]  = std::sqrt(1.0f - jr[k0] * jr[k0]);
        if(jr[3]==1.0) {
            break; // reached limits of floating point precision
        }

        q[0]    = (q[3]*jr[0] + q[0]*jr[3] + q[1]*jr[2] - q[2]*jr[1]);
        q[1]    = (q[3]*jr[1] - q[0]*jr[2] + q[1]*jr[3] + q[2]*jr[0]);
        q[2]    = (q[3]*jr[2] + q[0]*jr[1] - q[1]*jr[0] + q[2]*jr[3]);
        q[3]    = (q[3]*jr[3] - q[0]*jr[0] - q[1]*jr[1] - q[2]*jr[2]);
        mq      = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        q[0]   /= mq;
        q[1]   /= mq;
        q[2]   /= mq;
        q[3]   /= mq;
    }

}

///////////////////////////////////////////////////////////////////////////////
/** Reconstruct matrix from eigen decomposition of a 3D matrix
 * 
 * The diagonal matrix D = QT * A * Q;  and  A = Q*D*QT
 * 
 * @param D     \input diagonal matrix of eigenvalues
 * @param Q     \input matrix of eigenvectors
 * @param A     \output reconstructed symmetric matrix
 */
void myRHEA::eigenDecomposition2Matrix(
  const vector<vector<double>> &D, const vector<vector<double>> &Q, vector<vector<double>> &A)
{

    // A = Q*D*QT
    vector<vector<double>> QT(3, vector<double>(3, 0.0));
    vector<vector<double>> B(3, vector<double>(3, 0.0));

    // compute QT
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            QT[j][i] = Q[i][j];
        }
    }

    //mat-vec, B = Q*D
    matrixMultiplicate(Q,D,B);

    // mat-vec, A = (Q*D)*QT = B*QT
    matrixMultiplicate(B,QT,A);

}

///////////////////////////////////////////////////////////////////////////////
/**
 * Sort eigenvalues and eigenvectors 
 * 
 * Sort eigenvalues in decreasing order (and eigenvectors equivalently), with
 * Qij[0][0] >= Qij[1][1] >= Qij[2][2]
 * 
 * @param Q   \inputoutput matrix of eigenvectors, with Qij[0][:] the first eigenvector
 * @param D   \inputoutput diagonal matrix of eigenvalues
 * 
 */
void myRHEA::sortEigenDecomposition(vector<vector<double>> &Q, vector<vector<double>> &D){

    // create pairs of indices and eigenvalues for sorting
    vector<pair<double,size_t>> sortedEigVal(3);
    for (size_t i = 0; i < 3; ++i) {
        sortedEigVal[i] = {D[i][i], i}; // {value, idx} pair
    }

    // sort eigenvalues in descending order
    std::sort(sortedEigVal.begin(), sortedEigVal.end(), [](const std::pair<double, long unsigned int>& a, const std::pair<double, long unsigned int>& b) {
        return a.first > b.first;
    });

    // Rearrange eigenvalues and eigenvectors based on the sorted indices
    vector<vector<double>> tempQ(3, vector<double>(3, 0.0));
    vector<vector<double>> tempD(3, vector<double>(3, 0.0));
    for (size_t i = 0; i < 3; ++i) {
        tempD[i][i] = D[sortedEigVal[i].second][sortedEigVal[i].second];
        for (size_t j = 0; j < 3; j++){
            tempQ[j][i] = Q[j][sortedEigVal[i].second];
        }
    }

    // update matrices
    Q = tempQ;
    D = tempD;

}

///////////////////////////////////////////////////////////////////////////////
/** Matrix-Matrix Multiplication 3D
 * 
 * Matrix multiplication C = A*B
 * 
 * @param A     \input matrix
 * @param B     \input matrix
 * @param C     \output matrix result of matrix-matrix multiplication 
 */
void myRHEA::matrixMultiplicate(
  const vector<vector<double>> &A, const vector<vector<double>> &B, vector<vector<double>> &C)
{

    // Perform multiplication C = A*B
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0;
            for (int k = 0; k < 3; ++k) {
                sum = sum + A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

}

///////////////////////////////////////////////////////////////////////////////
/// Truncate and normalize eigen-values if not satisfied ( 0 <= eigVal_i <= 1)
void myRHEA::truncateAndNormalizeEigVal(vector<double> &lambda){
    
    // 1st: truncate coordinates within range [0,1]
    for (double &coord : lambda) {
        coord = min(max(coord, 0.0), 1.0); 
    }
    
    // 2nd: normalize coordinates vector to sum(coordinates) = 1
    double sumCoord = std::accumulate(lambda.begin(), lambda.end(), 0.0); // calculate eigen-values sum
    if (sumCoord != 0.0) { // avoid division by zero
        for (double &coord : lambda) {
            coord /= sumCoord;
        }
    } else {               // if sumCoord is zero, handle setting all to 1/3
        for (double &coord : lambda) {
            coord = 1.0 / lambda.size();
        }
    }

}

///////////////////////////////////////////////////////////////////////////////
/// Enforce realizability conditions to Rij d.o.f.
void myRHEA::enforceRealizability(double &Rkk, double &phi1, double &phi2, double &phi3, double &xmap1, double &xmap2) {
    
    /// Realizability condition Rkk: Rkk >= 0.0
    Rkk = max(Rkk, 0.0);

    /// Realizability condition phi1, phi2, phi3: none

    /// Realizability condition xmap1, xmap2: 0 <= eigen-values <= 1
    vector<double> lambda(3, 0.0);
    barycentricCoord2eigVal(xmap1, xmap2, lambda);  // update lambda
    if ( !(0.0 <= lambda[0] && lambda[0] <= 1.0 &&
           0.0 <= lambda[1] && lambda[1] <= 1.0 &&
           0.0 <= lambda[2] && lambda[2] <= 1.0) ){ 
        // Enforce realizability by truncating & normalizing eigen-values
        truncateAndNormalizeEigVal(lambda);         // update lambda
    }
    eigVal2barycentricCoord(lambda, xmap1, xmap2);  // update xmap1, xmap2

}

///////////////////////////////////////////////////////////////////////////////
/// From rotation matrix of eigenvectors calculate Euler angles (convention Z-X-Z)
void myRHEA::eigVect2eulerAngles(const vector<vector<double>> &Q, double &phi1, double &phi2, double &phi3){
    // phi2       has range [0, pi]    (range of 'acos' function used in its calculation)
    // phi1, phi3 has range (-pi, pi]  (range of 'atan2' function used in their calculation)
    phi2 = std::acos(Q[2][2]);
    if (std::abs(std::sin(phi2)) > EPS) {
        /// Calculate phi1 using: Q[2][0] = s2 * s1 ; Q[2][1] = - s2 * c1 ;
        /// Calculate phi3 using: Q[0][2] = s3 * s2 ; Q[1][2] =   c3 * s2 ;
        phi1 = std::atan2(Q[2][0], - Q[2][1]);
        phi3 = std::atan2(Q[0][2], Q[1][2]);
    } else {
        /// Set phi3=0 (c3=1, s3=0), solve for phi1
        phi1 = std::atan2(Q[0][1], Q[0][0]);
        phi3 = 0.0;
    }
}

///////////////////////////////////////////////////////////////////////////////
/// From Euler angles (convention Z-X-Z) calculate rotation matrix of eigen-vectors
/*  Note: Expression of the rotation matrix of eigenvectors in terms of Euler angles
    extracted from Classical Mechanics 2nd Edition, H. Goldstein, 1908, pag 147, eq. 4-46,
    with notation phi: phi1, theta: phi2, psi: phi3
*/  
void myRHEA::eulerAngles2eigVect(const double &phi1, const double &phi2, const double &phi3, vector<vector<double>> &Q) {
    Q.assign(3, vector<double>(3, 0.0));
    // Calculate trigonometric values
    double c1 = cos(phi1);
    double s1 = sin(phi1);
    double c2 = cos(phi2);
    double s2 = sin(phi2);
    double c3 = cos(phi3);
    double s3 = sin(phi3);
    // Calculate the elements of the rotation matrix
    Q[0][0] =   c3 * c1 - c2 * s1 * s3 ;
    Q[0][1] =   c3 * s1 + c2 * c1 * s3 ;
    Q[0][2] =   s3 * s2 ;
    Q[1][0] = - s3 * c1 - c2 * s1 * c3 ;
    Q[1][1] = - s3 * s1 + c2 * c1 * c3 ;
    Q[1][2] =   c3 * s2 ;
    Q[2][0] =   s2 * s1 ;
    Q[2][1] = - s2 * c1 ;
    Q[2][2] =   c2 ;
}

///////////////////////////////////////////////////////////////////////////////
/// Transform eigen-values vector (lambda) to barycentric map coordinates (xmap1, xmap2) 
void myRHEA::eigVal2barycentricCoord(const vector<double> &lambda, double &xmap1, double &xmap2){
    // Assuming lambda is always of size 3
    xmap1 = lambda[0] * x1c[0] + lambda[1] * x2c[0] + lambda[2] * x3c[0];
    xmap2 = lambda[0] * x1c[1] + lambda[1] * x2c[1] + lambda[2] * x3c[1];
}

///////////////////////////////////////////////////////////////////////////////
/// Direct barycentric mapping: from eigenvalues diagonal matrix to barycentric coordinates
/// TODO: check this is equivalent to transformation eigVal2barycentricCoord
void myRHEA::eigValMatrix2barycentricCoord(const vector<vector<double>> &D, double &xmap1, double &xmap2){
    // Assuming D is always size [3,3]
    xmap1 = x1c[0] * (    D[0][0] - D[1][1]) \
          + x2c[0] * (2.0*D[1][1] - 2.0*D[2][2]) \
          + x3c[0] * (3.0*D[2][2] + 1.0);
    xmap2 = x1c[1] * (    D[0][0] - D[1][1]) \
          + x2c[1] * (2.0*D[1][1] - 2.0*D[2][2]) \
          + x3c[1] * (3.0*D[2][2] + 1.0);
}

///////////////////////////////////////////////////////////////////////////////
/* Transform barycentric map coordinates (xmap1, xmap2) to eigen-values vector (lambda)
        xmap1,2:     barycentric map coordinates
        lambda1,2,3: eigen-values, satifying lambda1 + lambda2 + lambda3 = 1
        transformation:  lambda1,2 = Tinv * (xmapping1,2 - t) 
 */
void myRHEA::barycentricCoord2eigVal(const double &xmap1, const double &xmap2, vector<double> &lambda){
    // Assuming lambda is always of size 3
    vector<double> xmap = {xmap1, xmap2};
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            lambda[i] += Tinv[i][j] * (xmap[j] - t[j]);
        }
    }
    lambda[2] = 1.0 - lambda[0] - lambda[1];    // from barycentric coord. condition sum(lambda_i) = 1.0
}

///////////////////////////////////////////////////////////////////////////////
/// Transform barycentric map coordinates (xmap1, xmap2) to eigen-values diagonal matrix (D)
void myRHEA::barycentricCoord2eigValMatrix(const double &xmap1, const double &xmap2, vector<vector<double>> &D){
    D.assign(3, vector<double>(3, 0.0));
    vector<double> xmap = {xmap1, xmap2};
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 2; j++){
            D[i][i] += Tinv[i][j] * (xmap[j] - t[j]);
        }
    }
    D[2][2] = 1.0 - D[0][0] - D[1][1];
}

///////////////////////////////////////////////////////////////////////////////
/// Transform Rij d.o.f. to Rij matrix
void myRHEA::Rijdof2matrix(const double &Rkk, const vector<vector<double>> &D, const vector<vector<double>> &Q, vector<vector<double>> &R){
    /// Assume R has dimensions [3][3]
    vector<vector<double>> A(3, vector<double>(3, 0.0));
    eigenDecomposition2Matrix(D, Q, A); // update A
    for (int q = 0; q < 3; q++){
        for (int r = 0; r < 3; r++){
            R[q][r] = Rkk * ((1.0 / 3.0) * Deltaij[q][r] + A[q][r]);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

/* Read witness file and extract witness points coordinates
    Reads witness points from 'witness_file', and defines attributes:
    twp_x_positions, twp_y_positions, twp_z_positions, num_witness_probes
*/
void myRHEA::readWitnessPoints() {

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0){
        cout << "\nReading witness points..." << endl;

        /// Read file (only with 1 mpi process to avoid file accessing errors)
        std::ifstream file(witness_file);
        if (!file.is_open()) {
            std::cerr << "Unable to open file: " << witness_file << std::endl;
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            double ix, iy, iz;
            if (!(iss >> ix >> iy >> iz)) {
                std::cerr << "Error reading line: " << line << std::endl;
                return;
            }
            
            /// Fill witness points position tensors
            twp_x_positions.push_back(ix);
            twp_y_positions.push_back(iy);
            twp_z_positions.push_back(iz);
        }
        file.close();
        num_witness_probes = static_cast<int>(twp_x_positions.size());
    
        cout << "Number of witness probes (global_state_size): " << num_witness_probes << endl;
    }

    // Broadcast the number of witness probes to all mpi processes
    MPI_Bcast(&num_witness_probes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize the vectors to hold data in all processes
    if (my_rank != 0) {
        twp_x_positions.resize(num_witness_probes);
        twp_y_positions.resize(num_witness_probes);
        twp_z_positions.resize(num_witness_probes);
    }

    // Broadcast the witness probes coordinates to all processes
    MPI_Bcast(twp_x_positions.data(), num_witness_probes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(twp_y_positions.data(), num_witness_probes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(twp_z_positions.data(), num_witness_probes, MPI_DOUBLE, 0, MPI_COMM_WORLD);

}

/* Pre-process witness points
   Builds 'temporal_witness_probes', vector of TemporalPointProbe elements 
   Inspired in SOD2D subroutine: CFDSolverBase_preprocWitnessPoints
*/
void myRHEA::preproceWitnessPoints() {
    
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0) {
        cout << "\nPreprocessing witness points..." << endl;
    }

    /// Construct (initialize) temporal point probes for witness points
    TemporalPointProbe temporal_witness_probe(mesh, topo);
    temporal_witness_probes.resize( num_witness_probes );
    for(int twp = 0; twp < num_witness_probes; ++twp) {
        /// Set parameters of temporal point probe
        temporal_witness_probe.setPositionX( twp_x_positions[twp] );
        temporal_witness_probe.setPositionY( twp_y_positions[twp] );
        temporal_witness_probe.setPositionZ( twp_z_positions[twp] );
        /// Insert temporal point probe to vector
        temporal_witness_probes[twp] = temporal_witness_probe;
        /// Locate closest grid point to probe
        temporal_witness_probes[twp].locateClosestGridPointToProbe();
    }

    /// Calculate state local size (num witness probes of my_rank) and debugging logs
    int state_local_size2_counter = 0;
    for(int twp = 0; twp < num_witness_probes; ++twp) {
        /// Owner rank writes to file
        if( temporal_witness_probes[twp].getGlobalOwnerRank() == my_rank ) {
            state_local_size2_counter += 1;
        }
    }
    
    // Each mpi process updates attribute 'state_local_size2'
#if _INCLUDE_YCOORD_INTO_RL_STATE_
    /// each observation point / slide includes 3-D state data: ( ||avg_u||_2, ||avg_v||_2, ||avg_w||_2, ||rmsf_u||_2, ||rmsf_v||_2, ||rmsf_w||_2, t_RL )
    this->state_local_size2 = 7 * state_local_size2_counter;
#else
    /// each observation point / slide includes 2-D state data: ( ||avg_u||_2, ||avg_v||_2, ||avg_w||_2, ||rmsf_u||_2, ||rmsf_v||_2, ||rmsf_w||_2, )
    this->state_local_size2 = 6 * state_local_size2_counter;
#endif
    cout << "Rank " << my_rank << " has num. local witness points: " << state_local_size2_counter << ", and state local size: " << state_local_size2 << endl;
    cout.flush();
    MPI_Barrier(MPI_COMM_WORLD); /// TODO: only here for cout debugging purposes, can be deleted?

}


/*  Read control cubes
    Control cubes are the 3d regions where RL control actions will be applied
    This method reads the file containing the 3 points defining the control cube
    Several cubes can be introduced

    Defines attributes 'num_control_vertices' and 'control_cubes_vertices' in all mpi processes
*/
/// Inspired in SOD2D subroutine: BLMARLFlowSolverIncomp_readControlRectangles
void myRHEA::readControlCubes(){

    /// Get attrib. 'num_control_cubes' and 'control_cubes_vertices' (in rank=0)
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0) {
        cout << "\nReading control cubes..." << endl;

        /// Read file (only with 1 mpi process to avoid file accessing errors)
        std::ifstream file(control_cubes_file);
        if (!file.is_open()) {
            cerr << "Unable to open file: " << control_cubes_file << endl;
            return;
        }
        
        /// Read first line, contains the number of control cubes
        file >> this->num_control_cubes;
        file.ignore();                                          // Ignore the newline character after 'num_control_cubes' integer
        
        /// Read all following lines, contains control cube vertices
        int cube_count   = 0;
        std::string empty_line;
        while (!file.eof()) { // finishes when end-of-file is reached
            std::array<std::array<double, 3>, 4> cube_vertices;
            for (int i = 0; i < 4; i++) {                       /// loop along the 4 vertices
                if (!(file >> cube_vertices[i][0] >> cube_vertices[i][1] >> cube_vertices[i][2])) {
                    cerr << "Error reading cube vertices" << endl;
                    return;
                }
            }
            control_cubes_vertices.push_back(cube_vertices);    // store single cube vertices in 'control_cubes_vertices' tensor
            cube_count++;                                       // update cube count
            std::getline(file, empty_line);                     // read the empty line separating cubes
        }

        /// Check num. cubes read & stored corresponds to expected 'num_control_cubes'
        if (cube_count != num_control_cubes) {
            cerr << "Mismatch between expected and actual number of cubes" << endl;
            cerr << "Additional info: cube_count = " << cube_count << ", num_control_cubes: " << num_control_cubes << endl;
            return;
        } 
        cout << "Number of control cubes: " << num_control_cubes << endl;

        file.close();

    }

    /// Broadcast the number of control cubes 'num_control_cubes' to all mpi processes
    MPI_Bcast(&num_control_cubes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /// Resize 'control_cubes_vertices' tensor in all processes with rank != 0
    if (my_rank != 0) {
        control_cubes_vertices.resize(num_control_cubes);       /// shape [num_control_cubes, 4, 3]
    }

    /// Broadcast the control cubes vertices coordinates from rank=0 to all processes
    MPI_Bcast(control_cubes_vertices.data(), num_control_cubes * 4 * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

}

/*  Locate grid points within mesh partition which are located inside each cube defined by 4 vertices of 3 coordinates
    Updates 'action_mask' field: 0.0 for no action, 1.0 for control points of 1st pseudo env (1st control cube, mpi rank 0), 2.0 for control points of 2ns pseudo env (2nd control cube, mpi rank 1), etc.
    Updates 'num_control_points', 'action_global_size2', 'n_rl_envs'
    Inspired in SOD2D subroutine: BLMARLFlowSolverIncomp_getControlNodes
*/
void myRHEA::getControlCubes() {

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
      
    /// Initialize local variables
    std::array<std::array<double,3>,4> cube_vertices; 
    std::array<double,3> dir1, dir2, dir3, cube_origin, vec_candidate;
    double size1, size2, size3, x, y, z, proj1, proj2, proj3;
    bool isInside;
    int num_control_points_local, num_control_points_per_cube_global, num_control_points_per_cube_local;
    control_cubes_y_central.resize(num_control_cubes);

    /// Set counters to 0
    n_rl_envs = 0;
    action_global_size2 = 0;                // global attribute
    num_control_points = 0;                 // global attribute
    num_control_points_local = 0;           // local var
    num_control_points_per_cube_global = 0; // global var
    num_control_points_per_cube_local = 0;  // local var

    /// Initialize action_mask to 0 everywhere
    action_mask = 0.0;
    int num_cubes_local = 0;                // number of cubes contained in local mpi process

    for (int icube = 0; icube < num_control_cubes; icube++) {

        num_control_points_per_cube_local  = 0;
        num_control_points_per_cube_global = 0;    
        
        /// Define cube vertices and direction vectors of cube -> cube defined by 3 direction vectors
        cube_vertices = control_cubes_vertices[icube];  // take cube data of cube #icube
        for (int icoord = 0; icoord < 3; icoord++) {
            cube_origin[icoord] = cube_vertices[0][icoord];
            dir1[icoord] = cube_vertices[1][icoord] - cube_vertices[0][icoord];     // direction vertex 0 -> vertex 1
            dir2[icoord] = cube_vertices[2][icoord] - cube_vertices[0][icoord];     // direction vertex 0 -> vertex 2
            dir3[icoord] = cube_vertices[3][icoord] - cube_vertices[0][icoord];     // direction vertex 0 -> vertex 3
        }

        /// y-coordinade of cube central point
        control_cubes_y_central[icube] = cube_origin[1] + 0.5 * ( dir1[1] + dir2[1] + dir3[1] );
        
        /// Transform director vectors to unitary vectors for further calculations of dot product
        size1 = myNorm(dir1);
        size2 = myNorm(dir2);
        size3 = myNorm(dir3);
        for (int icoord = 0; icoord < 3; icoord++) {
            dir1[icoord] = dir1[icoord] / size1;
            dir2[icoord] = dir2[icoord] / size2;
            dir3[icoord] = dir3[icoord] / size3;
        }

        /// Inner points: locate grid points located inside the cube & update action_mask
        for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
            for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                    
                    /// Geometric stuff
                    x = mesh->x[i];
                    y = mesh->y[j];
                    z = mesh->z[k];
                    
                    /// Define vector from cube origin to point
                    vec_candidate = {x - cube_origin[0], y - cube_origin[1], z - cube_origin[2]}; 
                    
                    /// Define vector projection to cube unitary vectors 
                    proj1 = myDotProduct(vec_candidate, dir1);
                    proj2 = myDotProduct(vec_candidate, dir2);
                    proj3 = myDotProduct(vec_candidate, dir3);
                    
                    /// Check condition: point is inside cube
                    isInside =    (0 <= proj1) && (proj1 <= size1)\
                               && (0 <= proj2) && (proj2 <= size2)\
                               && (0 <= proj3) && (proj3 <= size3);

                    /// Store point if inside the cube
                    if (isInside) {
                        action_mask[I1D(i,j,k)] = 1.0 + icube;
                        num_control_points_local++;
                        num_control_points_per_cube_local++;
                    }
                }
            }
        }

        /// Check if cube contains any control point
        MPI_Allreduce(&num_control_points_per_cube_local, &num_control_points_per_cube_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); // update num_control_points_per_cube_global
        if (num_control_points_per_cube_global == 0){
            cerr << "[myRHEA::getControlCubes] ERROR: Not found control points for cube #" << icube << endl;
            MPI_Abort( MPI_COMM_WORLD, 1);
        } else {
            n_rl_envs++;
        }

        /// Update number of cubes in each mpi process
        if (num_control_points_per_cube_local != 0) {
            num_cubes_local += 1;
        }

        /// Logging
        if (my_rank == 0) {
            cout << "[myRHEA::getControlCubes] Cube " << icube << " has " << num_control_points_per_cube_global << " control points" << endl;
        }
    }

    /// Calculate total num_control_points from local values
    MPI_Allreduce(&num_control_points_local, &num_control_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); // update 'num_control_points'

    /// Debugging: boundaries of RL environments / mpi processes 
    cout << "Rank " << my_rank << " has RL environment / mpi process domain inner boundaries: "  
         << "x in (" << x_field[I1D(topo->iter_common[_INNER_][_INIX_],0,0)] << ", " << x_field[I1D(topo->iter_common[_INNER_][_ENDX_],0,0)] << "), "
         << "y in (" << y_field[I1D(0,topo->iter_common[_INNER_][_INIY_],0)] << ", " << y_field[I1D(0,topo->iter_common[_INNER_][_ENDY_],0)] << "), "
         << "z in (" << z_field[I1D(0,0,topo->iter_common[_INNER_][_INIZ_])] << ", " << z_field[I1D(0,0,topo->iter_common[_INNER_][_ENDZ_])] << ")" << endl;
    cout << "Rank " << my_rank << " has inner boundaries local indices: "
         << "x-idx in (" << topo->iter_common[_INNER_][_INIX_] << ", " << topo->iter_common[_INNER_][_ENDX_] << "), "
         << "y-idx in (" << topo->iter_common[_INNER_][_INIY_] << ", " << topo->iter_common[_INNER_][_ENDY_] << "), "
         << "z-idx in (" << topo->iter_common[_INNER_][_INIZ_] << ", " << topo->iter_common[_INNER_][_ENDZ_] << ")" << endl;

    /// Check 1: num. of control cubes == num. mpi processes, and mpi processes must be distributed only along y-coordinate
    action_global_size2 = n_rl_envs * action_dim;
    if ((np_x == 1) && (np_y == n_rl_envs) && (np_z == 1) && (n_rl_envs == np_y) && (num_cubes_local == 1)) {
        /// Logging successful distribution
        if (my_rank == 0) {
            stringstream ss;
            ss << "[myRHEA::getControlCubes] Correct RL environments (control cubes) and computational domain distribution: "
               << "1 RL environment per MPI process distributed along y-coordinate, with "
               << "np_x: " << np_x << ", np_y: " << np_y << ", np_z: " << np_z << ", number of RL env: " << n_rl_envs;
            cout << ss.str() << endl;
        }
    } else {
        stringstream ss;
        ss << "[myRHEA::getControlCubes] ERROR: Invalid RL environments & computational domain distribution, with "
           << "np_x: " << np_x << ", np_y: " << np_y << ", np_z: " << np_z << ", number of RL env: " << n_rl_envs
           << ", Rank " << my_rank << " has " << num_cubes_local << " RL environments / control cubes";
        cerr << endl << ss.str() << endl;     
        MPI_Abort( MPI_COMM_WORLD, 1);
    }

    /// Check 2: action_mask has valid values
    MPI_Barrier(MPI_COMM_WORLD);            /// MPI_Barrier necessary for action_global_size2 to be updated by all mpi processes
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                if ( ( action_mask[I1D(i,j,k)] < 0.0 ) || ( static_cast<int>(action_mask[I1D(i,j,k)]) > action_global_size2 ) ) {
                    cerr << "OUT OF RANGE ERROR: Invalid action_mask index at " << i << ","  << j << "," << k << ": " << action_mask[I1D(i,j,k)] 
                            << ", action global size: " << action_global_size2 << endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
    }

    /// Check 3: action_global_size2 == n_rl_envs * action_dim
    if (action_global_size2 != n_rl_envs * action_dim){
        cerr << "Rank " << my_rank << " has action global size (" << action_global_size2 << ") != n_rl_envs (" << n_rl_envs << ") * action_dim (" << action_dim << ")" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    /// Logging
    if (my_rank == 0) {
        cout << "[myRHEA::getControlCubes] Total number of control points: " << num_control_points << endl;
        cout << "[myRHEA::getControlCubes] Action global size (num. cubes with at least 1 control point): " << action_global_size2 << endl;
        cout << "[myRHEA::getControlCubes] Number of RL pseudo-environments: " << n_rl_envs << endl; 
        cout << "[myRHEA::getControlCubes] Action dimension: " << action_dim << endl; 
        for (int icube = 0; icube < num_control_cubes; icube++) {
            cout << "[myRHEA::getControlCubes] Cube #" << icube << " has central point with y-coordinate: " << control_cubes_y_central[icube] << endl;
        }
    }
    cout.flush();
    MPI_Barrier(MPI_COMM_WORLD);

}

///////////////////////////////////////////////////////////////////////////////

void myRHEA::initializeFromRestart() {
    
    // Call the parent class method
    FlowSolverRHEA::initializeFromRestart();

    // Add additional functionality
#if _ACTIVE_CONTROL_BODY_FORCE_
    begin_actuation_time    += current_time;
    previous_actuation_time += current_time;
    final_time              += current_time;

    // Logging
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
        cout << fixed << setprecision(cout_precision);
        cout << "[myRHEA::initializeFromRestart] From restart current_time = " << scientific << current_time 
            << ", updated begin_actuation_time = " << scientific << begin_actuation_time
            << ", previous_actuation_time = " << scientific << previous_actuation_time
            << ", final_time = " << scientific << final_time << endl;
    }
#endif

}

///////////////////////////////////////////////////////////////////////////////
/// Calculate local state values (local to single mpi process, which corresponds to RL environment)
/// -> updates attributes: state_local
/// -> state values: turbulent kinetic energy
void myRHEA::updateState() {
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int state_local_size2_counter = 0;
    state_local.resize(state_local_size2);
    std::fill(state_local.begin(), state_local.end(), 0.0);
#if _WITNESS_XZ_SLICES_
    int j_index;
    int xz_slice_points_counter;
#else
    int i_index, j_index, k_index;
#endif  /// of _WITNESS_XZ_SLICES_

    for(int twp = 0; twp < num_witness_probes; ++twp) {
        /// Owner rank writes to file
        if( temporal_witness_probes[twp].getGlobalOwnerRank() == my_rank ) {

#if _WITNESS_XZ_SLICES_
            /// Get local index j
            j_index = temporal_witness_probes[twp].getLocalIndexJ(); 
            /// Calculate state value
            /// > Average state value (l2 error of rmsf_u_field) over regular grid xz-slice
            xz_slice_points_counter = 0;
            for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {

                    state_local[state_local_size2_counter]   += std::pow(avg_u_field[I1D(i,j_index,k)],  2.0);
                    state_local[state_local_size2_counter+1] += std::pow(avg_v_field[I1D(i,j_index,k)],  2.0);
                    state_local[state_local_size2_counter+2] += std::pow(avg_w_field[I1D(i,j_index,k)],  2.0);
                    state_local[state_local_size2_counter+3] += std::pow(rmsf_u_field[I1D(i,j_index,k)], 2.0);
                    state_local[state_local_size2_counter+4] += std::pow(rmsf_v_field[I1D(i,j_index,k)], 2.0);
                    state_local[state_local_size2_counter+5] += std::pow(rmsf_w_field[I1D(i,j_index,k)], 2.0);
#if _INCLUDE_YCOORD_INTO_RL_STATE_
                    state_local[state_local_size2_counter+6] += std::pow(y_field[I1D(i,j_index,k)],      2.0);
#endif /// of _INCLUDE_YCOORD_INTO_RL_STATE_
                    xz_slice_points_counter += 1;
                }
            }
            state_local[state_local_size2_counter]   = std::sqrt( state_local[state_local_size2_counter]   / xz_slice_points_counter );
            state_local[state_local_size2_counter+1] = std::sqrt( state_local[state_local_size2_counter+1] / xz_slice_points_counter );
            state_local[state_local_size2_counter+2] = std::sqrt( state_local[state_local_size2_counter+2] / xz_slice_points_counter );
            state_local[state_local_size2_counter+3] = std::sqrt( state_local[state_local_size2_counter+3] / xz_slice_points_counter );
            state_local[state_local_size2_counter+4] = std::sqrt( state_local[state_local_size2_counter+4] / xz_slice_points_counter );
            state_local[state_local_size2_counter+5] = std::sqrt( state_local[state_local_size2_counter+5] / xz_slice_points_counter );
#if _INCLUDE_YCOORD_INTO_RL_STATE_
            state_local[state_local_size2_counter+6] = std::sqrt( state_local[state_local_size2_counter+6] / xz_slice_points_counter );
#endif /// of _INCLUDE_YCOORD_INTO_RL_STATE_

#else ///  _WITNESS_XZ_SLICES_ 0
            /// Get local indices i, j, k
            i_index = temporal_witness_probes[twp].getLocalIndexI(); 
            j_index = temporal_witness_probes[twp].getLocalIndexJ(); 
            k_index = temporal_witness_probes[twp].getLocalIndexK();
            /// Calculate state value/s
            state_local[state_local_size2_counter]   = avg_u_field[I1D(i_index,j_index,k_index)];
            state_local[state_local_size2_counter+1] = avg_v_field[I1D(i_index,j_index,k_index)];
            state_local[state_local_size2_counter+2] = avg_w_field[I1D(i_index,j_index,k_index)];
            state_local[state_local_size2_counter+3] = rmsf_u_field[I1D(i_index,j_index,k_index)];
            state_local[state_local_size2_counter+4] = rmsf_v_field[I1D(i_index,j_index,k_index)];
            state_local[state_local_size2_counter+5] = rmsf_w_field[I1D(i_index,j_index,k_index)];
#if _INCLUDE_YCOORD_INTO_RL_STATE_
            state_local[state_local_size2_counter+6] = y_field[I1D(i_index,j_index,k_index)];
#endif /// of _INCLUDE_YCOORD_INTO_RL_STATE_
#endif /// of _WITNESS_XZ_SLICES_

            /// Update local state counter
#if _INCLUDE_YCOORD_INTO_RL_STATE_
            state_local_size2_counter += 7;
#else
            state_local_size2_counter += 6;
#endif
        }
    }
}   

///////////////////////////////////////////////////////////////////////////////

/// Calculate reward local value (local to single mpi process, which corresponds to RL environment)
/// If _RL_CONTROL_IS_SUPERVISED_ 1:
/// -> updates attributes: reward_local
/// else _RL_CONTROL_IS_SUPERVISED_ 0:
/// -> updates attributes: reward_local, avg_u_previous_field, rmsf_u_previous_field, rmsf_v_previous_field, rmsf_w_previous_field
void myRHEA::calculateReward() {
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    // Reward weight coefficients
    double b_param = 3.0;
    double d_param = 0.0;
    double c1 = 10.0;
    double c2 = 0.01;
    double c3 = 0.01;
    double c4 = 1.0;
    double c5 = 1.0;
    double c6 = 1.0;
    double c7 = 0.1;    // action penalization coefficient

#if _RL_CONTROL_IS_SUPERVISED_  /// Supervised Reward
    double l2_err_avg_u = 0.0,  l2_err_avg_v = 0.0,  l2_err_avg_w = 0.0;
    double l2_err_rmsf_u = 0.0, l2_err_rmsf_v = 0.0, l2_err_rmsf_w = 0.0;
    double l2_avg_u_reference  = 0.0;   // avg_v_reference, avg_w_reference = 0.0 everywhere in the profile
    double l2_rmsf_u_reference = 0.0, l2_rmsf_v_reference = 0.0, l2_rmsf_w_reference = 0.0;
    double l2_rl_f_rhou = 0.0, l2_rl_f_rhov = 0.0, l2_rl_f_rhow = 0.0, l2_rl_f = 0.0;
    double length_y = 0.0, delta_y  = 0.0;
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                delta_y             = 0.5 * ( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] );
                l2_err_avg_u        += std::pow(avg_u_field[I1D(i,j,k)]  - avg_u_reference_field[I1D(i,j,k)],  2.0) * delta_y;
                l2_err_avg_v        += std::pow(avg_v_field[I1D(i,j,k)]  - 0.0,                                2.0) * delta_y;
                l2_err_avg_w        += std::pow(avg_w_field[I1D(i,j,k)]  - 0.0,                                2.0) * delta_y;
                l2_err_rmsf_u       += std::pow(rmsf_u_field[I1D(i,j,k)] - rmsf_u_reference_field[I1D(i,j,k)], 2.0) * delta_y;
                l2_err_rmsf_v       += std::pow(rmsf_v_field[I1D(i,j,k)] - rmsf_v_reference_field[I1D(i,j,k)], 2.0) * delta_y;
                l2_err_rmsf_w       += std::pow(rmsf_w_field[I1D(i,j,k)] - rmsf_w_reference_field[I1D(i,j,k)], 2.0) * delta_y;
                l2_avg_u_reference  += std::pow(avg_u_reference_field[I1D(i,j,k)],  2.0) * delta_y;
                l2_rmsf_u_reference += std::pow(rmsf_u_reference_field[I1D(i,j,k)], 2.0) * delta_y;
                l2_rmsf_v_reference += std::pow(rmsf_v_reference_field[I1D(i,j,k)], 2.0) * delta_y;
                l2_rmsf_w_reference += std::pow(rmsf_w_reference_field[I1D(i,j,k)], 2.0) * delta_y;
                l2_rl_f_rhou        += std::pow(rl_f_rhou_field[I1D(i,j,k)],        2.0) * delta_y;
                l2_rl_f_rhov        += std::pow(rl_f_rhov_field[I1D(i,j,k)],        2.0) * delta_y;
                l2_rl_f_rhow        += std::pow(rl_f_rhow_field[I1D(i,j,k)],        2.0) * delta_y;
                length_y            += delta_y; 
            }
        }
    }
    l2_err_avg_u        = std::sqrt( l2_err_avg_u  / length_y );
    l2_err_avg_v        = std::sqrt( l2_err_avg_v  / length_y );
    l2_err_avg_w        = std::sqrt( l2_err_avg_w  / length_y );
    l2_err_rmsf_u       = std::sqrt( l2_err_rmsf_u / length_y );
    l2_err_rmsf_v       = std::sqrt( l2_err_rmsf_v / length_y );
    l2_err_rmsf_w       = std::sqrt( l2_err_rmsf_w / length_y );
    l2_avg_u_reference  = std::sqrt( l2_avg_u_reference  / length_y );
    l2_rmsf_u_reference = std::sqrt( l2_rmsf_u_reference / length_y );
    l2_rmsf_v_reference = std::sqrt( l2_rmsf_v_reference / length_y );
    l2_rmsf_w_reference = std::sqrt( l2_rmsf_w_reference / length_y );
    l2_rl_f_rhou        = std::sqrt( l2_rl_f_rhou / length_y );
    l2_rl_f_rhov        = std::sqrt( l2_rl_f_rhov / length_y );
    l2_rl_f_rhow        = std::sqrt( l2_rl_f_rhow / length_y );
    l2_rl_f             = std::sqrt( std::pow(l2_rl_f_rhou, 2.0) + std::pow(l2_rl_f_rhov, 2.0) + std::pow(l2_rl_f_rhow, 2.0) );
    reward_local = ( b_param - (   ( c1 * l2_err_avg_u  / l2_avg_u_reference ) \
                                 + ( c2 * l2_err_avg_v ) \
                                 + ( c3 * l2_err_avg_w ) \
                                 + ( c4 * l2_err_rmsf_u / l2_rmsf_u_reference ) \
                                 + ( c5 * l2_err_rmsf_v / l2_rmsf_v_reference ) \
                                 + ( c6 * l2_err_rmsf_w / l2_rmsf_w_reference ) \
                                 + ( c7 * l2_rl_f * 1.0 ) ) ) * std::pow(current_time - begin_actuation_time, d_param);
    /// Debugging
    cout << "[myRHEA::calculateReward] Rank " << my_rank << " has local reward: "  << reward_local << ", with reward terms:"
         << c1 * l2_err_avg_u  / l2_avg_u_reference << " " 
	     << c2 * l2_err_avg_v << " "
	     << c3 * l2_err_avg_w << " "
	     << c4 * l2_err_rmsf_u / l2_rmsf_u_reference << " "
	     << c5 * l2_err_rmsf_v / l2_rmsf_v_reference << " "
	     << c6 * l2_err_rmsf_w / l2_rmsf_w_reference << " "
         << c7 * l2_rl_f << endl
	     << "reward scaler dt_RL: " <<  std::pow(current_time - begin_actuation_time, d_param) << endl;

#else                           /// Unsupervised Reward
    /// Initialize variables
    double l2_d_avg_u = 0.0, l2_d_avg_v = 0.0, l2_d_avg_w = 0.0;
    double l2_d_rmsf_u = 0.0, l2_d_rmsf_v = 0.0, l2_d_rmsf_w = 0.0; 
    double l2_avg_u_previous  = 0.0;
    double l2_rmsf_u_previous = 0.0, l2_rmsf_v_previous = 0.0, l2_rmsf_w_previous = 0.0; 
    double l2_rl_f_rhou = 0.0, l2_rl_f_rhov = 0.0, l2_rl_f_rhow = 0.0, l2_rl_f = 0.0;
    double total_volume_local = 0.0;
    double delta_x, delta_y, delta_z, delta_volume;

    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Geometric stuff
                delta_x = 0.5*( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ); 
                delta_y = 0.5*( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ); 
                delta_z = 0.5*( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
                delta_volume =  delta_x * delta_y * delta_z;
                /// Calculate volume-averaged l2_d_avg_u & l2_d_rmsf_u
                l2_d_avg_u         += std::pow(avg_u_field[I1D(i,j,k)]  - avg_u_previous_field[I1D(i,j,k)], 2.0)  * delta_volume;
                l2_d_avg_v         += std::pow(avg_v_field[I1D(i,j,k)]  - 0.0, 2.0)  * delta_volume;
                l2_d_avg_w         += std::pow(avg_w_field[I1D(i,j,k)]  - 0.0, 2.0)  * delta_volume;
                l2_d_rmsf_u        += std::pow(rmsf_u_field[I1D(i,j,k)] - rmsf_u_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
                l2_d_rmsf_v        += std::pow(rmsf_v_field[I1D(i,j,k)] - rmsf_v_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
                l2_d_rmsf_w        += std::pow(rmsf_w_field[I1D(i,j,k)] - rmsf_w_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
                l2_avg_u_previous  += std::pow(avg_u_previous_field[I1D(i,j,k)], 2.0)  * delta_volume;
                l2_rmsf_u_previous += std::pow(rmsf_u_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
                l2_rmsf_v_previous += std::pow(rmsf_v_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
                l2_rmsf_w_previous += std::pow(rmsf_w_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
                l2_rl_f_rhou       += std::pow(rl_f_rhou_field[I1D(i,j,k)], 2.0) * delta_volume;
                l2_rl_f_rhov       += std::pow(rl_f_rhov_field[I1D(i,j,k)], 2.0) * delta_volume;
                l2_rl_f_rhow       += std::pow(rl_f_rhow_field[I1D(i,j,k)], 2.0) * delta_volume;
                total_volume_local += delta_volume;
            }
        }
    }
    l2_d_avg_u         = std::sqrt( l2_d_avg_u  / total_volume_local);
    l2_d_avg_v         = std::sqrt( l2_d_avg_v  / total_volume_local);
    l2_d_avg_w         = std::sqrt( l2_d_avg_w  / total_volume_local);
    l2_d_rmsf_u        = std::sqrt( l2_d_rmsf_u / total_volume_local);
    l2_d_rmsf_v        = std::sqrt( l2_d_rmsf_v / total_volume_local);
    l2_d_rmsf_w        = std::sqrt( l2_d_rmsf_w / total_volume_local);
    l2_avg_u_previous  = std::sqrt( l2_avg_u_previous  / total_volume_local);
    l2_rmsf_u_previous = std::sqrt( l2_rmsf_u_previous / total_volume_local);
    l2_rmsf_v_previous = std::sqrt( l2_rmsf_v_previous / total_volume_local);
    l2_rmsf_w_previous = std::sqrt( l2_rmsf_w_previous / total_volume_local);
    l2_rl_f_rhou       = std::sqrt( l2_rl_f_rhou / total_volume_local );
    l2_rl_f_rhov       = std::sqrt( l2_rl_f_rhov / total_volume_local );
    l2_rl_f_rhow       = std::sqrt( l2_rl_f_rhow / total_volume_local );
    l2_rl_f            = std::sqrt( std::pow(l2_rl_f_rhou, 2.0) + std::pow(l2_rl_f_rhov, 2.0) + std::pow(l2_rl_f_rhow, 2.0) );
    reward_local       = ( b_param - (   ( c1 * l2_d_avg_u / l2_avg_u_previous ) \
                                   + ( c2 * l2_d_avg_v ) \
                                   + ( c3 * l2_d_avg_w ) \
                                   + ( c4 * l2_d_rmsf_u / l2_rmsf_u_previous ) \
                                   + ( c5 * l2_d_rmsf_v / l2_rmsf_v_previous ) \
                                   + ( c6 * l2_d_rmsf_w / l2_rmsf_w_previous ) \
                                   + ( c7 * l2_rl_f ) ) ) * std::pow(current_time - begin_actuation_time, d_param);

    /// Debugging
    cout << "[myRHEA::calculateReward] Rank " << my_rank << ":" << endl
         << "local reward: "  << reward_local << endl
         << "reward terms: " 
         << c1 * l2_d_avg_u / l2_avg_u_previous << " " 
         << c2 * l2_d_avg_v << " "
         << c3 * l2_d_avg_w << " "
         << c4 * l2_d_rmsf_u / l2_rmsf_u_previous << " "
         << c5 * l2_d_rmsf_v / l2_rmsf_v_previous << " " 
         << c6 * l2_d_rmsf_w / l2_rmsf_w_previous << " " 
         << c7 * l2_rl_f << endl
         << "reward scaler dt_RL: " << std::pow(current_time - begin_actuation_time, d_param) << endl;
    
    /// Update avg_u,v,w_previous_field & rmsf_u,v,w_previous_field for next reward calculation
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                avg_u_previous_field[I1D(i,j,k)]  = avg_u_field[I1D(i,j,k)];
                rmsf_u_previous_field[I1D(i,j,k)] = rmsf_u_field[I1D(i,j,k)];
                rmsf_v_previous_field[I1D(i,j,k)] = rmsf_v_field[I1D(i,j,k)];
                rmsf_w_previous_field[I1D(i,j,k)] = rmsf_w_field[I1D(i,j,k)];
            }
        }
    } 
#endif

}

///////////////////////////////////////////////////////////////////////////////

/// MAIN
int main(int argc, char** argv) {

    /// Initialize MPI
    MPI_Init(&argc, &argv);

#ifdef _OPENACC
    /// OpenACC distribution on multiple accelerators (GPU)
    acc_device_t my_device_type;
    int num_devices, gpuId, local_rank;
    MPI_Comm shmcomm;    

    MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm );
    MPI_Comm_rank( shmcomm, &local_rank );           
    my_device_type = acc_get_device_type();                      
    num_devices = acc_get_num_devices( my_device_type );
    gpuId = local_rank % num_devices;
    acc_set_device_num( gpuId, my_device_type );
//    /// OpenACC distribution on multiple accelerators (GPU)
//    acc_device_t device_type = acc_get_device_type();
//    if ( acc_device_nvidia == device_type ) {
//       int ngpus = acc_get_num_devices( acc_device_nvidia );
//       int devicenum = atoi( getenv( "OMPI_COMM_WORLD_LOCAL_RANK" ) );
//       acc_set_device_num( devicenum, acc_device_nvidia );
//    }
//    acc_init(device_type);
#endif

    /// Process command line arguments
#if _ACTIVE_CONTROL_BODY_FORCE_
    string configuration_file, tag, restart_data_file, t_action, t_episode, t_begin_control, db_clustered, global_step;
    if (argc >= 9 ) {
        /// Extract and validate input arguments
        configuration_file  = argv[1];
        tag                 = argv[2];
        restart_data_file   = argv[3];
        t_action            = argv[4];  // 0.0001 
        t_episode           = argv[5];  // 1.0
        t_begin_control     = argv[6];  // 0.0
        db_clustered        = argv[7];  // False
        global_step         = argv[8];  // 0
    } else {
        cerr << "Proper usage: RHEA.exe configuration_file.yaml <tag> <restart_data_file> <t_action> <t_episode> <t_begin_control> <db_clustered> <global_step>" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    /// Construct my RHEA
    myRHEA my_RHEA( configuration_file, tag, restart_data_file, t_action, t_episode, t_begin_control, db_clustered, global_step );

#else
    string configuration_file;
    if( argc >= 2 ) {
        configuration_file = argv[1];
    } else {
        cout << "Proper usage: RHEA.exe configuration_file.yaml" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    /// Construct my RHEA
    myRHEA my_RHEA( configuration_file );

#endif

    /// Execute my RHEA
    my_RHEA.execute();

    /// Destruct my RHEA ... destructor is called automatically

    /// Finalize MPI
    MPI_Finalize();

    /// Return exit code of program
    return 0;

}
