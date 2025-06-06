#include "myRHEA.hpp"

#ifdef _OPENACC
#include <openacc.h>
#endif
#include <numeric>
#include <algorithm>
#include <cmath>        // std::pow

using namespace std;

////////// COMPILATION DIRECTIVES //////////
/// TODO: move these env parameters to an .h to be included by SmartRedisManager.cpp & .h files 
#define _FEEDBACK_LOOP_BODY_FORCE_ 0				/// Activate feedback loop for the body force moving the flow
#define _ACTIVE_CONTROL_BODY_FORCE_ 1               /// Activate active control for the body force
#define _FIXED_TIME_STEP_ 1                         /// Activate fixed time step
#define _REGULARIZE_RL_ACTION_ 1                    /// Activate regularization for RL action or RL source term w.r.t. momentum equation rhs 
#define _SPACE_AVERAGE_RL_ACTION_ 1
#define _RL_CONTROL_IS_SUPERVISED_ 1

/// AUXILIAR PARAMETERS ///
//const double pi = 2.0*asin( 1.0 );				/// pi number (fixed)

/// PROBLEM PARAMETERS ///
//const double R_specific = 287.058;				/// Specific gas constant
const double gamma_0    = 1.4;					    /// Heat capacity ratio
//const double c_p        = gamma_0*R_specific/( gamma_0 - 1.0 );	/// Isobaric heat capacity
const double delta      = 1.0;					    /// Channel half-height
const double Re_tau     = 180.0;				    /// Friction Reynolds number
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
const double kappa_vK   = 0.41;                                 /// von Kármán constant
const double y_0        = nu/( 9.0*u_tau );                     /// Smooth-wall roughness
const double u_0        = ( u_tau/kappa_vK )*( log( delta/y_0 ) + ( y_0/delta ) - 1.0 );        /// Volume average of a log-law velocity profile
const double alpha_u    = 1.0;                      /// Magnitude of velocity perturbations
const double alpha_P    = 0.1;                      /// Magnitude of pressure perturbations

const double fixed_time_step = 5.0e-5;				/// Fixed time step
const int cout_precision = 10;		                /// Output precision (fixed) 

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

#if _REGULARIZE_RL_ACTION_
const double reg_lambda = 0.5;
#endif

#if _SPACE_AVERAGE_RL_ACTION_
const double dy_space_averaging = 0.05;
#endif

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
// Reference profiles (_ALL_ coordinates, including boundaries)
const double rmsf_u_reference_profile[] = {   /// only inner points
    0.03690199, 0.1431543,  0.29195798, 0.48225412, 0.71205642,
    0.97664947, 1.26629004, 1.56505508, 1.85261352, 2.10877344, 2.31835921,
    2.47388978, 2.57533985, 2.62801624, 2.63998813, 2.62003648, 2.57639729,
    2.51619351, 2.44528228, 2.36829681, 2.2887528,  2.20919931, 2.13139338,
    2.05649897, 1.98526843, 1.91813391, 1.85525897, 1.79665625, 1.74224477,
    1.69185002, 1.64511183, 1.60155319, 1.56075297, 1.52229051, 1.48575306,
    1.45088976, 1.41756569, 1.38558302, 1.35464851, 1.32452854, 1.29504633,
    1.26605857, 1.23749725, 1.20930749, 1.18144407, 1.15382691, 1.1263415,
    1.09888169, 1.07135264, 1.04376051, 1.01622561, 0.98886524, 0.961823,
    0.93527317, 0.90938298, 0.8843933,  0.86065179, 0.83851344, 0.81831759,
    0.80045051, 0.78534573, 0.77341852, 0.76501258, 0.76041414, 0.75986774,
    0.76347267, 0.77113252, 0.78258477, 0.79739211, 0.81506093, 0.83519439,
    0.85742979, 0.88134533, 0.90650513, 0.93256037, 0.95926898, 0.98645905,
    1.0140314,  1.04189356, 1.06990284, 1.09793686, 1.12593849, 1.15387311,
    1.18171928, 1.20952308, 1.23741252, 1.26555016, 1.29406449, 1.32301877,
    1.35250179, 1.38268985, 1.41378505, 1.44603111, 1.47971288, 1.51521798,
    1.55298152, 1.59335986, 1.63668176, 1.6832809,  1.73342543, 1.78737121,
    1.84536218, 1.90761598, 1.97421662, 2.04497903, 2.11942243, 2.19678625,
    2.27593113, 2.35511504, 2.43181083, 2.50254356, 2.56271377, 2.60648167,
    2.62673914, 2.61527376, 2.56331886, 2.462785,   2.30831166, 2.09987094,
    1.84491662, 1.55860558, 1.26108918, 0.9726394,  0.70912843, 0.48026814,
    0.29075563, 0.14256579, 0.0367508,
};
const double rmsf_v_reference_profile[] = {   /// only inner points
    5.19263543e-05, 8.32340521e-04, 4.19425126e-03,
    1.14944157e-02, 2.40604517e-02, 4.25900905e-02, 6.73643331e-02,
    9.80268749e-02, 1.33979034e-01, 1.74318152e-01, 2.18174576e-01,
    2.64602748e-01, 3.12747371e-01, 3.61713621e-01, 4.10677271e-01,
    4.58820486e-01, 5.05426730e-01, 5.49848478e-01, 5.91562938e-01,
    6.30141763e-01, 6.65288987e-01, 6.96815154e-01, 7.24630288e-01,
    7.48715739e-01, 7.69137633e-01, 7.86019130e-01, 7.99533221e-01,
    8.09886598e-01, 8.17327011e-01, 8.22119809e-01, 8.24538002e-01,
    8.24847832e-01, 8.23299585e-01, 8.20140303e-01, 8.15605174e-01,
    8.09902551e-01, 8.03208258e-01, 7.95676249e-01, 7.87465476e-01,
    7.78705922e-01, 7.69489106e-01, 7.59872416e-01, 7.49905287e-01,
    7.39651952e-01, 7.29191215e-01, 7.18613188e-01, 7.08007946e-01,
    6.97458513e-01, 6.87040657e-01, 6.76810793e-01, 6.66815071e-01,
    6.57104095e-01, 6.47736174e-01, 6.38785991e-01, 6.30344254e-01,
    6.22504952e-01, 6.15352754e-01, 6.08956112e-01, 6.03365081e-01,
    5.98613586e-01, 5.94738862e-01, 5.91790703e-01, 5.89818873e-01,
    5.88853244e-01, 5.88893588e-01, 5.89916404e-01, 5.91887521e-01,
    5.94768386e-01, 5.98520029e-01, 6.03104269e-01, 6.08477898e-01,
    6.14592061e-01, 6.21396613e-01, 6.28843221e-01, 6.36881036e-01,
    6.45452094e-01, 6.54491455e-01, 6.63929515e-01, 6.73703672e-01,
    6.83768386e-01, 6.94079747e-01, 7.04572391e-01, 7.15151379e-01,
    7.25712933e-01, 7.36173502e-01, 7.46471558e-01, 7.56542929e-01,
    7.66307468e-01, 7.75675774e-01, 7.84558491e-01, 7.92845296e-01,
    8.00407132e-01, 8.07089044e-01, 8.12716571e-01, 8.17119885e-01,
    8.20124780e-01, 8.21535007e-01, 8.21131333e-01, 8.18668409e-01,
    8.13888400e-01, 8.06521755e-01, 7.96304984e-01, 7.82987938e-01,
    7.66344909e-01, 7.46185084e-01, 7.22379487e-01, 6.94856921e-01,
    6.63616901e-01, 6.28738437e-01, 5.90408763e-01, 5.48912095e-01,
    5.04662511e-01, 4.58181283e-01, 4.10125671e-01, 3.61228490e-01,
    3.12324973e-01, 2.64249695e-01, 2.17896522e-01, 1.74112681e-01,
    1.33836235e-01, 9.79337132e-02, 6.73078431e-02, 4.25588662e-02,
    2.40452015e-02, 1.14881643e-02, 4.19231267e-03, 8.32000724e-04,
    5.19304452e-05,
};
const double rmsf_w_reference_profile[] = {   /// only inner points
    0.01948791, 0.07236491, 0.14048191, 0.21854762, 0.30134447,
    0.38422926, 0.46372599, 0.53776682, 0.60578033, 0.66813924, 0.72556149,
    0.7784629,  0.82682753, 0.87027868, 0.90840991, 0.9410056,  0.96820929,
    0.99047726, 1.00846791, 1.02290692, 1.03446606, 1.0436535,  1.05080021,
    1.05610352, 1.05965133, 1.06143301, 1.06139017, 1.05945506, 1.05558825,
    1.04986742, 1.04247338, 1.03354293, 1.02316966, 1.01149968, 0.99872216,
    0.98504761, 0.97066364, 0.95568457, 0.94016161, 0.92414558, 0.90772899,
    0.89102661, 0.87407916, 0.85687722, 0.83937806, 0.82155696, 0.80350285,
    0.78535132, 0.76720944, 0.74919407, 0.73147353, 0.71422028, 0.69752028,
    0.68140774, 0.66594259, 0.65126839, 0.63760227, 0.62515297, 0.61408656,
    0.60457467, 0.59677854, 0.59081111, 0.5867123,  0.58446302, 0.58407969,
    0.58564841, 0.58921523, 0.59472467, 0.6020483,  0.61103605, 0.62153503,
    0.63340248, 0.64649235, 0.66062643, 0.6756682,  0.6915292,  0.70809044,
    0.72519448, 0.74264529, 0.76027573, 0.77797812, 0.79566103, 0.81328445,
    0.83091367, 0.84860092, 0.86626787, 0.88374511, 0.90091175, 0.91772015,
    0.93409562, 0.9499468,  0.96518311, 0.97970322, 0.99338264, 1.00605326,
    1.0176119,  1.02806158, 1.03734615, 1.04523526, 1.05145798, 1.05586077,
    1.0583537,  1.05889038, 1.05748795, 1.05420118, 1.04903966, 1.04192644,
    1.03265671, 1.02084506, 1.00598659, 0.98751452, 0.96483021, 0.93735162,
    0.90464737, 0.86655389, 0.82324995, 0.77511818, 0.72252004, 0.66544938,
    0.60346088, 0.5358123,  0.46212095, 0.38295741, 0.3003878,  0.21788092,
    0.14006928, 0.07215958, 0.01943448,
};
#endif
#endif

////////// myRHEA CLASS //////////

myRHEA::myRHEA(const string name_configuration_file, const string tag, const string restart_data_file, const string t_action, const string t_episode, const string t_begin_control, const string db_clustered, const string global_step) : FlowSolverRHEA(name_configuration_file) {

#if _ACTIVE_CONTROL_BODY_FORCE_
    rl_f_rhou_field.setTopology(topo, "rl_f_rhou_field");
    rl_f_rhov_field.setTopology(topo, "rl_f_rhov_field");
    rl_f_rhow_field.setTopology(topo, "rl_f_rhow_field");
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
    rmsf_u_reference_field.setTopology(topo, "rmsf_u_reference_field");
    rmsf_v_reference_field.setTopology(topo, "rmsf_v_reference_field");
    rmsf_w_reference_field.setTopology(topo, "rmsf_w_reference_field");
#endif
    initRLParams(tag, restart_data_file, t_action, t_episode, t_begin_control, db_clustered, global_step);
    initSmartRedis();
#endif

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
    this->witness_file = string(rl_case_path) + "/config_control_witness/witness8.txt";
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
    action_global_previous.resize(action_global_size2);
    action_global_instant.resize(action_global_size2);
    state_local.resize(state_local_size2);
    std::fill(action_global.begin(), action_global.end(), 0.0);
    std::fill(action_global_previous.begin(), action_global_previous.end(), 0.0);
    std::fill(action_global_instant.begin(), action_global_instant.end(), 0.0);
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
                rmsf_u_reference_field[I1D(i,j,k)] = rmsf_u_reference_profile[global_j];
                rmsf_v_reference_field[I1D(i,j,k)] = rmsf_v_reference_profile[global_j];
                rmsf_w_reference_field[I1D(i,j,k)] = rmsf_w_reference_profile[global_j];
            }
        }
        /// Debugging
        cout << "Rank " << my_rank << ": j=" << j << " (local), global_j=" << global_j 
             << ", Reference value of rmsf_u: " << rmsf_u_reference_profile[global_j]
             << ", rmsf_v: " << rmsf_v_reference_profile[global_j]
             << ", rmsf_w: " << rmsf_w_reference_profile[global_j] << endl;
    }
    /// -------------- Define control variables --------------
    l1_error_current  = 1.0;
    l1_error_previous = 1.0;
#else
    rmsf_u_field_local = 0.0;    rmsf_u_field_local_previous = 0.0;   rmsf_u_field_local_two_previous = 0.0;
    rmsf_v_field_local = 0.0;    rmsf_v_field_local_previous = 0.0;   rmsf_v_field_local_two_previous = 0.0;
    rmsf_w_field_local = 0.0;    rmsf_w_field_local_previous = 0.0;   rmsf_w_field_local_two_previous = 0.0;
#endif

    /// Initialize additional attribute members
#if _RL_CONTROL_IS_SUPERVISED_
    first_actuation_time_done   = false;
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
    double delta_y = mesh->getGloby(1) - mesh->getGloby(0);

    /// Calculate tau_wall_numerical
    double tau_w_numerical = mu*( global_avg_u_inner_w - global_avg_u_boundary_w )/delta_y;
    
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
                calculateReward();                  /// initializing 'rmsf_u_field_local_previous', 'rmsf_v_field_local_previous', 'rmsf_w_field_local_previous' (unsupervised) or 'l1_error_prevoius' (supervised)...
                first_actuation_time_done = true;
                if (my_rank == 0) {
                    cout << endl << endl << "[myRHEA::calculateSourceTerms] Initializing 'rmsf_u_field_local_previous', 'rmsf_v_field_local_previous', 'rmsf_w_field_local_previous'" << endl;
                }
            } 

            /// Check if new action is needed
            if (current_time - previous_actuation_time >= actuation_period) {

                if ( action_dim != 6 ) {
                    cerr << "[myRHEA::calculateSourceTerms] _ACTIVE_CONTROL_BODY_FORCE_=1 new action calculation only implemented for action_dim == 6, but action_dim = " << action_dim << endl;
                    MPI_Abort( MPI_COMM_WORLD, 1);
                }

                if ( !first_actuation_period_done ) {
                    calculateReward();              /// initializing 'rmsf_u_field_local_two_previous', 'rmsf_v_field_local_two_previous', 'rmsf_w_field_local_two_previous'
                    first_actuation_period_done = true;
                    previous_actuation_time = previous_actuation_time + actuation_period;
                    if (my_rank == 0) {
                        cout << endl << endl << "[myRHEA::calculateSourceTerms] Initializing 'rmsf_u_field_local_two_previous', 'rmsf_v_field_local_two_previous', 'rmsf_w_field_local_two_previous'" << endl;
                        cout << endl << "[myRHEA::calculateSourceTerms] RL control is activated at current time (" << scientific << current_time << ") " << "> time begin control (" << scientific << begin_actuation_time << ")" << endl;
                    }
                } else {

                    MPI_Barrier(MPI_COMM_WORLD);
                    timers->start( "rl_smartredis_communications" );

                    /// Logging
                    if (my_rank == 0) {
                        cout << endl << "[myRHEA::calculateSourceTerms] Performing SmartRedis communications (state, action, reward) at time instant " << current_time << endl;
                    }

                    /// Save old action values and time - useful for interpolating to new action
                    /// TODO: use action_global and action_global_previous for something (output writing) or delete them
                    action_global_previous  = action_global;     // TODO: delete 'action_global_previous' if not used, only necessary when using smooth action with 'smoothControlFunction'
                    previous_actuation_time = previous_actuation_time + actuation_period;   // TODO: delete if not used (if action smoothing is deactivated)

                    // SmartRedis communications 
                    // Writing state, reward, time
                    updateState();
                    manager->writeState(state_local, state_key);
                    calculateReward();                              /// update 'reward_local' attribute
                    manager->writeReward(reward_local, reward_key);
                    manager->writeTime(current_time, time_key);
                    // Reading read action...
                    manager->readAction(action_key);
                    action_global = manager->getActionGlobal();     /// action_global: vector<double> of size action_global_size2 = action_dim * n_rl_envs (currently only 1 action variable per rl env.)
                    /// action_local  = manager->getActionLocal();  /// action_local not used
                    /// Update & Write step size (from 1) to 0 if the next time that we require actuation value is the last one
                    if (current_time + 2.0 * actuation_period > final_time) {
                        manager->writeStepType(0, step_type_key);
                    }

                    MPI_Barrier(MPI_COMM_WORLD);
                    timers->stop( "rl_smartredis_communications" );


                    MPI_Barrier(MPI_COMM_WORLD);
                    timers->start( "rl_update_DeltaRij" );

                    /// Calculate smooth action 
                    /// OPTION 1 (smooth action implementation): if DeltaRii_field are re-calculated at each time instant,
                    /// This option requires loop "if (current_time - previous_actuation_time >= actuation_period) {" to finish here!
                    /// smoothControlFunction();        /// updates 'action_global_instant' 
                    /// OPTION 2 (discrete non-smooth action implementation): if DeltaRii_field are calculated just once for each action, discrete value step
                    for (int i_act=0; i_act<action_global_size2; i_act++) {
                        action_global_instant[i_act] = action_global[i_act];
                    }

                    /// Initialize variables
                    double Rkk, thetaZ, thetaY, thetaX, xmap1, xmap2; 
                    double DeltaRkk, DeltaThetaZ, DeltaThetaY, DeltaThetaX, DeltaXmap1, DeltaXmap2; 
                    double Rkk_inv, Akk;
                    bool   isNegligibleAction, isNegligibleRkk;
                    size_t actuation_idx;
                    vector<vector<double>> Aij(3, vector<double>(3, 0.0));
                    vector<vector<double>> Dij(3, vector<double>(3, 0.0));
                    vector<vector<double>> Qij(3, vector<double>(3, 0.0));
                    vector<vector<double>> RijPert(3, vector<double>(3, 0.0));
#if _SPACE_AVERAGE_RL_ACTION_
                    size_t actuation_idx_max = static_cast<size_t>(num_control_cubes) - 1;
                    double y_local_top_wall, y_local_bottom_wall;
                    double y_dist_to_local_top_wall, y_dist_to_local_bottom_wall;
                    double DeltaRkk_current, DeltaRkk_prev, DeltaRkk_next;
                    double y_ratio_aux;
                    double DeltaThetaZ_current, DeltaThetaZ_prev, DeltaThetaZ_next;          
                    double DeltaThetaY_current, DeltaThetaY_prev, DeltaThetaY_next;          
                    double DeltaThetaX_current, DeltaThetaX_prev, DeltaThetaX_next;          
                    double DeltaXmap1_current,  DeltaXmap1_prev,  DeltaXmap1_next;      
                    double DeltaXmap2_current,  DeltaXmap2_prev,  DeltaXmap2_next;      
                    int space_avg_bottom_counter = 0;
                    int space_avg_top_counter = 0;
                    int applied_action_counter = 0;
#endif
                    
                    /// Calculate DeltaRij = Rij_perturbated - Rij_original
                    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {

                                /// If [i,j,k] is a control point (non-zero action_mask) -> introduce action
                                if (action_mask[I1D(i,j,k)] != 0.0) {

                                    /// Get perturbation values from RL agent
                                    actuation_idx = static_cast<size_t>(action_mask[I1D(i,j,k)]) - 1;   /// type size_t
#if _SPACE_AVERAGE_RL_ACTION_
                                    applied_action_counter += 1;
                                    /// Apply action space-averaging if needed
                                    y_local_bottom_wall         = 0.5 * ( y_field[I1D(i,topo->iter_common[_INNER_][_INIY_],k)] + y_field[I1D(i,topo->iter_common[_INNER_][_INIY_]-1,k)] );
                                    y_local_top_wall            = 0.5 * ( y_field[I1D(i,topo->iter_common[_INNER_][_ENDY_],k)] + y_field[I1D(i,topo->iter_common[_INNER_][_ENDY_]+1,k)] );
                                    y_dist_to_local_bottom_wall = abs(y_local_bottom_wall - y_field[I1D(i,j,k)]);
                                    y_dist_to_local_top_wall    = abs(y_local_top_wall    - y_field[I1D(i,j,k)]);
                                    /// -> action averaging with previous action if close to local bottom wall (& not in global bottom wall, act_idx != 0)
                                    if ( ( y_dist_to_local_bottom_wall < dy_space_averaging ) && ( actuation_idx > 0) ) {
                                        /// define current (my_rank) and previous action (my_rank - 1)
                                        DeltaRkk_current    = action_global_instant[actuation_idx * action_dim + 0]; DeltaRkk_prev    = action_global_instant[(actuation_idx-1) * action_dim + 0];
                                        DeltaThetaZ_current = action_global_instant[actuation_idx * action_dim + 1]; DeltaThetaZ_prev = action_global_instant[(actuation_idx-1) * action_dim + 1];
                                        DeltaThetaY_current = action_global_instant[actuation_idx * action_dim + 2]; DeltaThetaY_prev = action_global_instant[(actuation_idx-1) * action_dim + 2];
                                        DeltaThetaX_current = action_global_instant[actuation_idx * action_dim + 3]; DeltaThetaX_prev = action_global_instant[(actuation_idx-1) * action_dim + 3];
                                        DeltaXmap1_current  = action_global_instant[actuation_idx * action_dim + 4]; DeltaXmap1_prev  = action_global_instant[(actuation_idx-1) * action_dim + 4];
                                        DeltaXmap2_current  = action_global_instant[actuation_idx * action_dim + 5]; DeltaXmap2_prev  = action_global_instant[(actuation_idx-1) * action_dim + 5];
                                        /// calculate space-averaged action
                                        y_ratio_aux = y_dist_to_local_bottom_wall / dy_space_averaging;     // <= 1, only = 1 when y_dist_to_local_bottom_wall == dy_space_averaging
                                        DeltaRkk    = ( ( DeltaRkk_current    + DeltaRkk_prev    ) / 2.0 ) + y_ratio_aux * ( ( DeltaRkk_current    - DeltaRkk_prev    ) / 2.0 );
                                        DeltaThetaZ = ( ( DeltaThetaZ_current + DeltaThetaZ_prev ) / 2.0 ) + y_ratio_aux * ( ( DeltaThetaZ_current - DeltaThetaZ_prev ) / 2.0 );
                                        DeltaThetaY = ( ( DeltaThetaY_current + DeltaThetaY_prev ) / 2.0 ) + y_ratio_aux * ( ( DeltaThetaY_current - DeltaThetaY_prev ) / 2.0 );
                                        DeltaThetaX = ( ( DeltaThetaX_current + DeltaThetaX_prev ) / 2.0 ) + y_ratio_aux * ( ( DeltaThetaX_current - DeltaThetaX_prev ) / 2.0 );
                                        DeltaXmap1  = ( ( DeltaXmap1_current  + DeltaXmap1_prev  ) / 2.0 ) + y_ratio_aux * ( ( DeltaXmap1_current  - DeltaXmap1_prev  ) / 2.0 );
                                        DeltaXmap2  = ( ( DeltaXmap2_current  + DeltaXmap2_prev  ) / 2.0 ) + y_ratio_aux * ( ( DeltaXmap2_current  - DeltaXmap2_prev  ) / 2.0 );
                                        /// Update counter for debugging
                                        space_avg_bottom_counter += 1;
                                    }
                                    /// -> action averaging with next action if close to local top wall (& not in global top wall, act_idx != )
                                    else if ( ( y_dist_to_local_top_wall < dy_space_averaging ) && ( actuation_idx < actuation_idx_max ) ) {
                                        /// define current (my_rank) and next action (my_rank + 1)
                                        DeltaRkk_current    = action_global_instant[actuation_idx * action_dim + 0]; DeltaRkk_next    = action_global_instant[(actuation_idx+1) * action_dim + 0];
                                        DeltaThetaZ_current = action_global_instant[actuation_idx * action_dim + 1]; DeltaThetaZ_next = action_global_instant[(actuation_idx+1) * action_dim + 1];
                                        DeltaThetaY_current = action_global_instant[actuation_idx * action_dim + 2]; DeltaThetaY_next = action_global_instant[(actuation_idx+1) * action_dim + 2];
                                        DeltaThetaX_current = action_global_instant[actuation_idx * action_dim + 3]; DeltaThetaX_next = action_global_instant[(actuation_idx+1) * action_dim + 3];
                                        DeltaXmap1_current  = action_global_instant[actuation_idx * action_dim + 4]; DeltaXmap1_next  = action_global_instant[(actuation_idx+1) * action_dim + 4];
                                        DeltaXmap2_current  = action_global_instant[actuation_idx * action_dim + 5]; DeltaXmap2_next  = action_global_instant[(actuation_idx+1) * action_dim + 5];
                                        /// calculate space-averaged action
                                        y_ratio_aux = y_dist_to_local_top_wall / dy_space_averaging;     // <= 1, only = 1 when y_dist_to_local_top_wall == dy_space_averaging
                                        DeltaRkk    = ( ( DeltaRkk_current    + DeltaRkk_next    ) / 2.0 ) + y_ratio_aux * ( ( DeltaRkk_current    - DeltaRkk_next    ) / 2.0 );
                                        DeltaThetaZ = ( ( DeltaThetaZ_current + DeltaThetaZ_next ) / 2.0 ) + y_ratio_aux * ( ( DeltaThetaZ_current - DeltaThetaZ_next ) / 2.0 );
                                        DeltaThetaY = ( ( DeltaThetaY_current + DeltaThetaY_next ) / 2.0 ) + y_ratio_aux * ( ( DeltaThetaY_current - DeltaThetaY_next ) / 2.0 );
                                        DeltaThetaX = ( ( DeltaThetaX_current + DeltaThetaX_next ) / 2.0 ) + y_ratio_aux * ( ( DeltaThetaX_current - DeltaThetaX_next ) / 2.0 );
                                        DeltaXmap1  = ( ( DeltaXmap1_current  + DeltaXmap1_next  ) / 2.0 ) + y_ratio_aux * ( ( DeltaXmap1_current  - DeltaXmap1_next  ) / 2.0 );
                                        DeltaXmap2  = ( ( DeltaXmap2_current  + DeltaXmap2_next  ) / 2.0 ) + y_ratio_aux * ( ( DeltaXmap2_current  - DeltaXmap2_next  ) / 2.0 );                                 
                                        /// Update counter for debugging
                                        space_avg_top_counter += 1;
                                    }
                                    /// -> action not space averaged, far from the pseudo-environment boundaries
                                    else {
                                        DeltaRkk    = action_global_instant[actuation_idx * action_dim + 0];
                                        DeltaThetaZ = action_global_instant[actuation_idx * action_dim + 1];
                                        DeltaThetaY = action_global_instant[actuation_idx * action_dim + 2];
                                        DeltaThetaX = action_global_instant[actuation_idx * action_dim + 3];
                                        DeltaXmap1  = action_global_instant[actuation_idx * action_dim + 4];
                                        DeltaXmap2  = action_global_instant[actuation_idx * action_dim + 5];
                                    }
#else                               /// Do not apply action space-averaging 
                                    DeltaRkk    = action_global_instant[actuation_idx * action_dim + 0];
                                    DeltaThetaZ = action_global_instant[actuation_idx * action_dim + 1];
                                    DeltaThetaY = action_global_instant[actuation_idx * action_dim + 2];
                                    DeltaThetaX = action_global_instant[actuation_idx * action_dim + 3];
                                    DeltaXmap1  = action_global_instant[actuation_idx * action_dim + 4];
                                    DeltaXmap2  = action_global_instant[actuation_idx * action_dim + 5];
#endif
                                    /// Calculate DeltaRij_field from DeltaRij d.o.f. (action), if action is not negligible 
                                    isNegligibleAction = (abs(DeltaRkk) < EPS && abs(DeltaThetaZ) < EPS && abs(DeltaThetaY) < EPS && abs(DeltaThetaX) < EPS && abs(DeltaXmap1) < EPS && abs(DeltaXmap2) < EPS);
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

                                        /// Eigen-vectors Euler Rotation angles (dof #2-4)
                                        eigVect2eulerAngles(Qij, thetaZ, thetaY, thetaX);      // update thetaZ, thetaY, thetaX

                                        /// Eigen-values Barycentric coordinates (dof #5-6)
                                        eigValMatrix2barycentricCoord(Dij, xmap1, xmap2);      // update xmap1, xmap2

                                        /// Build perturbed Rij d.o.f. -> x_new = x_old + Delta_x * x_old
                                        /// Delta_* are standarized values between 'action_bounds' RL parameter
                                        Rkk    = Rkk    * (1 + DeltaRkk);
                                        thetaZ = thetaZ * (1 + DeltaThetaZ);
                                        thetaY = thetaY * (1 + DeltaThetaY);
                                        thetaX = thetaX * (1 + DeltaThetaX);
                                        xmap1  = xmap1  * (1 + DeltaXmap1);
                                        xmap2  = xmap2  * (1 + DeltaXmap2);

                                        /// Enforce realizability to perturbed Rij d.o.f
                                        enforceRealizability(Rkk, thetaZ, thetaY, thetaX, xmap1, xmap2);    // update Rkk, thetaZ, thetaY, thetaX, xmap1, xmap2, if necessary

                                        /// Calculate perturbed & realizable Rij
                                        eulerAngles2eigVect(thetaZ, thetaY, thetaX, Qij);                   // update Qij
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
#if _SPACE_AVERAGE_RL_ACTION_
                    /// Debugging
                    cout << "Rank " << my_rank << " has applied accion in " << applied_action_counter << " points, with " 
                         << " space averaging in " << space_avg_bottom_counter << " (bottom) and " << space_avg_top_counter << " (top) points" << endl; 
#endif
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
                            }
                        }
                    }

                    MPI_Barrier(MPI_COMM_WORLD);
                    timers->stop( "rl_update_control_term" );

                }   /// end if ( !first_actuation_period_done )
            }       /// end if (current_time - previous_actuation_time >= actuation_period), new action was required

        } else {    /// current_time <= begin_actuation_time

            for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                    for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                        rl_f_rhou_field[I1D(i,j,k)] = 0.0;
                        rl_f_rhov_field[I1D(i,j,k)] = 0.0;
                        rl_f_rhow_field[I1D(i,j,k)] = 0.0;
                    }
                }
            }
            if (my_rank == 0) {
                cout << "[myRHEA::calculateSourceTerms] RL Control is NOT applied yet, as current time (" << scientific << current_time << ") " << "< time begin control (" << scientific << begin_actuation_time << ")" << endl;
            }
        }
    }               /// end if (rk_time_stage == 1)
    
#endif

    /// Update halo values
    f_rhou_field.update();
    f_rhov_field.update();
    f_rhow_field.update();
    f_rhoE_field.update();
    rl_f_rhou_field.update();
    rl_f_rhov_field.update();
    rl_f_rhow_field.update();

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

/// TODO: check, remove function if not used anymore for checks
void myRHEA::timeAdvanceConservedVariables() {

    /// TODO: check, remove lines below
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /// Debugging variables
    double rhou_inv_flux_ratio   = 0.0;
    double rhou_vis_flux_ratio   = 0.0;
    double f_rhou_field_ratio    = 0.0;
    double rl_f_rhou_field_ratio = 0.0;
    int ratio_counter            = 0;
#if _REGULARIZE_RL_ACTION_
    /// ---- smooth regularization of RL control load by hyperbolic tangent function ----
    /// Apply smooth regularization when RL control term << RHS term
    double rhou_rhs, rhov_rhs, rhow_rhs;
    double rhou_rl_f, rhov_rl_f, rhow_rl_f;
    double rhou_rl_f_reg, rhov_rl_f_reg, rhow_rl_f_reg;
    /// Debugging additional variables
    double rl_f_rhou_field_reg_factor = 0.0;
    int saturated_actions_counter = 0;
    /// ---- smooth regularization of RL control load by hyperbolic tangent function ----
#endif

    /// Coefficients of explicit Runge-Kutta stages
    double rk_a = 0.0, rk_b = 0.0, rk_c = 0.0;
    runge_kutta_method->setStageCoefficients(rk_a,rk_b,rk_c,rk_time_stage);    

    /// Inner points: rho, rhou, rhov, rhow and rhoE
    double f_rhouvw = 0.0;
    double rho_rhs_flux = 0.0, rhou_rhs_flux = 0.0, rhov_rhs_flux = 0.0, rhow_rhs_flux = 0.0, rhoE_rhs_flux = 0.0;
#if _OPENACC_MANUAL_DATA_MOVEMENT_
    const int local_size_x = _lNx_;
    const int local_size_y = _lNy_;
    const int local_size_z = _lNz_;
    const int local_size   = local_size_x*local_size_y*local_size_z;
    const int inix = topo->iter_common[_INNER_][_INIX_];
    const int iniy = topo->iter_common[_INNER_][_INIY_];
    const int iniz = topo->iter_common[_INNER_][_INIZ_];
    const int endx = topo->iter_common[_INNER_][_ENDX_];
    const int endy = topo->iter_common[_INNER_][_ENDY_];
    const int endz = topo->iter_common[_INNER_][_ENDZ_];
    #pragma acc enter data copyin (this)
    #pragma acc data copyin (u_field.vector[0:local_size],v_field.vector[0:local_size],w_field.vector[0:local_size])
    #pragma acc data copyin (rho_0_field.vector[0:local_size],rhou_0_field.vector[0:local_size],rhov_0_field.vector[0:local_size],rhow_0_field.vector[0:local_size],rhoE_0_field.vector[0:local_size])
    #pragma acc data copyin (rho_inv_flux.vector[0:local_size],rhou_inv_flux.vector[0:local_size],rhov_inv_flux.vector[0:local_size],rhow_inv_flux.vector[0:local_size],rhoE_inv_flux.vector[0:local_size])
    #pragma acc data copyin (rhou_vis_flux.vector[0:local_size],rhov_vis_flux.vector[0:local_size],rhow_vis_flux.vector[0:local_size],rhoE_vis_flux.vector[0:local_size])
    #pragma acc data copyin (f_rhou_field.vector[0:local_size],f_rhov_field.vector[0:local_size],f_rhow_field.vector[0:local_size],f_rhoE_field.vector[0:local_size])
    #pragma acc enter data copyin (rho_field.vector[0:local_size],rhou_field.vector[0:local_size],rhov_field.vector[0:local_size],rhow_field.vector[0:local_size],rhoE_field.vector[0:local_size])
    #pragma acc parallel loop collapse (3)
    for(int i = inix; i <= endx; i++) {
        for(int j = iniy; j <= endy; j++) {
            for(int k = iniz; k <= endz; k++) {
#else
    #pragma acc parallel loop collapse (3)
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
#endif
#if _REGULARIZE_RL_ACTION_
                /// ---- smooth regularization of RL control load by hyperbolic tangent function ----
                /// Calculate RL control load ratio wrt RHS with no RL control load
                rhou_rhs      = ( -1.0 ) * rhou_inv_flux[I1D(i,j,k)] + rhou_vis_flux[I1D(i,j,k)] + f_rhou_field[I1D(i,j,k)];
                rhov_rhs      = ( -1.0 ) * rhov_inv_flux[I1D(i,j,k)] + rhov_vis_flux[I1D(i,j,k)] + f_rhov_field[I1D(i,j,k)];
                rhow_rhs      = ( -1.0 ) * rhow_inv_flux[I1D(i,j,k)] + rhow_vis_flux[I1D(i,j,k)] + f_rhow_field[I1D(i,j,k)];
                rhou_rl_f     = rl_f_rhou_field[I1D(i,j,k)];
                rhov_rl_f     = rl_f_rhov_field[I1D(i,j,k)];
                rhow_rl_f     = rl_f_rhow_field[I1D(i,j,k)];
                /// Apply smooth regularization approach when |rl_f| < |rhs|/10.0, E = 0.1 achieves regularization to be smooth enough
                /// TODO: improve regularization, not smooth! (specially else if)
                if ( std::abs(rhou_rl_f) < ( std::abs(rhou_rhs) * reg_lambda ) ) {
                    rhou_rl_f_reg = ( rhou_rhs * reg_lambda ) * std::tanh( rhou_rl_f / ( 0.5 * ( rhou_rhs * reg_lambda ) + EPS) );
                    rhov_rl_f_reg = ( rhov_rhs * reg_lambda ) * std::tanh( rhov_rl_f / ( 0.5 * ( rhov_rhs * reg_lambda ) + EPS) );
                    rhow_rl_f_reg = ( rhow_rhs * reg_lambda ) * std::tanh( rhow_rl_f / ( 0.5 * ( rhow_rhs * reg_lambda ) + EPS) );
                } else { /// saturate actions
                    rhou_rl_f_reg = rhou_rl_f  * int( abs( (rhou_rhs * reg_lambda) / (rhou_rl_f + EPS) ) ); 
                    rhov_rl_f_reg = rhov_rl_f  * int( abs( (rhov_rhs * reg_lambda) / (rhov_rl_f + EPS) ) ); 
                    rhow_rl_f_reg = rhow_rl_f  * int( abs( (rhow_rhs * reg_lambda) / (rhow_rl_f + EPS) ) ); 
                    saturated_actions_counter += 1;
                }
                /// TODO: improve this regularization, check commented code above!
                /// Update 'rl_f_rhow_field'
                rl_f_rhou_field[I1D(i,j,k)] = rhou_rl_f_reg;
                rl_f_rhov_field[I1D(i,j,k)] = rhov_rl_f_reg;
                rl_f_rhow_field[I1D(i,j,k)] = rhow_rl_f_reg;
                /// ---- smooth regularization of RL control load by hyperbolic tangent function ----
#endif
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
#if _REGULARIZE_RL_ACTION_
                    rl_f_rhou_field_reg_factor   += std::abs( rhou_rl_f_reg               / ( rhou_rl_f + EPS) );
#endif
                    ratio_counter += 1;
                }
	        }
        }
    }
    rhou_inv_flux_ratio          /= ratio_counter;
    rhou_vis_flux_ratio          /= ratio_counter;
    f_rhou_field_ratio           /= ratio_counter;
    rl_f_rhou_field_ratio        /= ratio_counter;
#if _REGULARIZE_RL_ACTION_
    rl_f_rhou_field_reg_factor   /= ratio_counter;
#endif
    
    /// Summarized output
    if ( my_rank < 4) {
        cout << endl << "Rank " << my_rank << " u-RHS Ratios: " << rhou_inv_flux_ratio << ", " << rhou_vis_flux_ratio << ", " << f_rhou_field_ratio << ", " << rl_f_rhou_field_ratio << endl;
#if _REGULARIZE_RL_ACTION_
        cout << "Rank " << my_rank << " u-RHS Ratio f_rl / f_rl_nonReg: " << rl_f_rhou_field_reg_factor << ", # saturated: " << saturated_actions_counter << endl;
#endif
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
/// TODO: end remove


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
void myRHEA::enforceRealizability(double &Rkk, double &thetaZ, double &thetaY, double &thetaX, double &xmap1, double &xmap2) {
    
    /// Realizability condition Rkk: Rkk >= 0.0
    Rkk = max(Rkk, 0.0);

    /// Realizability condition thetaZ, thetaY, thetaX: none

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
/* Calculate rotation angles from rotation matrix of eigenvectors
   Attention: the rotation matrix of eigen-vectors must be indeed a proper rotation matrix. 
   A proper rotation matrix is orthogonal (meaning its inverse is its transpose) and has a determinant of +1.
   This ensures that the matrix represents a rotation without improper reflection or scaling.
   This has been check to be satisfied (+ computational error) at 15 feb. 2024 
*/
void myRHEA::eigVect2eulerAngles(const vector<vector<double>> &Q, double &thetaZ, double &thetaY, double &thetaX){
    
    // thetaY         has range [-pi/2, pi/2] (range of 'asin' function used in its calculation)
    // thetaZ, thetaX has range (-pi, pi]     (range of 'atan2' function used in their calculation)
    thetaY = std::asin(-Q[2][0]);
    if (std::abs(std::cos(thetaY)) > EPS) { // Avoid gimbal lock
        thetaZ = std::atan2(Q[1][0], Q[0][0]);
        thetaX = std::atan2(Q[2][1], Q[2][2]);
    } else {                                  // Gimbal lock, set yaw to 0 and calculate roll
        thetaZ = 0;
        thetaX = std::atan2(-Q[0][1], Q[1][1]);
    }

}

///////////////////////////////////////////////////////////////////////////////
/// Transform Euler angles to rotation matrix of eigen-vectors
void myRHEA::eulerAngles2eigVect(const double &thetaZ, const double &thetaY, const double &thetaX, vector<vector<double>> &Q) {
    Q.assign(3, vector<double>(3, 0.0));
    // Calculate trigonometric values
    double cz = cos(thetaZ);
    double sz = sin(thetaZ);
    double cy = cos(thetaY);
    double sy = sin(thetaY);
    double cx = cos(thetaX);
    double sx = sin(thetaX);
    // Calculate the elements of the rotation matrix
    Q[0][0] = cy * cz;
    Q[0][1] = cy * sz;
    Q[0][2] = -sy;
    Q[1][0] = (sx * sy * cz) - (cx * sz);
    Q[1][1] = (sx * sy * sz) + (cx * cz);
    Q[1][2] = sx * cy;
    Q[2][0] = (cx * sy * cz) + (sx * sz);
    Q[2][1] = (cx * sy * sz) - (sx * cz);
    Q[2][2] = cx * cy;
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
    this->state_local_size2 = state_local_size2_counter;  // each mpi process updates attribute 'state_local_size2'
    cout << "Rank " << my_rank << " has " << state_local_size2 << " local witness points" << endl;
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
    int i_index, j_index, k_index;
    int state_local_size2_counter = 0;
    state_local.resize(state_local_size2);
    std::fill(state_local.begin(), state_local.end(), 0.0);
    for(int twp = 0; twp < num_witness_probes; ++twp) {
        /// Owner rank writes to file
        if( temporal_witness_probes[twp].getGlobalOwnerRank() == my_rank ) {
            /// Get local indices i, j, k
            i_index = temporal_witness_probes[twp].getLocalIndexI(); 
            j_index = temporal_witness_probes[twp].getLocalIndexJ(); 
            k_index = temporal_witness_probes[twp].getLocalIndexK();
            /// Get state data: turbulent kinetic energy
            state_local[state_local_size2_counter] = 0.5 * (   pow(rmsf_u_field[I1D(i_index,j_index,k_index)], 2.0) 
                                                             + pow(rmsf_v_field[I1D(i_index,j_index,k_index)], 2.0)
                                                             + pow(rmsf_w_field[I1D(i_index,j_index,k_index)], 2.0) );
            state_local_size2_counter += 1;
        }
    }
}   

///////////////////////////////////////////////////////////////////////////////

/// Calculate reward local value (local to single mpi process, which corresponds to RL environment)
/// If _RL_CONTROL_IS_SUPERVISED_ 1:
/// -> updates attributes: reward_local, l1_error_current, l1_error_previous
/// else _RL_CONTROL_IS_SUPERVISED_ 0:
/// -> updates attributes: reward_local, rmsf_u_field_local, rmsf_u_field_local_previous, rmsf_u_field_local_two_previous
///                                      rmsf_v_field_local, rmsf_v_field_local_previous, rmsf_v_field_local_two_previous
///                                      rmsf_w_field_local, rmsf_w_field_local_previous, rmsf_w_field_local_two_previous
void myRHEA::calculateReward() {
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

#if _RL_CONTROL_IS_SUPERVISED_  /// Supervised Reward
    double l1_err_rmsf_u  = 0.0;
    double l1_err_rmsf_v  = 0.0;
    double l1_err_rmsf_w  = 0.0;
    int    l1_err_counter = 0;
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                l1_err_rmsf_u  += std::abs( (rmsf_u_field[I1D(i,j,k)] - rmsf_u_reference_field[I1D(i,j,k)]) );
                l1_err_rmsf_v  += std::abs( (rmsf_v_field[I1D(i,j,k)] - rmsf_v_reference_field[I1D(i,j,k)]) );
                l1_err_rmsf_w  += std::abs( (rmsf_w_field[I1D(i,j,k)] - rmsf_w_reference_field[I1D(i,j,k)]) );
                l1_err_counter += 1;
            }
        }
    }
    l1_err_rmsf_u /= l1_err_counter;
    l1_err_rmsf_v /= l1_err_counter;
    l1_err_rmsf_w /= l1_err_counter;
    l1_error_current = ( l1_err_rmsf_u + l1_err_rmsf_v + l1_err_rmsf_w ) / 3.0;
    reward_local = ( l1_error_previous - l1_error_current ) / ( l1_error_previous + EPS );
    /// Debugging
    cout << "[myRHEA::calculateReward] Rank " << my_rank << " has local reward: "  << reward_local << ", L1 error: " << l1_error_current << ", L1 error previous: " << l1_error_previous << endl;
    /// Update 'l1_error_previous'
    l1_error_previous = l1_error_current;

#else                           /// Unsupervised Reward
    /// Initialize variables
    rmsf_u_field_local = 0.0; 
    rmsf_v_field_local = 0.0; 
    rmsf_w_field_local = 0.0; 
    double total_volume_local = 0.0;
    double delta_x, delta_y, delta_z, delta_volume;
    double d_rmsf_u_field_local_previous, d_rmsf_u_field_local;
    double d_rmsf_v_field_local_previous, d_rmsf_v_field_local;
    double d_rmsf_w_field_local_previous, d_rmsf_w_field_local;

    /// Calculate avg_u_field_rl_envs, the temporal-average of u_field space-averaged over each rl environment
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Geometric stuff
                delta_x = 0.5*( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ); 
                delta_y = 0.5*( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ); 
                delta_z = 0.5*( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
                delta_volume =  delta_x * delta_y * delta_z;
                /// Calculate volume-averaged rmsf_u, rmsf_v, rmsf_w
                rmsf_u_field_local += rmsf_u_field[I1D(i,j,k)] * delta_volume;
                rmsf_v_field_local += rmsf_v_field[I1D(i,j,k)] * delta_volume;
                rmsf_w_field_local += rmsf_w_field[I1D(i,j,k)] * delta_volume;
                total_volume_local += delta_volume;
            }
        }
    }
    rmsf_u_field_local /= total_volume_local;
    rmsf_v_field_local /= total_volume_local;
    rmsf_w_field_local /= total_volume_local;
    d_rmsf_u_field_local_previous = std::abs( rmsf_u_field_local_two_previous - rmsf_u_field_local_previous );
    d_rmsf_v_field_local_previous = std::abs( rmsf_v_field_local_two_previous - rmsf_v_field_local_previous );
    d_rmsf_w_field_local_previous = std::abs( rmsf_w_field_local_two_previous - rmsf_w_field_local_previous );
    d_rmsf_u_field_local          = std::abs( rmsf_u_field_local_previous     - rmsf_u_field_local );
    d_rmsf_v_field_local          = std::abs( rmsf_v_field_local_previous     - rmsf_v_field_local );
    d_rmsf_w_field_local          = std::abs( rmsf_w_field_local_previous     - rmsf_w_field_local );
    reward_local = ( d_rmsf_u_field_local_previous - d_rmsf_u_field_local )
                 + ( d_rmsf_v_field_local_previous - d_rmsf_v_field_local )
                 + ( d_rmsf_w_field_local_previous - d_rmsf_w_field_local );
    ///  TODO: maybe do relative adding: / std::abs( d_avg_u_field_local_previous + EPS );

    /// Debugging
    cout << "[myRHEA::calculateReward] Rank " << my_rank << " has local reward: "  << reward_local << ", with rmsf_u (k-2): " << rmsf_u_field_local_two_previous << ", rmsf_u (k-1): " << rmsf_u_field_local_previous << ", rmsf_u (k): " << rmsf_u_field_local 
         << ", d_rmsf_u (k-1): " << d_rmsf_u_field_local_previous << ", d_rmsf_u (k): " << d_rmsf_u_field_local << endl;
    
    /// Update rmsf_u,v,w_field_local_previous & rmsf_u,v,w_field_local_two_previous for next reward calculation
    rmsf_u_field_local_two_previous = rmsf_u_field_local_previous;
    rmsf_v_field_local_two_previous = rmsf_v_field_local_previous;
    rmsf_w_field_local_two_previous = rmsf_w_field_local_previous;
    rmsf_u_field_local_previous     = rmsf_u_field_local;
    rmsf_v_field_local_previous     = rmsf_v_field_local;
    rmsf_w_field_local_previous     = rmsf_w_field_local;
#endif

}

///////////////////////////////////////////////////////////////////////////////
/// Update 'action_global_instant' - smooth transition from action_global_previous and action_global during all actuation_period
void myRHEA::smoothControlFunction() {
    double actuation_period_fraction, f1, f2, f3;
    actuation_period_fraction = ( current_time - previous_actuation_time ) / actuation_period;
    f1 = exp(-1.0 / actuation_period_fraction);
    f2 = exp(-1.0 / (1.0 - actuation_period_fraction));
    f3 = f1 / (f1 + f2);
    for (int idx=0; idx<action_global_size2; idx++) {
        action_global_instant[idx] = action_global_previous[idx] + f3 * (action_global[idx] - action_global_previous[idx]);
    }
    /// Logging
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank == 0) {
        cout << "[myRHEA::smoothControlFunction] Rank " << my_rank << " has smooth global action: ";
        for (int idx=0; idx<action_global_size2; idx++) {
            cout << action_global_instant[idx] << " ";
        }
        cout << endl;
    }
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
