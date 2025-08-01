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
#define _CORRECT_U_BULK_ 0                          /// Activate correction of u_bulk
#define _FIXED_TIME_STEP_ 1                         /// Activate fixed time step
#define _ACTIVE_CONTROL_BODY_FORCE_ 1               /// Activate active control for the body force
#define _RL_CONTROL_IS_SUPERVISED_ 1
#define _TEMPORAL_SMOOTHING_RL_ACTION_ 1
#define _WITNESS_XYZ_AVG_ 1
#define _RL_EARLY_EPISODE_TERMINATION_FUNC_U_BULK_ 1
#define _ZERO_NET_FLUX_PERTURBATION_LOAD_ 0


const int fstream_precision = 15;	                /// Fstream precision (fixed)

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
//const double kappa    = c_p*mu/Pr;				/// Thermal conductivity	
const double Re_b       = pow( Re_tau/0.09, 1.0/0.88 );		/// Bulk (approximated) Reynolds number
const double u_b        = nu*Re_b/( 2.0*delta );		    /// Bulk (approximated) velocity
const double P_0        = rho_0*u_b*u_b/( gamma_0*Ma*Ma );	/// Reference pressure
//const double T_0      = P_0/( rho_0*R_specific );	/// Reference temperature
//const double L_x      = 4.0*pi*delta;			    /// Streamwise length
//const double L_y      = 2.0*delta;				/// Wall-normal height
//const double L_z      = 4.0*pi*delta/3.0;		    /// Spanwise width
const double kappa_vK   = 0.41;                     /// von Kármán constant
const double y_0        = nu/( 9.0*u_tau );         /// Smooth-wall roughness
const double u_0        = ( u_tau/kappa_vK )*( log( delta/y_0 ) + ( y_0/delta ) - 1.0 );    /// Volume average of a log-law velocity profile
const double alpha_u    = 1.0;                      /// Magnitude of velocity perturbations
const double alpha_P    = 0.1;                      /// Magnitude of pressure perturbations

const double fixed_time_step = 5.0e-5;              /// Time step value [s]
const int cout_precision = 10;		                /// Output precision (fixed) 

#if _FEEDBACK_LOOP_BODY_FORCE_
/// Estimated uniform body force to drive the flow
double controller_output = tau_w/delta;			    /// Initialize controller output
double controller_error  = 0.0;			        	/// Initialize controller error
double controller_K_p    = 1.0e-1;		        	/// Controller proportional gain
#endif
#if _CORRECT_U_BULK_
const double u_bulk_reference = 14.647;
#endif
#if _RL_EARLY_EPISODE_TERMINATION_FUNC_U_BULK_
const double avg_u_bulk_max = u_b * (1.0 + 0.05);
const double avg_u_bulk_min = u_b * (1.0 - 0.05);
#endif


#if _ACTIVE_CONTROL_BODY_FORCE_

#ifndef RL_CASE_PATH
    #error "RL_CASE_PATH is not defined! Please define it during compilation."
#endif
const char* rl_case_path = RL_CASE_PATH;  // Use compile-time constant value

int action_dim = 3;
int state_dim  = 9;

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
/* From snapshot '3d_turbulent_channel_flow_7190000.h5, with
    Averaging Time: 399.9999999712278
    Time: 718.9999998911522
    Iteration: 7190000
    // TODO: impose y-dir symmetry in reference profile
*/
double avg_u_reference_profile[] = {   /// only inner points
    0.10342281,   0.40168769,  0.81973025,  1.35442584,  2.00144415,
    2.7538973,    3.60033596,  4.52274942,  5.49586068,  6.48885072,  7.46942506,
    8.40868533,   9.28487377, 10.08486012, 10.80351397, 11.44184742, 12.00490018,
    12.49991861, 12.93503351, 13.31840939, 13.65776883, 13.96011569, 14.23158904,
    14.47743877, 14.70209247, 14.90924081, 15.10192143, 15.28260333, 15.4532846,
    15.61555575, 15.7706748,  15.91964368, 16.06321371, 16.20197901, 16.33647276,
    16.46713437, 16.5942905,  16.71820226, 16.83910086, 16.95713312, 17.07234382,
    17.18474329, 17.29431162, 17.40099892, 17.5047136,  17.60530288, 17.70259503,
    17.79643222, 17.88668055, 17.97321066, 18.05584239, 18.13434288, 18.20845237,
    18.27787899, 18.3423134,  18.4014535,  18.45500493, 18.50267886, 18.54420882,
    18.57934041, 18.60779966, 18.62932158, 18.64371633, 18.65089467, 18.65085147,
    18.64364821, 18.62940317, 18.60827606, 18.5804413,  18.54606009, 18.50528555,
    18.45830039, 18.4053455,  18.3466972,  18.28262601, 18.21339665, 18.13928481,
    18.06056675, 17.97751204, 17.89037272, 17.79937732, 17.70476479, 17.60678733,
    17.50564977, 17.40148611, 17.29437853, 17.18437146, 17.07148554, 16.95573216,
    16.83711277, 16.7156073,  16.59112882, 16.46348781, 16.33241547, 16.1975592,
    16.05848752, 15.91467019, 15.76541351, 15.60986461, 15.44700608, 15.27557883,
    15.09398396, 14.90020641, 14.69179864, 14.46579433, 14.21856953, 13.94574492,
    13.64213092, 13.30168909, 12.91753059, 12.48201168, 11.98700292, 11.4243873,
    10.78690808, 10.0694643,   9.27093495,  8.39632609,  7.45866279,  6.47963838,
    5.48812567,   4.5164103,   3.59529876,  2.75004826,  1.99864942,  1.35253656,
    0.81858808,   0.40112858,  0.103279,
};
double rmsf_u_reference_profile[] = {   /// only inner points
    0.03693094, 0.14326817, 0.29219301, 0.48264651, 0.71264068,
    0.97745593, 1.26734189, 1.56636617, 1.8541867,  2.11059875, 2.32041365,
    2.47613894, 2.57773734, 2.6304997,  2.64248065, 2.62246018, 2.57869208,
    2.51833137, 2.4472686,  2.37015516, 2.2905061,  2.21086129, 2.13296444,
    2.05796972, 1.98664047, 1.91942904, 1.85651757, 1.79791357, 1.74350175,
    1.69308275, 1.64628986, 1.6026794,  1.56186836, 1.52342687, 1.48692708,
    1.45210389, 1.41879643, 1.38681321, 1.35589129, 1.32580306, 1.29635615,
    1.26738523, 1.23880836, 1.21057368, 1.18265406, 1.15499074, 1.12748184,
    1.10002145, 1.07250133, 1.04490422, 1.01733614, 0.9899129,  0.96276798,
    0.93606624, 0.9099773,  0.88474802, 0.86072866, 0.83828613, 0.81778075,
    0.79960902, 0.78421298, 0.77200783, 0.76332171, 0.75843629, 0.75761136,
    0.76097405, 0.7684513,  0.77978207, 0.79452384, 0.81217324, 0.83230825,
    0.85454468, 0.87844569, 0.90357488, 0.92959879, 0.95628693, 0.98347081,
    1.01104976, 1.03892535, 1.06695284, 1.09500638, 1.12302431, 1.15097771,
    1.17885936, 1.20670881, 1.23462569, 1.26277734, 1.2913267,  1.32036434,
    1.34999458, 1.3803817,  1.41170284, 1.44419979, 1.47816055, 1.51396361,
    1.55202032, 1.5926537,  1.63617732, 1.68292995, 1.73320017, 1.78724257,
    1.8452715,  1.90749809, 1.97403207, 2.04472547, 2.11912411, 2.19646985,
    2.2756058,  2.35477519, 2.43144797, 2.50215119, 2.56228877, 2.60603352,
    2.62629511, 2.61487355, 2.56300252, 2.46257889, 2.308219,   2.09987147,
    1.8449755,  1.55868732, 1.26116863, 0.97270423, 0.70917569, 0.48029967,
    0.29077458, 0.14257514, 0.03675325,
};
double rmsf_v_reference_profile[] = {   /// only inner points
    5.18759754e-05, 8.31311585e-04, 4.18915953e-03,
    1.14807155e-02, 2.40323789e-02, 4.25416480e-02, 6.72898658e-02,
    9.79216765e-02, 1.33839374e-01, 1.74141708e-01, 2.17961052e-01,
    2.64354907e-01, 3.12471430e-01, 3.61418611e-01, 4.10373071e-01,
    4.58516335e-01, 5.05131525e-01, 5.49571882e-01, 5.91314993e-01,
    6.29931370e-01, 6.65122670e-01, 6.96696258e-01, 7.24558588e-01,
    7.48687880e-01, 7.69146379e-01, 7.86055539e-01, 7.99587548e-01,
    8.09949665e-01, 8.17386627e-01, 8.22159405e-01, 8.24542156e-01,
    8.24806849e-01, 8.23213123e-01, 8.20016941e-01, 8.15455027e-01,
    8.09733911e-01, 8.03029462e-01, 7.95495630e-01, 7.87288240e-01,
    7.78535145e-01, 7.69328928e-01, 7.59727963e-01, 7.49780256e-01,
    7.39548179e-01, 7.29107979e-01, 7.18546766e-01, 7.07950725e-01,
    6.97397479e-01, 6.86958469e-01, 6.76690720e-01, 6.66645798e-01,
    6.56881260e-01, 6.47460003e-01, 6.38456710e-01, 6.29960461e-01,
    6.22062688e-01, 6.14846618e-01, 6.08381554e-01, 6.02721997e-01,
    5.97908660e-01, 5.93983851e-01, 5.90999284e-01, 5.89003629e-01,
    5.88022514e-01, 5.88049096e-01, 5.89055261e-01, 5.91008714e-01,
    5.93878170e-01, 5.97633024e-01, 6.02240641e-01, 6.07661507e-01,
    6.13848259e-01, 6.20748461e-01, 6.28306368e-01, 6.36460057e-01,
    6.45140156e-01, 6.54274280e-01, 6.63793553e-01, 6.73641881e-01,
    6.83778245e-01, 6.94158884e-01, 7.04716288e-01, 7.15355735e-01,
    7.25975079e-01, 7.36489266e-01, 7.46833582e-01, 7.56940600e-01,
    7.66727906e-01, 7.76106814e-01, 7.84990253e-01, 7.93273763e-01,
    8.00835044e-01, 8.07522556e-01, 8.13164492e-01, 8.17589662e-01,
    8.20619850e-01, 8.22054487e-01, 8.21669180e-01, 8.19217636e-01,
    8.14445048e-01, 8.07085318e-01, 7.96876769e-01, 7.83566967e-01,
    7.66927950e-01, 7.46767107e-01, 7.22954530e-01, 6.95420067e-01,
    6.64163920e-01, 6.29263447e-01, 5.90905478e-01, 5.49375603e-01,
    5.05090142e-01, 4.58571336e-01, 4.10476056e-01, 3.61537044e-01,
    3.12591057e-01, 2.64474192e-01, 2.18081051e-01, 1.74259317e-01,
    1.33947976e-01, 9.80146802e-02, 6.73630941e-02, 4.25937304e-02,
    2.40649619e-02, 1.14976681e-02, 4.19581309e-03, 8.32698233e-04,
    5.19548806e-05,
};
double rmsf_w_reference_profile[] = {   /// only inner points
    0.01946789, 0.07229103, 0.14033935, 0.21832706, 0.30104129,
    0.3838422,  0.46325556, 0.53721453, 0.60514955, 0.66743757, 0.72480219,
    0.7776623,  0.82599797, 0.86942278, 0.90752461, 0.9400955,  0.96729639,
    0.98960046, 1.0076801,  1.02226834, 1.03402857, 1.04344423, 1.05080993,
    1.05628562, 1.05993981, 1.06176311, 1.0617133,  1.05974414, 1.05583026,
    1.05004144, 1.0425474,  1.03349154, 1.02297942, 1.01115456, 0.9982023,
    0.98435078, 0.96982206, 0.95476498, 0.93924679, 0.92329678, 0.90696629,
    0.89033091, 0.87341234, 0.85619793, 0.83864515, 0.82074452, 0.8026018,
    0.78436097, 0.76613729, 0.74805352, 0.73027436, 0.7129664,  0.69622167,
    0.680082,   0.6645955,  0.64989096, 0.63619692, 0.62373997, 0.61268514,
    0.60319373, 0.59541486, 0.58945866, 0.58537586, 0.5831625,  0.58284267,
    0.58449543, 0.58815707, 0.59376503, 0.60118433, 0.61025791, 0.62081998,
    0.63271464, 0.64580214, 0.65992874, 0.67497796, 0.69087725, 0.70752137,
    0.72474942, 0.74235039, 0.76014619, 0.77802368, 0.79589122, 0.81371061,
    0.83153037, 0.84937254, 0.86713998, 0.88465936, 0.90181065, 0.91856671,
    0.93489396, 0.9507282,  0.96597548, 0.98051576, 0.99420537, 1.00688133,
    1.01844689, 1.02888551, 1.03812446, 1.0459531,  1.05212649, 1.05648464,
    1.05893597, 1.05945012, 1.058045,   1.05476706, 1.04961952, 1.04252216,
    1.03327339, 1.02149502, 1.00667898, 0.98824349, 0.96557201, 0.93807238,
    0.90531925, 0.86716573, 0.82380391, 0.77562219, 0.72298129, 0.66587084,
    0.60384213, 0.5361513,  0.4624148,  0.3832028,  0.30058204, 0.21802315,
    0.14016156, 0.07220751, 0.01944749,
};
#endif  /// of _RL_CONTROL_IS_SUPERVISED_
#endif  /// of _ACTIVE_CONTROL_BODY_FORCE_

////////// myRHEA CLASS //////////

myRHEA::myRHEA(const string name_configuration_file, const string tag, const string restart_data_file, const string t_action, const string t_episode, const string t_begin_control, const string db_clustered, const string global_step) : FlowSolverRHEA(name_configuration_file) {

#if _ACTIVE_CONTROL_BODY_FORCE_
    rl_f_rhou_field.setTopology(topo, "rl_f_rhou_field");
    rl_f_rhov_field.setTopology(topo, "rl_f_rhov_field");
    rl_f_rhow_field.setTopology(topo, "rl_f_rhow_field");
#if _TEMPORAL_SMOOTHING_RL_ACTION_
    rl_f_rhou_field_prev_step.setTopology(topo, "rl_f_rhou_field_prev_step");
    rl_f_rhov_field_prev_step.setTopology(topo, "rl_f_rhov_field_prev_step");
    rl_f_rhow_field_prev_step.setTopology(topo, "rl_f_rhow_field_prev_step");
    rl_f_rhou_field_curr_step.setTopology(topo, "rl_f_rhou_field_curr_step");
    rl_f_rhov_field_curr_step.setTopology(topo, "rl_f_rhov_field_curr_step");
    rl_f_rhow_field_curr_step.setTopology(topo, "rl_f_rhow_field_curr_step");
#endif  /// of _TEMPORAL_SMOOTHING_RL_ACTION_
    d_DeltaRxx_x_field.setTopology(topo, "d_DeltaRxx_x"); 
    d_DeltaRxy_x_field.setTopology(topo, "d_DeltaRxy_x"); 
    d_DeltaRxz_x_field.setTopology(topo, "d_DeltaRxz_x"); 
    d_DeltaRxy_y_field.setTopology(topo, "d_DeltaRxy_y"); 
    d_DeltaRyy_y_field.setTopology(topo, "d_DeltaRyy_y"); 
    d_DeltaRyz_y_field.setTopology(topo, "d_DeltaRyz_y"); 
    d_DeltaRxz_z_field.setTopology(topo, "d_DeltaRxz_z"); 
    d_DeltaRyz_z_field.setTopology(topo, "d_DeltaRyz_z"); 
    d_DeltaRzz_z_field.setTopology(topo, "d_DeltaRzz_z"); 
    d_DeltaRxj_j_field.setTopology(topo, "d_DeltaRxj_j");
    d_DeltaRyj_j_field.setTopology(topo, "d_DeltaRyj_j");
    d_DeltaRzj_j_field.setTopology(topo, "d_DeltaRzj_j");
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
#endif  /// of _RL_CONTROL_IS_SUPERVISED_

    /// Add fields to write in .h5 and .xdmf files 
    /// TODO: cause execution error during initializeFromRestart because these fields not in restart file
    /// TODO: these lines only for debugging, remove to avoid execution errors
    writer_reader->addField(&d_DeltaRxx_x_field);
    writer_reader->addField(&d_DeltaRxy_x_field);
    writer_reader->addField(&d_DeltaRxz_x_field);
    writer_reader->addField(&d_DeltaRxy_y_field);
    writer_reader->addField(&d_DeltaRyy_y_field);
    writer_reader->addField(&d_DeltaRyz_y_field);
    writer_reader->addField(&d_DeltaRxz_z_field);
    writer_reader->addField(&d_DeltaRyz_z_field);
    writer_reader->addField(&d_DeltaRzz_z_field);
    writer_reader->addField(&d_DeltaRxj_j_field);
    writer_reader->addField(&d_DeltaRyj_j_field);
    writer_reader->addField(&d_DeltaRzj_j_field);

    initRLParams(tag, restart_data_file, t_action, t_episode, t_begin_control, db_clustered, global_step);
    initSmartRedis();
#endif  /// of _ACTIVE_CONTROL_BODY_FORCE_

};


void myRHEA::execute() {
    
    /// Start timer: execute
    timers->start( "execute" );

    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /// Set output (cout) precision
    cout.precision( cout_precision );

    /// Start RHEA simulation
    if( my_rank == 0 ) cout << endl << "RHEA (v" << version_number << "): START SIMULATION" << endl;

    /// Initialize variables from restart file or by setting initial conditions
    if( use_restart ) {

        /// Initialize from restart file
        this->initializeFromRestart();

        if( artificial_compressibility_method ) {

            /// Calculate thermodynamic (bulk) pressure
            P_thermo = this->calculateVolumeAveragedPressure();

            /// Calculate alpha value of artificial compressibility method
            alpha_acm = this->calculateAlphaArtificialCompressibilityMethod();

	        /// Calculate artificially modified thermodynamics
            this->calculateArtificiallyModifiedThermodynamics();

	        /// Calculate artificially modified transport coefficients
            this->calculateArtificiallyModifiedTransportCoefficients();

	    }

    } else {

        /// Set initial conditions
        this->setInitialConditions();

        /// Initialize thermodynamics
        this->initializeThermodynamics();

        if( artificial_compressibility_method ) {

            /// Calculate thermodynamic (bulk) pressure
            P_thermo = this->calculateVolumeAveragedPressure();

            /// Calculate alpha value of artificial compressibility method
            alpha_acm = this->calculateAlphaArtificialCompressibilityMethod();

	    /// Calculate artificially modified thermodynamics
            this->calculateArtificiallyModifiedThermodynamics();	    

	    /// Calculate artificially modified transport coefficients
            this->calculateArtificiallyModifiedTransportCoefficients();

	    } else {

            /// Calculate transport coefficients
            this->calculateTransportCoefficients();

	    }

    }

    /// Calculate conserved variables from primitive variables
    this->primitiveToConservedVariables();

    /// Update previous state of conserved variables
    this->updatePreviousStateConservedVariables();    
    
    /// Start timer: time_iteration_loop
    timers->start( "time_iteration_loop" );

    /// Iterate flow solver RHEA in time
    for(int time_iter = current_time_iter; time_iter < final_time_iter; time_iter++) {

        /// Start timer: calculate_time_step
        timers->start( "calculate_time_step" );

        /// Calculate time step
        this->calculateTimeStep();
        if( ( current_time + delta_t ) > final_time ) delta_t = final_time - current_time;

        /// Stop timer: calculate_time_step
        timers->stop( "calculate_time_step" );

        /// Stop timer: execute
        timers->stop( "execute" );

        /// Start timer: output_solver_state
        timers->start( "output_solver_state" );

        /// Print time iteration information (if criterion satisfied)
        if( ( current_time_iter%print_frequency_iter == 0 ) and ( my_rank == 0 ) ) {
            cout << endl << "Time iteration " << current_time_iter << ": " 
                 << "time = " << scientific << current_time << " [s], "
                 << "time-step = " << scientific << delta_t << " [s], "
                 << "wall-clock time = " << scientific << timers->getAccumulatedMaxTime( "execute" )/3600.0 << " [h]" << endl;
        }

        /// Output current state data to file (if criterion satisfied)
        if( current_time_iter%output_frequency_iter == 0 ) this->outputCurrentStateData();

        /// Output current 2d slices state data to file (if criterion satisfied)
        this->output2dSlicesCurrentStateData();
        
        /// Output temporal point probes data to files (if criterion satisfied)
        this->outputTemporalPointProbesData();

        /// Stop timer: output_solver_state
        timers->stop( "output_solver_state" );

        /// Start timer: execute
        timers->start( "execute" );

#if _CORRECT_U_BULK_ || _RL_EARLY_EPISODE_TERMINATION_FUNC_U_BULK_
        /// Correct streamwise bulk velocity (with u_field) and/or check RL early episode termination (with avg_u_field)
        this->manageStreamwiseBulkVelocity();
#endif

        /// Start timer: rk_iteration_loop
        timers->start( "rk_iteration_loop" );

        /// Runge-Kutta time-integration steps
        for(rk_time_stage = 1; rk_time_stage <= rk_number_stages; rk_time_stage++) {

            /// Start timer: calculate_thermophysical_properties
            timers->start( "calculate_thermophysical_properties" );

            if( artificial_compressibility_method ) {

	        /// Calculate artificially modified transport coefficients
                this->calculateArtificiallyModifiedTransportCoefficients();

	        } else {

            /// Calculate transport coefficients
            this->calculateTransportCoefficients();

	        }

            /// Stop timer: calculate_thermophysical_properties
            timers->stop( "calculate_thermophysical_properties" );

            /// Start timer: calculate_inviscid_fluxes
            timers->start( "calculate_inviscid_fluxes" );

            /// Calculate inviscid fluxes
            this->calculateInviscidFluxes();

            /// Stop timer: calculate_inviscid_fluxes
            timers->stop( "calculate_inviscid_fluxes" );

            /// Start timer: calculate_viscous_fluxes
            timers->start( "calculate_viscous_fluxes" );

            /// Calculate viscous fluxes
            this->calculateViscousFluxes();

            /// Stop timer: calculate_viscous_fluxes
            timers->stop( "calculate_viscous_fluxes" );

            /// Start timer: calculate_source_terms
            timers->start( "calculate_source_terms" );

            /// Calculate source terms
            this->calculateSourceTerms();

            /// Stop timer: calculate_source_terms
            timers->stop( "calculate_source_terms" );

            /// Start timer: time_advance_conserved_variables
            timers->start( "time_advance_conserved_variables" );

            /// Advance conserved variables in time
            this->timeAdvanceConservedVariables();

            /// Stop timer: time_advance_conserved_variables
            timers->stop( "time_advance_conserved_variables" );

            /// Start timer: conserved_to_primitive_variables
            timers->start( "conserved_to_primitive_variables" );

            /// Calculate primitive variables from conserved variables
            this->conservedToPrimitiveVariables();

            /// Stop timer: conserved_to_primitive_variables
            timers->stop( "conserved_to_primitive_variables" );

            /// Start timer: calculate_thermodynamics_from_primitive_variables
            timers->start( "calculate_thermodynamics_from_primitive_variables" );

            /// Calculate thermodynamics from primitive variables
            this->calculateThermodynamicsFromPrimitiveVariables();

            if( artificial_compressibility_method ) {

                /// Calculate thermodynamic (bulk) pressure
                P_thermo = this->calculateVolumeAveragedPressure();

                /// Calculate alpha value of artificial compressibility method
                alpha_acm = this->calculateAlphaArtificialCompressibilityMethod();

                /// Calculate artificially modified thermodynamics
                this->calculateArtificiallyModifiedThermodynamics();	    

            }

            /// Stop timer: calculate_thermodynamics_from_primitive_variables
            timers->stop( "calculate_thermodynamics_from_primitive_variables" );

            /// Start timer: update_boundaries
            timers->start( "update_boundaries" );

            /// Update boundary values
            this->updateBoundaries();
            
            /// Stop timer: update_boundaries
            timers->stop( "update_boundaries" );

        }

        /// Stop timer: rk_iteration_loop
        timers->stop( "rk_iteration_loop" );

        /// Start timer: update_time_averaged_quantities
        timers->start( "update_time_averaged_quantities" );

        /// Update time-averaged quantities
        if( time_averaging_active ) this->updateTimeAveragedQuantities();

        /// Stop timer: update_time_averaged_quantities
        timers->stop( "update_time_averaged_quantities" );

        /// Start timer: temporal_hook_function
        timers->start( "temporal_hook_function" );

        /// Temporal hook function
        this->temporalHookFunction();

        /// Stop timer: temporal_hook_function
        timers->stop( "temporal_hook_function" );

        /// Start timer: update_previous_state_conserved_variables
        timers->start( "update_previous_state_conserved_variables" );

        /// Update previous state of conserved variables
        this->updatePreviousStateConservedVariables();

        /// Update time and time iteration
        current_time += delta_t;
        current_time_iter += 1;

        /// Check if simulation is completed: current_time > final_time
        if( current_time >= final_time ) break;

        /// Stop timer: update_previous_state_conserved_variables
        timers->stop( "update_previous_state_conserved_variables" );

    }

    /// Stop timer: time_iteration_loop
    timers->stop( "time_iteration_loop" );

    /// Print timers information
    if( print_timers ) timers->printTimers( timers_information_file );

    /// Print time advancement information
    if( my_rank == 0 ) {
        cout << "Time advancement completed -> " 
            << "iteration = " << current_time_iter << ", "
            << "time = " << scientific << current_time << " [s]" << endl;
    }

    /// Output current state data to file
    this->outputCurrentStateData();

    /// Output current 2d slices state data to file (if criterion satisfied)
    this->output2dSlicesCurrentStateData();
        
    /// Output temporal point probes data to files (if criterion satisfied)
    this->outputTemporalPointProbesData();

    /// End RHEA simulation
    if( my_rank == 0 ) cout << endl << "RHEA (v" << version_number << "): END SIMULATION" << endl;
    
    /// Stop timer: execute
    timers->stop( "execute" );

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
        this->episode_period          = std::stod(t_episode);
        this->begin_actuation_time    = std::stod(t_begin_control);
        this->previous_actuation_time = 0.0;
        this->initial_episode_time    = 0.0;
        this->final_time              = std::stod(t_episode);        // updated variable from previously defined value in FlowSolverRHEA::readConfigurationFile
        if (my_rank == 0) {
            cout << "[myRHEA::initRLParams] " 
                 << "actuation_period = " << scientific << this->actuation_period
                 << ", episode_period = " << scientific << this->episode_period
                 << ", begin_actuation_time = " << scientific << this->begin_actuation_time
                 << ", previous_actuation_time = " << scientific << this->previous_actuation_time
                 << ", initial_episode_time = " << scientific << this->initial_episode_time
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

    /// Additional arguments, defined here
    this->witness_file = string(rl_case_path) + "/config_control_witness/witnessPoints" + to_string(np_x*np_y*np_z) + ".txt";
    this->control_file = string(rl_case_path) + "/config_control_witness/controlPoints" + to_string(np_x*np_y*np_z) + ".txt";
    this->time_key      = "ensemble_" + tag + ".time";
    this->step_type_key = "ensemble_" + tag + ".step_type";
    this->state_key     = "ensemble_" + tag + ".state";
    this->action_key    = "ensemble_" + tag + ".action";
    this->reward_key    = "ensemble_" + tag + ".reward";

    /// Witness points
    readWitnessPoints(); 
    preproceWitnessPoints();        // updates attribute 'state_local_size2'

    /// Control cubic regions
    readControlPoints();
    preproceControlPoints();        // updates 'action_global_size2', 'n_rl_envs'
    
    /// Allocate action data
    /// Annotation: State is stored in arrays of different sizes on each MPI rank.
    ///             Actions is a global array living in all processes.
    action_global.resize(action_global_size2);
    state_local.resize(state_local_size2);
    std::fill(action_global.begin(), action_global.end(), 0.0);
    std::fill(state_local.begin(), state_local.end(), 0.0);

    /// Reward auxiliary variables and fields
#if _RL_CONTROL_IS_SUPERVISED_
    /// -------------- Build rmsf_u,v,w_reference_field --------------
    /// Accessing the global domain decomposition data
    int global_startY, global_j;
    int globalNy    = topo->getMesh()->getGNy();    // Total number of y-cells in the global domain
    int localNy     = topo->getlNy()-2;             // Local number of y-cells for this process (-2 for inner points only)
    int divy        = globalNy % np_y;              // Remainder for non-uniform decomposition
    /// Calculate the rank's position in the y-dimension based on rank and grid layout
    int plane_rank = my_rank % (np_x * np_y);
    int rank_in_y  = plane_rank / np_x;             // Rank's position in the y-dimension (row in the global grid)
    /// Calculate the global start index for this rank's local slice in the y-dimension
    if (rank_in_y < divy) {
        global_startY = rank_in_y * (localNy + 1);  // Extra cell for some ranks
    } else {
        global_startY = rank_in_y * localNy + divy; // Regular distribution for remaining ranks
    }
    /// Fill global profile data into local field data
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
        /// cout << "[Rank " << my_rank << "] j=" << j << " (local), global_j=" << global_j 
        ///      << ", Reference value of avg_u: " << avg_u_reference_profile[global_j]
        ///      << ", rmsf_u: " << rmsf_u_reference_profile[global_j]
        ///      << ", rmsf_v: " << rmsf_v_reference_profile[global_j]
        ///      << ", rmsf_w: " << rmsf_w_reference_profile[global_j] 
        ///      << endl;
    }
#else   /// _RL_CONTROL_IS_SUPERVISED_ 0
    avg_u_previous_field   = 0.0;   /// DistributedArray, 0.0 everywhere
    rmsf_u_previous_field  = 0.0;
    rmsf_v_previous_field  = 0.0;
    rmsf_w_previous_field  = 0.0;
#endif  /// _RL_CONTROL_IS_SUPERVISED_

    /// Initialize additional attribute members
#if _RL_CONTROL_IS_SUPERVISED_
    first_actuation_time_done   = false;
    first_actuation_period_done = false;
#else
    first_actuation_time_done   = false;
    first_actuation_period_done = false;
#endif
    last_communication          = false;

#if _RL_EARLY_EPISODE_TERMINATION_FUNC_U_BULK_
    rl_early_episode_termination = false;
#endif

};


/// Based on subroutine mod_smartredis::init_smartredis
void myRHEA::initSmartRedis() {
    
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


void myRHEA::initializeFromRestart() {
    
    // --------  Main code, adapted from parent class FlowSolverRHEA to initialize averaging fields using instantaneous data -------- 

    /// Read from file to restart solver: data, time and time iteration
    char char_restart_data_file[ restart_data_file.length() + 1 ]; 
    strcpy( char_restart_data_file, restart_data_file.c_str() );
    writer_reader->read( char_restart_data_file );
    current_time      = writer_reader->getAttributeDouble( "Time" );
    current_time_iter = writer_reader->getAttributeInt( "Iteration" );
    averaging_time    = writer_reader->getAttributeDouble( "AveragingTime" );
    
    if( reset_time_averaging ) {

	    /// Reset time averaging
        averaging_time = 0.0;

	    /// Initialize avg fields with instantaneous data
        for(int i = topo->iter_common[_ALL_][_INIX_]; i <= topo->iter_common[_ALL_][_ENDX_]; i++) {
            for(int j = topo->iter_common[_ALL_][_INIY_]; j <= topo->iter_common[_ALL_][_ENDY_]; j++) {
                for(int k = topo->iter_common[_ALL_][_INIZ_]; k <= topo->iter_common[_ALL_][_ENDZ_]; k++) {
                    avg_rho_field[I1D(i,j,k)]   = rho_field[I1D(i,j,k)];
                    avg_rhou_field[I1D(i,j,k)]  = rhou_field[I1D(i,j,k)];
                    avg_rhov_field[I1D(i,j,k)]  = rhov_field[I1D(i,j,k)];
                    avg_rhow_field[I1D(i,j,k)]  = rhow_field[I1D(i,j,k)];
                    avg_rhoE_field[I1D(i,j,k)]  = rhoE_field[I1D(i,j,k)];
                    avg_rhoP_field[I1D(i,j,k)]  = rho_field[I1D(i,j,k)]*P_field[I1D(i,j,k)];
                    avg_rhoT_field[I1D(i,j,k)]  = rho_field[I1D(i,j,k)]*T_field[I1D(i,j,k)];
                    avg_u_field[I1D(i,j,k)]     = u_field[I1D(i,j,k)];
                    avg_v_field[I1D(i,j,k)]     = v_field[I1D(i,j,k)];
                    avg_w_field[I1D(i,j,k)]     = w_field[I1D(i,j,k)];
                    avg_E_field[I1D(i,j,k)]     = E_field[I1D(i,j,k)];
                    avg_P_field[I1D(i,j,k)]     = P_field[I1D(i,j,k)];
                    avg_T_field[I1D(i,j,k)]     = T_field[I1D(i,j,k)];
                    avg_sos_field[I1D(i,j,k)]   = sos_field[I1D(i,j,k)];
                    avg_mu_field[I1D(i,j,k)]    = mu_field[I1D(i,j,k)];
                    avg_kappa_field[I1D(i,j,k)] = kappa_field[I1D(i,j,k)];
                    avg_c_v_field[I1D(i,j,k)]   = c_v_field[I1D(i,j,k)];
                    avg_c_p_field[I1D(i,j,k)]   = c_p_field[I1D(i,j,k)];
                }
            }
        }

        /// Reset fluctuating fields
        rmsf_rho_field     = 0.0;
        rmsf_rhou_field    = 0.0;
        rmsf_rhov_field    = 0.0;
        rmsf_rhow_field    = 0.0;
        rmsf_rhoE_field    = 0.0;
        rmsf_u_field       = 0.0;
        rmsf_v_field       = 0.0;
        rmsf_w_field       = 0.0;
        rmsf_E_field       = 0.0;
        rmsf_P_field       = 0.0;
        rmsf_T_field       = 0.0;
        rmsf_sos_field     = 0.0;
        rmsf_mu_field      = 0.0;
        rmsf_kappa_field   = 0.0;
        rmsf_c_v_field     = 0.0;
        rmsf_c_p_field     = 0.0;
        favre_uffuff_field = 0.0;
        favre_uffvff_field = 0.0;
        favre_uffwff_field = 0.0;
        favre_vffvff_field = 0.0;
        favre_vffwff_field = 0.0;
        favre_wffwff_field = 0.0;
        favre_uffEff_field = 0.0;
        favre_vffEff_field = 0.0;
        favre_wffEff_field = 0.0;

    }

    /// Update halo values
    rho_field.update();
    u_field.update();
    v_field.update();
    w_field.update();
    E_field.update();
    P_field.update();
    T_field.update();
    sos_field.update();
    mu_field.update();
    kappa_field.update();
    c_v_field.update();
    c_p_field.update();
    avg_rho_field.update();
    avg_rhou_field.update();
    avg_rhov_field.update();
    avg_rhow_field.update();
    avg_rhoE_field.update();
    avg_rhoP_field.update();
    avg_rhoT_field.update();
    avg_u_field.update();
    avg_v_field.update();
    avg_w_field.update();
    avg_E_field.update();
    avg_P_field.update();
    avg_T_field.update();
    avg_sos_field.update();
    avg_mu_field.update();
    avg_kappa_field.update();
    avg_c_v_field.update();
    avg_c_p_field.update();
    rmsf_rho_field.update();
    rmsf_rhou_field.update();
    rmsf_rhov_field.update();
    rmsf_rhow_field.update();
    rmsf_rhoE_field.update();
    rmsf_u_field.update();
    rmsf_v_field.update();
    rmsf_w_field.update();
    rmsf_E_field.update();
    rmsf_P_field.update();
    rmsf_T_field.update();
    rmsf_sos_field.update();
    rmsf_mu_field.update();
    rmsf_kappa_field.update();
    rmsf_c_v_field.update();
    rmsf_c_p_field.update();
    favre_uffuff_field.update();
    favre_uffvff_field.update();
    favre_uffwff_field.update();
    favre_vffvff_field.update();
    favre_vffwff_field.update();
    favre_wffwff_field.update();
    favre_uffEff_field.update();
    favre_vffEff_field.update();
    favre_wffEff_field.update();

    /// Fill mesh x, y, z, delta_x, delta_y, delta_z fields
    this->fillMeshCoordinatesSizesFields();

#if _ACTIVE_CONTROL_BODY_FORCE_
    // -------- Add additional functionality -------- 
    begin_actuation_time    += current_time;
    previous_actuation_time += current_time;
    initial_episode_time    += current_time;
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
                this->calculateStateAndReward();
                first_actuation_time_done = true;
                if (my_rank == 0) {
                    cout << endl << endl << "[myRHEA::calculateSourceTerms] Initializing auxiliary variables for state and reward calculations" << endl;
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
                    this->calculateStateAndReward();
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
                    if (last_communication) {
                        if (my_rank == 0) cout << "[myRHEA::calculateSourceTerms] Last smart redis communication at time: " << current_time << ", iteration: " << current_time_iter << endl;
                        if (tag == "0") this->outputCurrentStateDataRL(); /// Save state before episode termination
                    }
                    this->calculateStateAndReward();
                    manager->writeState(state_local, state_key);
                    manager->writeReward(reward_local, reward_key);
                    manager->writeTime(current_time, time_key);
                    // Reading new action...
                    manager->readAction(action_key);
                    action_global = manager->getActionGlobal();     /// action_global: vector<double> of size action_global_size2 = action_dim * n_rl_envs (currently only 1 action variable per rl env.)
                    /// action_local  = manager->getActionLocal();  /// action_local not used
                    /// Update & Write step size (from 1) to 0 if the next time that we require actuation value is the last one
                    if ( current_time + 2.0 * ( actuation_period + delta_t ) >= final_time ) {
                        if (my_rank == 0) cout << "[myRHEA::calculateSourceTerms] Set RL Step '0' to terminate episode at time: " << current_time << ", iteration: " << current_time_iter << endl;
                        manager->writeStepType(0, step_type_key);
                        last_communication = true;
                    }

                    MPI_Barrier(MPI_COMM_WORLD);
                    timers->stop( "rl_smartredis_communications" );

                    /// -------------------------------------------------------------------
                    ///             CALCULATE F_pert(x,t_p) FROM ACTION a_k^p
                    /// -------------------------------------------------------------------

                    MPI_Barrier(MPI_COMM_WORLD);
                    timers->start( "rl_update_DeltaRij" );

                    /// Initialize variables
                    int i,j,k;
                    int num_control_probes_local_counter = 0; 
                    double Rkk = 0.0, phi1 = 0.0, phi2 = 0.0, phi3 = 0.0, xmap1 = 0.0, xmap2 = 0.0;
                    double DeltaRkk = 0.0, DeltaPhi1 = 0.0, DeltaPhi2 = 0.0, DeltaPhi3 = 0.0, DeltaXmap1 = 0.0, DeltaXmap2 = 0.0; 
                    double DeltaRxx, DeltaRxy, DeltaRxz, DeltaRyy, DeltaRyz, DeltaRzz;
                    bool   isNegligibleAction, isNegligibleRkk;
                    size_t actuation_idx;
                    double Rkk_inv, Akk;
                    vector<double> tcp_position(3, 0.0);
                    vector<double> tcp_DeltaRij(6, 0.0);
                    vector<vector<double>> Aij(3, vector<double>(3, 0.0));
                    vector<vector<double>> Dij(3, vector<double>(3, 0.0));
                    vector<vector<double>> Qij(3, vector<double>(3, 0.0));                
                    vector<vector<double>> DijPert(3, vector<double>(3, 0.0));
                    vector<vector<double>> QijPert(3, vector<double>(3, 0.0));
                    vector<vector<double>> RijPert(3, vector<double>(3, 0.0));
                    
                    /// --------- Calculate DeltaRij = Rij_perturbated - Rij_original ----------
                    for(int tcp = 0; tcp < num_control_probes; ++tcp) {
                        if( temporal_control_probes[tcp].getGlobalOwnerRank() == my_rank ) {
                            
                            num_control_probes_local_counter++;

                            /// Probe local indexes
                            i = temporal_control_probes[tcp].getLocalIndexI(); 
                            j = temporal_control_probes[tcp].getLocalIndexJ(); 
                            k = temporal_control_probes[tcp].getLocalIndexK();
                            tcp_position = {x_field[I1D(i,j,k)], y_field[I1D(i,j,k)], z_field[I1D(i,j,k)]};
                            
                            /// Get perturbation values from RL agent
                            actuation_idx = static_cast<size_t>(tcp);   /// type size_t
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
                                DeltaRkk   = action_global[actuation_idx * action_dim + 0];
                                DeltaPhi1  = 0.0;
                                DeltaPhi2  = 0.0;
                                DeltaPhi3  = 0.0;
                                DeltaXmap1 = action_global[actuation_idx * action_dim + 1];
                                DeltaXmap2 = action_global[actuation_idx * action_dim + 2];
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
                                    
                            /// ------- Calculate DeltaRij_field from DeltaRij d.o.f. (action), if action is not negligible ---------
                            /// Rij trace (dof #1): Rkk
                            Rkk                = favre_uffuff_field[I1D(i,j,k)] + favre_vffvff_field[I1D(i,j,k)] + favre_wffwff_field[I1D(i,j,k)];
                            isNegligibleAction = (abs(DeltaRkk) < EPS && abs(DeltaPhi1) < EPS && abs(DeltaPhi2) < EPS && abs(DeltaPhi3) < EPS && abs(DeltaXmap1) < EPS && abs(DeltaXmap2) < EPS);
                            isNegligibleRkk    = (abs(Rkk) < EPS);
                            if (isNegligibleAction || isNegligibleRkk) {
                                DeltaRxx = 0.0;
                                DeltaRxy = 0.0;
                                DeltaRxz = 0.0;
                                DeltaRyy = 0.0;
                                DeltaRyz = 0.0;
                                DeltaRzz = 0.0;
                            } else {
                                /// -------- Perform Rij eigen-decomposition --------
                                /// Anisotropy tensor (symmetric, trace-free)
                                Rkk_inv = 1.0 / Rkk;
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
                                symmetricDiagonalize(Aij, Qij, Dij);                   // update Qij, Dij
                                sortEigenDecomposition(Qij, Dij);                      // update Qij, Dij s.t. eigenvalues in decreasing order

                                /// Eigen-vectors Euler ZXZ rotation angles (dof #2-4): phi1, phi2, phi3
                                eigVect2eulerAngles(Qij, phi1, phi2, phi3);             // update phi1, phi2, phi3

                                /// Eigen-values Barycentric coordinates (dof #5-6): xmap1, xmap2
                                eigValMatrix2barycentricCoord(Dij, xmap1, xmap2);       // update xmap1, xmap2

                                /// Build perturbed Rij d.o.f. -> x_new = x_old + Delta_x * x_old
                                /// Delta_* are standarized values between 'action_bounds' RL parameter > normalize actions to desired action bounds
                                Rkk   += DeltaRkk   * 20.0;
                                phi1  += DeltaPhi1  * M_PI;         // phi1 range: [-pi,pi]
                                phi2  += DeltaPhi2  * M_PI / 2.0;   // phi2 range: [0,pi]
                                phi3  += DeltaPhi3  * M_PI;         // phi3 range: [-pi,pi]
                                xmap1 += DeltaXmap1 * 5.0;          // xmap1 range: [0,1]
                                xmap2 += DeltaXmap2 * 5.0;          // xmap2 range: [0,1]

                                /// Enforce realizability to perturbed Rij d.o.f (update {Rkk,phi1,phi2,phi3,xmap1,xmap2}_field, if necessary)
                                enforceRealizability(Rkk, phi1, phi2, phi3, xmap1, xmap2);

                                /// Calculate perturbed & realizable Rij
                                eulerAngles2eigVect(phi1, phi2, phi3, QijPert);         // update QijPert
                                barycentricCoord2eigValMatrix(xmap1, xmap2, DijPert);   // update DijPert
                                sortEigenDecomposition(QijPert, DijPert);               // update QijPert & DijPert, if necessary
                                Rijdof2matrix(Rkk, DijPert, QijPert, RijPert);          // update RijPert

                                /// Calculate perturbed & realizable DeltaRij
                                DeltaRxx = RijPert[0][0] - favre_uffuff_field[I1D(i,j,k)];      // local values at control probe, once per mpi process / rl environment
                                DeltaRxy = RijPert[0][1] - favre_uffvff_field[I1D(i,j,k)];
                                DeltaRxz = RijPert[0][2] - favre_uffwff_field[I1D(i,j,k)];
                                DeltaRyy = RijPert[1][1] - favre_vffvff_field[I1D(i,j,k)];
                                DeltaRyz = RijPert[1][2] - favre_vffwff_field[I1D(i,j,k)];
                                DeltaRzz = RijPert[2][2] - favre_wffwff_field[I1D(i,j,k)];
                                tcp_DeltaRij = {DeltaRxx, DeltaRxy, DeltaRxz, DeltaRyy, DeltaRyz, DeltaRzz};

                                /// Debug control probe position & DeltaRij
                                cout << "[calculateSourceTerms] [Rank " << my_rank << "] Control probe (" << tcp_position[0] << ", " << tcp_position[1] << ", " << tcp_position[2] << "): "
                                     << "DeltaRij = (" << tcp_DeltaRij[0] << ", " << tcp_DeltaRij[1] << ", " << tcp_DeltaRij[2] << ", " << tcp_DeltaRij[3] << ", " << tcp_DeltaRij[4] << ", " << tcp_DeltaRij[5] << ")" << endl;
                            }
                        }
                    }

                    /// Check only 1 control probe per mpi process; if > 1 num. control probes found per mpi process then DeltaRij is updated more than once
                    if (num_control_probes_local_counter != 1){
                        cerr << "[calculateSourceTerms] ERROR: Rank " << my_rank << " has num. control probes: " << num_control_probes_local_counter << " != 1" << endl;
                        MPI_Abort( MPI_COMM_WORLD, 1);
                    }

                    MPI_Barrier(MPI_COMM_WORLD);
                    timers->stop( "rl_update_DeltaRij" );

                    MPI_Barrier(MPI_COMM_WORLD);
                    timers->start( "rl_update_control_term" );
                    
                    /// Calculate partial derivatives DeltaRij_j at (fine) mesh points from action-based DeltaRij values at (coarse) TCP 3-D mesh
                    /// > update d_DeltaRxx_x_field, d_DeltaRxy_x_field, d_DeltaRxz_x_field, 
                    ///          d_DeltaRxy_y_field, d_DeltaRyy_y_field, d_DeltaRyz_y_field, 
                    ///          d_DeltaRxz_z_field, d_DeltaRyz_z_field, d_DeltaRzz_z_field,
                    ///          d_DeltaRxj_j_field, d_DeltaRyj_j_field, d_DeltaRzj_j_field.
                    calculate_d_DeltaRij_j(tcp_position, tcp_DeltaRij);

                    /// Calculate and incorporate perturbation load F = \partial DeltaRij / \partial xj
                    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {                             

                                /// Apply perturbation load (\partial DeltaRij / \partial xj) into ui momentum equation
                                rl_f_rhou_field[I1D(i,j,k)] = ( -1.0 ) * rho_field[I1D(i,j,k)] * ( d_DeltaRxj_j_field[I1D(i,j,k)] );
                                rl_f_rhov_field[I1D(i,j,k)] = ( -1.0 ) * rho_field[I1D(i,j,k)] * ( d_DeltaRyj_j_field[I1D(i,j,k)] );
                                rl_f_rhow_field[I1D(i,j,k)] = ( -1.0 ) * rho_field[I1D(i,j,k)] * ( d_DeltaRzj_j_field[I1D(i,j,k)] );
                            }
                        }
                    }

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

    /// Save data only for ensemble #0 due to memory limitations
    if (tag == "0") {
        /// Print timers information
        if ( ( print_timers ) && (current_time_iter%print_frequency_iter == 0) ) {
            char filename_timers[1024];
            sprintf( filename_timers, "%s/rhea_exp/timers_info/timers_information_file_%d_ensemble%s_step%s.txt", 
                     rl_case_path, current_time_iter, tag.c_str(), global_step.c_str() );
            timers->printTimers( filename_timers );
        }
        /// Output current state in RL dedicated directory
        if ( current_time_iter%print_frequency_iter == 0 ) {
            this->outputCurrentStateDataRL();
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

/// Output current state in RL dedicated directory
void myRHEA::outputCurrentStateDataRL() {

    /// Set data path
    char path[1024];
    sprintf( path, "%s/rhea_exp/output_data", rl_case_path );

    /// Write to file current solver state, time, time iteration and averaging time
    writer_reader->setAttribute( "Time", current_time );
    writer_reader->setAttribute( "Iteration", current_time_iter );
    writer_reader->setAttribute( "AveragingTime", averaging_time );
    writer_reader->writeRL( current_time_iter, tag, global_step, path );

};


void myRHEA::timeAdvanceConservedVariables() {

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

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

#if _ZERO_NET_FLUX_PERTURBATION_LOAD_
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
#endif 

    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {

#if _ZERO_NET_FLUX_PERTURBATION_LOAD_
                /// Enforce net flux zero of the RL perturbation load
                rl_f_rhou_field[I1D(i,j,k)] -= global_rl_f_rhou_volume / global_volume; 
                rl_f_rhov_field[I1D(i,j,k)] -= global_rl_f_rhov_volume / global_volume;
                rl_f_rhow_field[I1D(i,j,k)] -= global_rl_f_rhow_volume / global_volume;
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

	        }
        }
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
        /// Initialize MPI stuff
        int my_rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        /// Iterate through temporal point probes
        for(int tpp = 0; tpp < number_temporal_point_probes; ++tpp) {
            /// Write temporal point probe data to file (if criterion satisfied)
            if( current_time_iter%tpp_output_frequency_iters[tpp] == 0 ) {
                /// Owner rank writes to file
                if( temporal_point_probes[tpp].getGlobalOwnerRank() == my_rank ) {
                    int i_index, j_index, k_index;
                    /// Get local indices i, j, k
                    i_index = temporal_point_probes[tpp].getLocalIndexI(); 
                    j_index = temporal_point_probes[tpp].getLocalIndexJ(); 
                    k_index = temporal_point_probes[tpp].getLocalIndexK();
                    /// Generate header string
                    string output_header_string; 
                    output_header_string  = "# t [s], x [m], y [m], z[m], rho [kg/m3], u [m/s], v [m/s], w [m/s]";
                    output_header_string += ", avg_u [m/s], avg_v [m/s], avg_w [m/s]";
                    output_header_string += ", rmsf_u [m/s], rmsf_v [m/s], rmsf_w [m/s]";
	                output_header_string += ", favre_uffuff [m2/s2], favre_uffvff [m2/s2], favre_uffwff [m2/s2], favre_vffvff [m2/s2], favre_vffwff [m2/s2], favre_wffwff [m2/s2]";
                    output_header_string += ", rhou_inv_flux [kg/m2s2], rhov_inv_flux [kg/m2s2], rhow_inv_flux [kg/m2s2]";
                    output_header_string += ", rhou_vis_flux [kg/m2s2], rhov_vis_flux [kg/m2s2], rhow_vis_flux [kg/m2s2]";
                    output_header_string += ", f_rhou [kg/m2s2], f_rhov [kg/m2s2], f_rhow [kg/m2s2]";
                    output_header_string += ", rl_f_rhou [kg/m2s2], rl_f_rhov [kg/m2s2], rl_f_rhow [kg/m2s2]";
                    output_header_string += ", rl_f_rhou_curr_step [kg/m2s2], rl_f_rhov_curr_step [kg/m2s2], rl_f_rhow_curr_step [kg/m2s2]";
                    output_header_string += ", d_DeltaRxj_j [m/s2], d_DeltaRyj_j [m/s2], d_DeltaRzj_j [m/s2]";
                    output_header_string += ", d_DeltaRxx_x [m/s2], d_DeltaRxy_x [m/s2], d_DeltaRxz_x [m/s2]";
                    output_header_string += ", d_DeltaRxy_y [m/s2], d_DeltaRyy_y [m/s2], d_DeltaRyz_y [m/s2]";
                    output_header_string += ", d_DeltaRxz_z [m/s2], d_DeltaRyz_z [m/s2], d_DeltaRzz_z [m/s2]";
                    /// Generate data string
                    ostringstream sstr; sstr.precision( fstream_precision ); sstr << fixed;
                    sstr << current_time;
                    sstr << "," << x_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << y_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << z_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << rho_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << u_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << v_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << w_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << avg_u_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << avg_v_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << avg_w_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << rmsf_u_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << rmsf_v_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << rmsf_w_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << favre_uffuff_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << favre_uffvff_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << favre_uffwff_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << favre_vffvff_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << favre_vffwff_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << favre_wffwff_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << rhou_inv_flux[I1D(i_index,j_index,k_index)];
                    sstr << "," << rhov_inv_flux[I1D(i_index,j_index,k_index)];
                    sstr << "," << rhow_inv_flux[I1D(i_index,j_index,k_index)];
                    sstr << "," << rhou_vis_flux[I1D(i_index,j_index,k_index)];
                    sstr << "," << rhov_vis_flux[I1D(i_index,j_index,k_index)];
                    sstr << "," << rhow_vis_flux[I1D(i_index,j_index,k_index)];                    
                    sstr << "," << f_rhou_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << f_rhov_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << f_rhow_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << rl_f_rhou_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << rl_f_rhov_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << rl_f_rhow_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << rl_f_rhou_field_curr_step[I1D(i_index,j_index,k_index)];
                    sstr << "," << rl_f_rhov_field_curr_step[I1D(i_index,j_index,k_index)];
                    sstr << "," << rl_f_rhow_field_curr_step[I1D(i_index,j_index,k_index)];
                    sstr << "," << d_DeltaRxj_j_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << d_DeltaRyj_j_field[I1D(i_index,j_index,k_index)]; 
                    sstr << "," << d_DeltaRzj_j_field[I1D(i_index,j_index,k_index)]; 
                    sstr << "," << d_DeltaRxx_x_field[I1D(i_index,j_index,k_index)]; 
                    sstr << "," << d_DeltaRxy_x_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << d_DeltaRxz_x_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << d_DeltaRxy_y_field[I1D(i_index,j_index,k_index)]; 
                    sstr << "," << d_DeltaRyy_y_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << d_DeltaRyz_y_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << d_DeltaRxz_z_field[I1D(i_index,j_index,k_index)]; 
                    sstr << "," << d_DeltaRyz_z_field[I1D(i_index,j_index,k_index)];
                    sstr << "," << d_DeltaRzz_z_field[I1D(i_index,j_index,k_index)];
                    string output_data_string = sstr.str();
                    /// Write (header string) data string to file
                    temporal_point_probes[tpp].writeDataStringToOutputFile(output_header_string, output_data_string);
                }
            }
        }	    
    }
};


void myRHEA::manageStreamwiseBulkVelocity() {

    double delta_x, delta_y, delta_z, delta_volume;
    double local_volume = 0.0;
    double local_u_volume = 0.0;
    double local_avg_u_volume = 0.0;
    double local_avg_v_volume = 0.0;
    double local_avg_w_volume = 0.0;
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                /// Geometric stuff
                delta_x = 0.5*( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ); 
                delta_y = 0.5*( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ); 
                delta_z = 0.5*( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
                delta_volume =  delta_x * delta_y * delta_z;
                /// Update values
                local_volume       += delta_volume;
                local_u_volume     += u_field[I1D(i,j,k)] * delta_volume;
                local_avg_u_volume += avg_u_field[I1D(i,j,k)] * delta_volume;
                local_avg_v_volume += avg_v_field[I1D(i,j,k)] * delta_volume;
                local_avg_w_volume += avg_w_field[I1D(i,j,k)] * delta_volume;
            }
        }
    }
    /// Communicate local values to obtain global & average values
    double global_volume       = 0.0;
    double global_u_volume     = 0.0;
    double global_avg_u_volume = 0.0;
    double global_avg_v_volume = 0.0;
    double global_avg_w_volume = 0.0;
    MPI_Allreduce(&local_volume,       &global_volume,       1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_u_volume,     &global_u_volume,     1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_avg_u_volume, &global_avg_u_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_avg_v_volume, &global_avg_v_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_avg_w_volume, &global_avg_w_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    /// Calculate u_bulk numeric
    double u_bulk_numeric     = global_u_volume     / global_volume;
    double avg_u_bulk_numeric = global_avg_u_volume / global_volume;
    double avg_v_bulk_numeric = global_avg_v_volume / global_volume;
    double avg_w_bulk_numeric = global_avg_w_volume / global_volume;

#if _CORRECT_U_BULK_
    /// Correct flow flux using instantaneous u_bulk
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                u_field[I1D(i,j,k)] += (u_bulk_reference - u_bulk_numeric);
            }
        }
    }
#endif

#if _RL_EARLY_EPISODE_TERMINATION_FUNC_U_BULK_
    int my_rank;
    double final_time_aux;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if ( avg_u_bulk_numeric >= avg_u_bulk_max && !rl_early_episode_termination) {
        /// Only executed once, if early termination is necessary 
        rl_early_episode_termination = true;
        final_time_aux = current_time + 3.0 * ( actuation_period + delta_t );
        if (final_time_aux < final_time) final_time = final_time_aux; 
        if( my_rank == 0 ) cout << endl << "RL EARLY EPISODE TERMINATION as maximum avg_u_bulk " << avg_u_bulk_max << " is reached with numerical avg_u_bulk " << avg_u_bulk_numeric << " at time " << current_time << ". Final time set to " << final_time << endl;
    } else if ( avg_u_bulk_numeric <= avg_u_bulk_min && !rl_early_episode_termination) {
        /// Only executed once, if early termination is necessary 
        rl_early_episode_termination = true;
        final_time_aux = current_time + 3.0 * ( actuation_period + delta_t );
        if (final_time_aux < final_time) final_time = final_time_aux; 
        if( my_rank == 0 ) cout << endl << "RL EARLY EPISODE TERMINATION as minimum avg_u_bulk " << avg_u_bulk_min << " is reached with numerical avg_u_bulk " << avg_u_bulk_numeric << " at time " << current_time << ". Final time set to " << final_time << endl;
    } else {
        if( my_rank == 0 ) cout << endl << "Numerical avg_u_bulk: " << avg_u_bulk_numeric << ", avg_v_bulk: " << avg_v_bulk_numeric << ", avg_w_bulk: " << avg_w_bulk_numeric << ", time: " << current_time << ", final time: " << final_time << endl;
    }
#endif /// of _RL_EARLY_EPISODE_TERMINATION_FUNC_U_BULK_

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
/// Calculate partial derivatives DeltaRij_j at (fine) mesh points from action-based DeltaRij values at (coarse) TCP 3-D mesh
/// Calculate and update partial derivative fields d_DeltaRij_j, i.e.:
/// > update d_DeltaRxx_x_field, d_DeltaRxy_x_field, d_DeltaRxz_x_field, 
///          d_DeltaRxy_y_field, d_DeltaRyy_y_field, d_DeltaRyz_y_field, 
///          d_DeltaRxz_z_field, d_DeltaRyz_z_field, d_DeltaRzz_z_field,
///          d_DeltaRxj_j_field, d_DeltaRyj_j_field, d_DeltaRzj_j_field.
/// Args:
/// tcp_position: position of the unique, local temporal control probe (TCP) of local mpi process
///       tensor<double> tcp_position(3) = {x, y, z}
/// tcp_DeltaRij: DeltaRij tensor at the unique, local temporal control probe (TCP) of local mpi process
///       tensor<double> tcp_DeltaRij(6) = {DeltaRxx, DeltaRxy, DeltaRxz, DeltaRyy, DeltaRyz, DeltaRzz}
void myRHEA::calculate_d_DeltaRij_j(vector<double> &tcp_position, vector<double> &tcp_DeltaRij) {

    constexpr int NUM_TCP_NEIGHBORS = 9;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) cout << "[calculate_d_DeltaRij_j] Calculating DeltaRij..." << endl;

    /// --------- Send/Recv DeltaRij values betw neighboring mpi processes / rl environments ---------

    /// Data exchange between neighboring temporal control probes (tcp), with single tcp per mpi process
    /// Send/Receive tcp_data_pos = [x,y,z] at tcp of corresponding mpi process
    ///          and tcp_data_DeltaRij = [DeltaRxx, DeltaRxy, DeltaRxz, DeltaRyy, DeltaRyz, DeltaRzz] at tcp of corresponding mpi process
    /// NUM_TCP_NEIGHBORS= 9 TCP considered: 
    ///     [tcp_data_111 (current MPI process), tcp_data_011, tcp_data_211, tcp_data_101, tcp_data_121, tcp_data_110, tcp_data_112]
    vector<vector<double>> tcp_data_pos(     NUM_TCP_NEIGHBORS, vector<double>(3, 0.0));
    vector<vector<double>> tcp_data_DeltaRij(NUM_TCP_NEIGHBORS, vector<double>(6, 0.0));

    /// Set local TCP data of current mpi process (central point in the coarse 3x3x3 grid of TCP)
    tcp_data_pos[0]      = {tcp_position[0], tcp_position[1], tcp_position[2]};
    tcp_data_DeltaRij[0] = {tcp_DeltaRij[0], tcp_DeltaRij[1], tcp_DeltaRij[2], tcp_DeltaRij[3], tcp_DeltaRij[4], tcp_DeltaRij[5]};
    
    // Fetch TCP data from neighboring MPI processes (unique TCP per MPI process)
    exchangeTcpData(tcp_data_pos, tcp_data_DeltaRij, NUM_TCP_NEIGHBORS);   /// update tcp_data
    
    /// Calculate d(DeltaRij)/dx_j by 1st and 2nd-order partial derivatives betw TCP neighbors
    /// > 1st-order partial derivatives for TCP mesh boundary probe
    /// > 2nd-order partial derivatives ofr TCP mesh inner probe
    /// ---- Partial derivatives in x-direction, d_DeltaRij_x ----
    double dx, d_DeltaRxx_x, d_DeltaRxy_x, d_DeltaRxz_x;
    bool isTcpBottomX = ( std::abs(tcp_data_pos[1][0]-0.0) < EPS );
    bool isTcpTopX    = ( std::abs(tcp_data_pos[2][0]-L_x) < EPS );
    if (!isTcpBottomX && !isTcpTopX) {
        /// Inner TCP in x-dir -> 2nd-order partial derivative in x-dir
        dx = tcp_data_pos[2][0] - tcp_data_pos[1][0];
        d_DeltaRxx_x     = ( tcp_data_DeltaRij[2][0] - tcp_data_DeltaRij[1][0] ) / ( dx );
        d_DeltaRxy_x     = ( tcp_data_DeltaRij[2][1] - tcp_data_DeltaRij[1][1] ) / ( dx );
        d_DeltaRxz_x     = ( tcp_data_DeltaRij[2][2] - tcp_data_DeltaRij[1][2] ) / ( dx );
        /// d_DeltaRyy_x = ( tcp_data_DeltaRij[2][3] - tcp_data_DeltaRij[1][3] ) / ( dx );
        /// d_DeltaRyz_x = ( tcp_data_DeltaRij[2][4] - tcp_data_DeltaRij[1][4] ) / ( dx );
        /// d_DeltaRzz_x = ( tcp_data_DeltaRij[2][5] - tcp_data_DeltaRij[1][5] ) / ( dx );
    } else if (isTcpBottomX && !isTcpTopX) {
        /// Boundary TCP in bottom x-dir -> 1st-order partial derivative in x-dir 
        dx = tcp_data_pos[2][0] - tcp_data_pos[0][0];
        d_DeltaRxx_x     = ( tcp_data_DeltaRij[2][0] - tcp_data_DeltaRij[0][0] ) / ( dx );
        d_DeltaRxy_x     = ( tcp_data_DeltaRij[2][1] - tcp_data_DeltaRij[0][1] ) / ( dx );
        d_DeltaRxz_x     = ( tcp_data_DeltaRij[2][2] - tcp_data_DeltaRij[0][2] ) / ( dx );
        /// d_DeltaRyy_x = ( tcp_data_DeltaRij[2][3] - tcp_data_DeltaRij[0][3] ) / ( dx );
        /// d_DeltaRyz_x = ( tcp_data_DeltaRij[2][4] - tcp_data_DeltaRij[0][4] ) / ( dx );
        /// d_DeltaRzz_x = ( tcp_data_DeltaRij[2][5] - tcp_data_DeltaRij[0][5] ) / ( dx );
    } else if (!isTcpBottomX && isTcpTopX) {
        /// Boundary TCP in top x-dir -> 1st-order partial derivative in x-dir 
        dx = tcp_data_pos[0][0] - tcp_data_pos[1][0];
        d_DeltaRxx_x     = ( tcp_data_DeltaRij[0][0] - tcp_data_DeltaRij[1][0] ) / ( dx );
        d_DeltaRxy_x     = ( tcp_data_DeltaRij[0][1] - tcp_data_DeltaRij[1][1] ) / ( dx );
        d_DeltaRxz_x     = ( tcp_data_DeltaRij[0][2] - tcp_data_DeltaRij[1][2] ) / ( dx );
        /// d_DeltaRyy_x = ( tcp_data_DeltaRij[0][3] - tcp_data_DeltaRij[1][3] ) / ( dx );
        /// d_DeltaRyz_x = ( tcp_data_DeltaRij[0][4] - tcp_data_DeltaRij[1][4] ) / ( dx );
        /// d_DeltaRzz_x = ( tcp_data_DeltaRij[0][5] - tcp_data_DeltaRij[1][5] ) / ( dx );
    } else {
        /// Only one TCP in x-dir -> null partial derivatives in x-dir
        d_DeltaRxx_x     = 0.0;
        d_DeltaRxy_x     = 0.0;
        d_DeltaRxz_x     = 0.0;
        /// d_DeltaRyy_x = 0.0;
        /// d_DeltaRyz_x = 0.0;
        /// d_DeltaRzz_x = 0.0;
    }

    /// ---- Partial derivatives in y-direction, d_DeltaRij_y ----
    double dy, d_DeltaRxy_y, d_DeltaRyy_y, d_DeltaRyz_y;
    bool isTcpBottomY = ( std::abs(tcp_data_pos[1][1]-0.0) < EPS );
    bool isTcpTopY    = ( std::abs(tcp_data_pos[2][1]-L_y) < EPS );
    if (!isTcpBottomY && !isTcpTopY) {
        /// Inner TCP in y-dir -> 2nd-order partial derivative in y-dir
        dy = tcp_data_pos[4][1] - tcp_data_pos[3][1];
        /// d_DeltaRxx_y = ( tcp_data_DeltaRij[4][0] - tcp_data_DeltaRij[3][0] ) / ( dy );
        d_DeltaRxy_y     = ( tcp_data_DeltaRij[4][1] - tcp_data_DeltaRij[3][1] ) / ( dy );
        /// d_DeltaRxz_y = ( tcp_data_DeltaRij[4][2] - tcp_data_DeltaRij[3][2] ) / ( dy );
        d_DeltaRyy_y     = ( tcp_data_DeltaRij[4][3] - tcp_data_DeltaRij[3][3] ) / ( dy );
        d_DeltaRyz_y     = ( tcp_data_DeltaRij[4][4] - tcp_data_DeltaRij[3][4] ) / ( dy );
        /// d_DeltaRzz_y = ( tcp_data_DeltaRij[4][5] - tcp_data_DeltaRij[3][5] ) / ( dy );
    } else if (isTcpBottomY && !isTcpTopY) {
        /// Boundary TCP in bottom y-dir -> 1st-order partial derivative in y-dir 
        dy = tcp_data_pos[4][1] - tcp_data_pos[0][1];
        /// d_DeltaRxx_y = ( tcp_data_DeltaRij[4][0] - tcp_data_DeltaRij[0][0] ) / ( dy );
        d_DeltaRxy_y     = ( tcp_data_DeltaRij[4][1] - tcp_data_DeltaRij[0][1] ) / ( dy );
        /// d_DeltaRxz_y = ( tcp_data_DeltaRij[4][2] - tcp_data_DeltaRij[0][2] ) / ( dy );
        d_DeltaRyy_y     = ( tcp_data_DeltaRij[4][3] - tcp_data_DeltaRij[0][3] ) / ( dy );
        d_DeltaRyz_y     = ( tcp_data_DeltaRij[4][4] - tcp_data_DeltaRij[0][4] ) / ( dy );
        /// d_DeltaRzz_y = ( tcp_data_DeltaRij[4][5] - tcp_data_DeltaRij[0][5] ) / ( dy );
    } else if (!isTcpBottomY && isTcpTopY) {
        /// Boundary TCP in top y-dir -> 1st-order partial derivative in y-dir 
        dy = tcp_data_pos[0][1] - tcp_data_pos[3][1];
        /// d_DeltaRxx_y = ( tcp_data_DeltaRij[0][0] - tcp_data_DeltaRij[3][0] ) / ( dy );
        d_DeltaRxy_y     = ( tcp_data_DeltaRij[0][1] - tcp_data_DeltaRij[3][1] ) / ( dy );
        /// d_DeltaRxz_y = ( tcp_data_DeltaRij[0][2] - tcp_data_DeltaRij[3][2] ) / ( dy );
        d_DeltaRyy_y     = ( tcp_data_DeltaRij[0][3] - tcp_data_DeltaRij[3][3] ) / ( dy );
        d_DeltaRyz_y     = ( tcp_data_DeltaRij[0][4] - tcp_data_DeltaRij[3][4] ) / ( dy );
        /// d_DeltaRzz_y = ( tcp_data_DeltaRij[0][5] - tcp_data_DeltaRij[3][5] ) / ( dy );
    } else {
        /// Only one TCP in y-dir -> null partial derivatives in y-dir
        /// d_DeltaRxx_y = 0.0;
        d_DeltaRxy_y     = 0.0;
        /// d_DeltaRxz_y = 0.0;
        d_DeltaRyy_y     = 0.0;
        d_DeltaRyz_y     = 0.0;
        /// d_DeltaRzz_y = 0.0;
    }

    /// ---- Partial derivatives in z-direction, d_DeltaRij_z ----
    double dz, d_DeltaRxz_z, d_DeltaRyz_z, d_DeltaRzz_z;
    bool isTcpBottomZ = ( std::abs(tcp_data_pos[5][2]-0.0) < EPS );
    bool isTcpTopZ    = ( std::abs(tcp_data_pos[6][2]-L_z) < EPS );
    if (!isTcpBottomZ && !isTcpTopZ) {
        /// Inner TCP in z-dir -> 2nd-order partial derivative in z-dir
        dz = tcp_data_pos[6][2] - tcp_data_pos[5][2];
        /// d_DeltaRxx_z = ( tcp_data_DeltaRij[6][0] - tcp_data_DeltaRij[5][0] ) / ( dz );
        /// d_DeltaRxy_z = ( tcp_data_DeltaRij[6][1] - tcp_data_DeltaRij[5][1] ) / ( dz );
        d_DeltaRxz_z     = ( tcp_data_DeltaRij[6][2] - tcp_data_DeltaRij[5][2] ) / ( dz );
        /// d_DeltaRyy_z = ( tcp_data_DeltaRij[6][3] - tcp_data_DeltaRij[5][3] ) / ( dz );
        d_DeltaRyz_z     = ( tcp_data_DeltaRij[6][4] - tcp_data_DeltaRij[5][4] ) / ( dz );
        d_DeltaRzz_z     = ( tcp_data_DeltaRij[6][5] - tcp_data_DeltaRij[5][5] ) / ( dz );
    } else if (isTcpBottomZ && !isTcpTopZ) {
        /// Boundary TCP in bottom z-dir -> 1st-order partial derivative in z-dir 
        dz = tcp_data_pos[6][2] - tcp_data_pos[0][2];
        /// d_DeltaRxx_z = ( tcp_data_DeltaRij[6][0] - tcp_data_DeltaRij[0][0] ) / ( dz );
        /// d_DeltaRxy_z = ( tcp_data_DeltaRij[6][1] - tcp_data_DeltaRij[0][1] ) / ( dz );
        d_DeltaRxz_z     = ( tcp_data_DeltaRij[6][2] - tcp_data_DeltaRij[0][2] ) / ( dz );
        /// d_DeltaRyy_z = ( tcp_data_DeltaRij[6][3] - tcp_data_DeltaRij[0][3] ) / ( dz );
        d_DeltaRyz_z     = ( tcp_data_DeltaRij[6][4] - tcp_data_DeltaRij[0][4] ) / ( dz );
        d_DeltaRzz_z     = ( tcp_data_DeltaRij[6][5] - tcp_data_DeltaRij[0][5] ) / ( dz );
    } else if (!isTcpBottomZ && isTcpTopZ) {
        /// Boundary TCP in top z-dir -> 1st-order partial derivative in z-dir 
        dz = tcp_data_pos[0][2] - tcp_data_pos[5][2];
        /// d_DeltaRxx_z = ( tcp_data_DeltaRij[0][0] - tcp_data_DeltaRij[5][0] ) / ( dz );
        /// d_DeltaRxy_z = ( tcp_data_DeltaRij[0][1] - tcp_data_DeltaRij[5][1] ) / ( dz );
        d_DeltaRxz_z     = ( tcp_data_DeltaRij[0][2] - tcp_data_DeltaRij[5][2] ) / ( dz );
        /// d_DeltaRyy_z = ( tcp_data_DeltaRij[0][3] - tcp_data_DeltaRij[5][3] ) / ( dz );
        d_DeltaRyz_z     = ( tcp_data_DeltaRij[0][4] - tcp_data_DeltaRij[5][4] ) / ( dz );
        d_DeltaRzz_z     = ( tcp_data_DeltaRij[0][5] - tcp_data_DeltaRij[5][5] ) / ( dz );
    } else {
        /// Only one TCP in z-dir -> null partial derivatives in z-dir
        /// d_DeltaRxx_z = 0.0;
        /// d_DeltaRxy_z = 0.0;
        d_DeltaRxz_z     = 0.0;
        /// d_DeltaRyy_z = 0.0;
        d_DeltaRyz_z     = 0.0;
        d_DeltaRzz_z     = 0.0;
    }

    /// Update d_DeltaRij_j fields
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {    
                d_DeltaRxx_x_field[I1D(i,j,k)] = d_DeltaRxx_x;
                d_DeltaRxy_x_field[I1D(i,j,k)] = d_DeltaRxy_x;
                d_DeltaRxz_x_field[I1D(i,j,k)] = d_DeltaRxz_x;
                d_DeltaRxy_y_field[I1D(i,j,k)] = d_DeltaRxy_y;
                d_DeltaRyy_y_field[I1D(i,j,k)] = d_DeltaRyy_y;
                d_DeltaRyz_y_field[I1D(i,j,k)] = d_DeltaRyz_y;
                d_DeltaRxz_z_field[I1D(i,j,k)] = d_DeltaRxz_z;
                d_DeltaRyz_z_field[I1D(i,j,k)] = d_DeltaRyz_z;
                d_DeltaRzz_z_field[I1D(i,j,k)] = d_DeltaRzz_z;
                d_DeltaRxj_j_field[I1D(i,j,k)] = d_DeltaRxx_x + d_DeltaRxy_y + d_DeltaRxz_z;
                d_DeltaRyj_j_field[I1D(i,j,k)] = d_DeltaRxy_x + d_DeltaRyy_y + d_DeltaRyz_z;
                d_DeltaRzj_j_field[I1D(i,j,k)] = d_DeltaRxz_x + d_DeltaRyz_y + d_DeltaRzz_z;
            }
        }
    }
    if (my_rank == 0) cout << "[calculate_d_DeltaRij_j] d_DeltaRij_j calculated using finite differences from TCP data" << endl;

};


///////////////////////////////////////////////////////////////////////////////
/// TODO: add function description
/*  
    vector<vector<double>> tcp_data(NUM_TCP_NEIGHBORS, vector<double>(TCP_DATA_SIZE, 0.0))
    Contains TCP_DAT_SIZE = 9-dim data (x, y, z, DeltaRxx, DeltaRxy, DeltaRxz, DeltaRyy, DeltaRyz, DeltaRzz) for each NUM_TCP_NEIGHBORS = 9 :
        tcp_data_111 (current MPI process), 
        tcp_data_011, tcp_data_211, 
        tcp_data_101, tcp_data_121, 
        tcp_data_110, tcp_data_112
    Correspondence betw tcp_data idx and neighbouring tcp:
        vector<double> tcp_data_111 = tcp_data[0]
        vector<double> tcp_data_011 = tcp_data[1]
        vector<double> tcp_data_211 = tcp_data[2]
        vector<double> tcp_data_101 = tcp_data[3]
        vector<double> tcp_data_121 = tcp_data[4]
        vector<double> tcp_data_110 = tcp_data[5]
        vector<double> tcp_data_112 = tcp_data[6]
}
*/ 
void myRHEA::exchangeTcpData(vector<vector<double>> &tcp_data_pos, vector<vector<double>> &tcp_data_DeltaRij, const int &num_tcp_neighbors) {

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (num_tcp_neighbors != 9) {
        cerr << "Function 'exchangeTcpData' implemented for num_tcp_neighbors = 9, but num_tcp_neighbors = " << num_tcp_neighbors << " found." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /// Data exchange auxiliary variables 
    MPI_Request requests[32];
    int req_count = 0;

    /// Compute mpi process coordinates in the coarse TCP 3D grid
    int tcp_iz = my_rank / (np_x * np_y);
    int tcp_iy = (my_rank % (np_x * np_y)) / np_x;
    int tcp_ix = my_rank % np_x;
    cout << "[exchangeTcpData] [Rank " << my_rank << "] TPC has indexes (tcp_ix, tcp_iy, tcp_iz) = (" << tcp_ix << ", " << tcp_iy << ", " << tcp_iz << ")" << endl;
    
    /// ----- Data exchange in (x)-direction -----
    /// Exchange data between TCP 111 [0] <-> 011 [1]
    if (tcp_ix > 0) {
        MPI_Isend(tcp_data_pos[0].data(),       3, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);        /// from 111 to 011
        MPI_Irecv(tcp_data_pos[1].data(),       3, MPI_DOUBLE, my_rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);        /// from 011 to 111
        MPI_Isend(tcp_data_DeltaRij[0].data(),  6, MPI_DOUBLE, my_rank - 1, 2, MPI_COMM_WORLD, &requests[req_count++]);        /// from 111 to 011
        MPI_Irecv(tcp_data_DeltaRij[1].data(),  6, MPI_DOUBLE, my_rank - 1, 3, MPI_COMM_WORLD, &requests[req_count++]);        /// from 011 to 111
    } else {
        tcp_data_pos[1]      = {0.0, tcp_data_pos[0][1], tcp_data_pos[0][2]};
        tcp_data_DeltaRij[1] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    /// Exchange data between TCP 111 [0] <-> 211 [2]
    if (tcp_ix < np_x - 1) {
        MPI_Isend(tcp_data_pos[0].data(),       3, MPI_DOUBLE, my_rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);        /// from 111 to 211
        MPI_Irecv(tcp_data_pos[2].data(),       3, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);        /// from 211 to 111
        MPI_Isend(tcp_data_DeltaRij[0].data(),  6, MPI_DOUBLE, my_rank + 1, 3, MPI_COMM_WORLD, &requests[req_count++]);        /// from 111 to 211
        MPI_Irecv(tcp_data_DeltaRij[2].data(),  6, MPI_DOUBLE, my_rank + 1, 2, MPI_COMM_WORLD, &requests[req_count++]);        /// from 211 to 111

    } else {
        tcp_data_pos[2]      = {L_x, tcp_data_pos[0][1], tcp_data_pos[0][2]};
        tcp_data_DeltaRij[2] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    
    /// ----- Data exchange in (y)-direction -----
    /// Exchange data between TCP 111 [0] <-> 101 [3]
    if (tcp_iy > 0) {
        MPI_Isend(tcp_data_pos[0].data(),       3, MPI_DOUBLE, my_rank - np_x, 4, MPI_COMM_WORLD, &requests[req_count++]);    /// from 111 to 101
        MPI_Irecv(tcp_data_pos[3].data(),       3, MPI_DOUBLE, my_rank - np_x, 5, MPI_COMM_WORLD, &requests[req_count++]);    /// from 101 to 111
        MPI_Isend(tcp_data_DeltaRij[0].data(),  6, MPI_DOUBLE, my_rank - np_x, 6, MPI_COMM_WORLD, &requests[req_count++]);    /// from 111 to 101
        MPI_Irecv(tcp_data_DeltaRij[3].data(),  6, MPI_DOUBLE, my_rank - np_x, 7, MPI_COMM_WORLD, &requests[req_count++]);    /// from 101 to 111
    } else {
        tcp_data_pos[3]      = {tcp_data_pos[0][0], 0.0, tcp_data_pos[0][2]};
        tcp_data_DeltaRij[3] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    /// Exchange data between TCP 111 [0] <-> 121 [4]
    if (tcp_iy < np_y - 1) {
        MPI_Isend(tcp_data_pos[0].data(),       3, MPI_DOUBLE, my_rank + np_x, 5, MPI_COMM_WORLD, &requests[req_count++]);    /// from 111 to 121
        MPI_Irecv(tcp_data_pos[4].data(),       3, MPI_DOUBLE, my_rank + np_x, 4, MPI_COMM_WORLD, &requests[req_count++]);    /// from 121 to 111
        MPI_Isend(tcp_data_DeltaRij[0].data(),  6, MPI_DOUBLE, my_rank + np_x, 7, MPI_COMM_WORLD, &requests[req_count++]);    /// from 111 to 121
        MPI_Irecv(tcp_data_DeltaRij[4].data(),  6, MPI_DOUBLE, my_rank + np_x, 6, MPI_COMM_WORLD, &requests[req_count++]);    /// from 121 to 111
    } else {
        tcp_data_pos[4]      = {tcp_data_pos[0][0], L_y, tcp_data_pos[0][2]};
        tcp_data_DeltaRij[4] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }

    /// ----- Data exchange in (z)-direction -----
    /// Exchange data between TCP 111 [0] <-> 110 [5]
    if (tcp_iz > 0) {
        MPI_Isend(tcp_data_pos[0].data(),       3, MPI_DOUBLE, my_rank - np_x * np_y, 8,  MPI_COMM_WORLD, &requests[req_count++]);   /// from 111 to 110
        MPI_Irecv(tcp_data_pos[5].data(),       3, MPI_DOUBLE, my_rank - np_x * np_y, 9,  MPI_COMM_WORLD, &requests[req_count++]);   /// from 110 to 111
        MPI_Isend(tcp_data_DeltaRij[0].data(),  6, MPI_DOUBLE, my_rank - np_x * np_y, 10, MPI_COMM_WORLD, &requests[req_count++]);   /// from 111 to 110
        MPI_Irecv(tcp_data_DeltaRij[5].data(),  6, MPI_DOUBLE, my_rank - np_x * np_y, 11, MPI_COMM_WORLD, &requests[req_count++]);   /// from 110 to 111
    } else {
        tcp_data_pos[5]      = {tcp_data_pos[0][0], tcp_data_pos[0][1], 0.0};
        tcp_data_DeltaRij[5] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    /// Exchange data between TCP 111 [0] <-> 112 [6]
    if (tcp_iz < np_z - 1) {
        MPI_Isend(tcp_data_pos[0].data(),        3, MPI_DOUBLE, my_rank + np_x * np_y, 9,  MPI_COMM_WORLD, &requests[req_count++]);   /// from 111 to 112
        MPI_Irecv(tcp_data_pos[6].data(),        3, MPI_DOUBLE, my_rank + np_x * np_y, 8,  MPI_COMM_WORLD, &requests[req_count++]);   /// from 112 to 111
        MPI_Isend(tcp_data_DeltaRij[0].data(),   6, MPI_DOUBLE, my_rank + np_x * np_y, 11, MPI_COMM_WORLD, &requests[req_count++]);   /// from 111 to 112
        MPI_Irecv(tcp_data_DeltaRij[6].data(),   6, MPI_DOUBLE, my_rank + np_x * np_y, 10, MPI_COMM_WORLD, &requests[req_count++]);   /// from 112 to 111
    } else {
        tcp_data_pos[6]      = {tcp_data_pos[0][0], tcp_data_pos[0][1], L_z};
        tcp_data_DeltaRij[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }

    /// Wait for all non-blocking communication to complete
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

    /// Validate data
    validateExchangeTcpData(tcp_data_pos); 

    if (my_rank == 0) cout << "[exchangeTcpData] TCP Positions and DeltaRij data exchanged" << endl;

}


/// Validate exchanged data (local control probe coordinates) between neighbouring mpi processes in all x,y,z-directions
void myRHEA::validateExchangeTcpData(const vector<vector<double>> &tcp_data_pos) {
    
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /// Check shared coordinate values
    /// Check x-coordinates match at y-z planes (x_match)
    bool x_match = checkMatch({tcp_data_pos[0][0], tcp_data_pos[3][0], tcp_data_pos[4][0], tcp_data_pos[5][0], tcp_data_pos[6][0]});
    /// Check y-coordinates match at x-z planes (y_match)
    bool y_match = checkMatch({tcp_data_pos[0][1], tcp_data_pos[1][1], tcp_data_pos[2][1], tcp_data_pos[5][1], tcp_data_pos[6][1]});
    /// Check z-coordinates match at x-y planes (z_match) 
    bool z_match = checkMatch({tcp_data_pos[0][2], tcp_data_pos[1][2], tcp_data_pos[2][2], tcp_data_pos[3][2], tcp_data_pos[4][2]});

    /// Check coordinates ordering (taking into account x,y,z_match is checked already)
    bool x_order_correct = (tcp_data_pos[1][0] < tcp_data_pos[0][0]) && (tcp_data_pos[0][0] < tcp_data_pos[2][0]);
    bool y_order_correct = (tcp_data_pos[3][1] < tcp_data_pos[0][1]) && (tcp_data_pos[0][1] < tcp_data_pos[4][1]);
    bool z_order_correct = (tcp_data_pos[5][2] < tcp_data_pos[0][2]) && (tcp_data_pos[0][2] < tcp_data_pos[6][2]);

    if (!(x_match && y_match && z_match && x_order_correct && y_order_correct && z_order_correct)) {
        cerr << "[Rank " << my_rank << "] Data exchange validation FAILED!, with tcp_data_pos:\n"
             << tcp_data_pos[0][0] << " " << tcp_data_pos[0][1] << " " << tcp_data_pos[0][2] << "\n"
             << tcp_data_pos[1][0] << " " << tcp_data_pos[1][1] << " " << tcp_data_pos[1][2] << "\n" 
             << tcp_data_pos[2][0] << " " << tcp_data_pos[2][1] << " " << tcp_data_pos[2][2] << "\n" 
             << tcp_data_pos[3][0] << " " << tcp_data_pos[3][1] << " " << tcp_data_pos[3][2] << "\n" 
             << tcp_data_pos[4][0] << " " << tcp_data_pos[4][1] << " " << tcp_data_pos[4][2] << "\n" 
             << tcp_data_pos[5][0] << " " << tcp_data_pos[5][1] << " " << tcp_data_pos[5][2] << "\n"
             << tcp_data_pos[6][0] << " " << tcp_data_pos[6][1] << " " << tcp_data_pos[6][2] << "\n" << endl;
        if (!x_match) cerr << "  -> X-coordinates mismatch!\n" << endl;
        if (!y_match) cerr << "  -> Y-coordinates mismatch!\n" << endl;
        if (!z_match) cerr << "  -> Z-coordinates mismatch!\n" << endl;
        if (!x_order_correct) cerr << "  -> X-coordinates order incorrect!\n" << endl;
        if (!y_order_correct) cerr << "  -> Y-coordinates order incorrect!\n" << endl;
        if (!z_order_correct) cerr << "  -> Z-coordinates order incorrect!\n" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1);
    } else {
        if (my_rank == 0) cout << "[validateExchangeTcpData] TCP Data exchanged and validated" << endl;
    }

}

///////////////////////////////////////////////////////////////////////////////
/// Check input doubles have the same value within EPS tolerance
bool myRHEA::checkMatch(initializer_list<double> values) {
    if (values.size() < 2) return true;

    auto it = values.begin();
    double first = *it;
    ++it;

    for (; it != values.end(); ++it) {
        if (std::abs(*it - first) > EPS) {
            return false;
        }
    }
    return true;
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

        /// Read lines, containing witness points coordinates
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
    this->state_local_size2 = state_dim * state_local_size2_counter;
    cout << "[Rank " << my_rank << "] num. local witness points: " << state_local_size2_counter << ", and state local size: " << state_local_size2 << endl;
    cout.flush();
    MPI_Barrier(MPI_COMM_WORLD);

}


/// Read control points
void myRHEA::readControlPoints(){

    /// Get attrib. 'num_control_probes' (in rank=0)
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0) {
        cout << "\nReading control points..." << endl;

        /// Read file (only with 1 mpi process to avoid file accessing errors)
        std::ifstream file(control_file);
        if (!file.is_open()) {
            cerr << "Unable to open file: " << control_file << endl;
            return;
        }
        
         /// Read lines, containing control points coordinates
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            double ix, iy, iz;
            if (!(iss >> ix >> iy >> iz)) {
                std::cerr << "Error reading line: " << line << std::endl;
                return;
            }
            
            /// Fill control points position tensors
            tcp_x_positions.push_back(ix);
            tcp_y_positions.push_back(iy);
            tcp_z_positions.push_back(iz);
        }
        file.close();
        
        num_control_probes = static_cast<int>(tcp_x_positions.size());
        cout << "Number of control probes: " << num_control_probes << endl;
    }

    /// Broadcast the number of control probes to all mpi processes
    MPI_Bcast(&num_control_probes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize the vectors to hold data in all processes
    if (my_rank != 0) {
        tcp_x_positions.resize(num_control_probes);
        tcp_y_positions.resize(num_control_probes);
        tcp_z_positions.resize(num_control_probes);
    }

    /// Broadcast the control probes coordinates from to all processes
    MPI_Bcast(tcp_x_positions.data(), num_control_probes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(tcp_y_positions.data(), num_control_probes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(tcp_z_positions.data(), num_control_probes, MPI_DOUBLE, 0, MPI_COMM_WORLD);

}

/* Pre-process control points
   Builds 'temporal_control_probes', vector of TemporalPointProbe elements 
*/
void myRHEA::preproceControlPoints() {

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0) {
        cout << "\nPreprocessing control points..." << endl;
    }

    /// Construct (initialize) temporal point probes for control points
    TemporalPointProbe temporal_control_probe(mesh, topo);
    temporal_control_probes.resize( num_control_probes );
    for(int tcp = 0; tcp < num_control_probes; ++tcp) {
        /// Set parameters of temporal point probe
        temporal_control_probe.setPositionX( tcp_x_positions[tcp] );
        temporal_control_probe.setPositionY( tcp_y_positions[tcp] );
        temporal_control_probe.setPositionZ( tcp_z_positions[tcp] );
        /// Insert temporal point probe to vector
        temporal_control_probes[tcp] = temporal_control_probe;
        /// Locate closest grid point to probe
        temporal_control_probes[tcp].locateClosestGridPointToProbe();
    }

    /// Calculate num. control probes local (of my_rank) and debugging logs
    int num_control_probes_local = 0;
    for(int tcp = 0; tcp < num_control_probes; ++tcp) {
        /// Owner rank writes to file
        if( temporal_control_probes[tcp].getGlobalOwnerRank() == my_rank ) {
            num_control_probes_local += 1;
        }
    }

    /// Calculate 'action_local_size2'
    int action_local_size2 = action_dim * num_control_probes_local;

    /// Calculate global attributes 'action_global_size2', 'n_rl_envs' for all mpi processes
    MPI_Allreduce(&action_local_size2,       &action_global_size2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); // update 'action_global_size2'
    MPI_Allreduce(&num_control_probes_local, &n_rl_envs,           1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); // update 'n_rl_envs'


    /// Logging
    cout << "[Rank " << my_rank << "] "
         << "Local number of control probes: " << num_control_probes_local
         << ", Total number of control probes: " << num_control_probes << " == Num. RL pseudo-environmens: " << n_rl_envs
         << ", Local action size: " << action_local_size2
         << ", Global action size: " << action_global_size2
         << ", Action dimension: " << action_dim
         << ", mpi processes grid: " << np_x << " " << np_y << " " << np_z << endl;  

    /// Check 0: each mpi process contains a single control point
    if (num_control_probes_local != 1){
        cerr << "[myRHEA::preproceControlCubes] ERROR: [Rank " << my_rank << "] num. control points: " << num_control_probes_local << " != 1"  << endl;
        MPI_Abort( MPI_COMM_WORLD, 1);
    }

    /// Check 1: action_global_size2 == n_rl_envs * action_dim
    if (action_global_size2 != n_rl_envs * action_dim){
        cerr << "[myRHEA::preproceControlCubes] ERROR: [Rank " << my_rank << "] action global size (" << action_global_size2 << ") != n_rl_envs (" << n_rl_envs << ") * action_dim (" << action_dim << ")" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /// Check 2: num. of control probes == num. mpi processes == num. RL environments
    if ((np_x * np_y * np_z == n_rl_envs) && (num_control_probes == n_rl_envs)) {
        /// Logging successful distribution
        if (my_rank == 0) {
            stringstream ss;
            ss << "[Rank 0] Correct RL environments (control probes) and computational domain distribution: "
               << "1 RL environment per MPI process distributed along y-coordinate, with "
               << "np_x: " << np_x << ", np_y: " << np_y << ", np_z: " << np_z << ", number of RL env: " << n_rl_envs;
            cout << ss.str() << endl;
        }
    } else {
        stringstream ss;
        ss << "[myRHEA::preproceControlPoints] ERROR: Invalid RL environments & computational domain distribution, with "
           << "np_x: " << np_x << ", np_y: " << np_y << ", np_z: " << np_z << ", number of RL env: " << n_rl_envs
           << ", Rank " << my_rank << " has " << num_control_probes_local << " RL environments / control probes";
        cerr << endl << ss.str() << endl;     
        MPI_Abort( MPI_COMM_WORLD, 1);
    }

    /// Check 3: tcp ordering correspond to mpi processes distribution order 
    int tcp_global_owner;
    if (my_rank == 0){
        for(int tcp = 0; tcp < num_control_probes; ++tcp) {
            tcp_global_owner = temporal_control_probes[tcp].getGlobalOwnerRank();
            if (tcp_global_owner == tcp) {
                cout << "Control probe #" << tcp << " at position (" << tcp_x_positions[tcp] << ", " << tcp_y_positions[tcp] << ", " << tcp_z_positions[tcp]
                     << ") is owned by mpi process with Rank " << tcp_global_owner << endl;
            } else {
                cerr << "[myRHEA::preproceControlPoints] ERROR: mismatch between temporal control probe #" << tcp << " != mpi process rank #" << tcp_global_owner << endl;
                MPI_Abort( MPI_COMM_WORLD, 1); 
            }  
        }
    }

    /// Debugging: boundaries of RL environments / mpi processes 
    /// cout << "[Rank " << my_rank << "] RL environment / mpi process domain inner boundaries: "  
    ///      << "x in (" << x_field[I1D(topo->iter_common[_INNER_][_INIX_],0,0)] << ", " << x_field[I1D(topo->iter_common[_INNER_][_ENDX_],0,0)] << "), "
    ///      << "y in (" << y_field[I1D(0,topo->iter_common[_INNER_][_INIY_],0)] << ", " << y_field[I1D(0,topo->iter_common[_INNER_][_ENDY_],0)] << "), "
    ///      << "z in (" << z_field[I1D(0,0,topo->iter_common[_INNER_][_INIZ_])] << ", " << z_field[I1D(0,0,topo->iter_common[_INNER_][_ENDZ_])] << ")" << endl;
    /// cout << "[Rank " << my_rank << "] inner boundaries local indices: "
    ///      << "x-idx in (" << topo->iter_common[_INNER_][_INIX_] << ", " << topo->iter_common[_INNER_][_ENDX_] << "), "
    ///      << "y-idx in (" << topo->iter_common[_INNER_][_INIY_] << ", " << topo->iter_common[_INNER_][_ENDY_] << "), "
    ///      << "z-idx in (" << topo->iter_common[_INNER_][_INIZ_] << ", " << topo->iter_common[_INNER_][_ENDZ_] << ")" << endl;

    cout.flush();
    MPI_Barrier(MPI_COMM_WORLD);

}

///////////////////////////////////////////////////////////////////////////////
/// Calculate State and Reward for each agent, and update auxiliary variables
void myRHEA::calculateStateAndReward() {
    this->calculateState();
    this->calculateReward();
    this->updatePreviousActuationFields();
} 

///////////////////////////////////////////////////////////////////////////////
/// Calculate local state values (local to single mpi process, which corresponds to RL environment)
/// -> updates attributes: state_local
/// -> state values: turbulent kinetic energy
void myRHEA::calculateState() {

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int state_local_size2_counter = 0;
    state_local.resize(state_local_size2);
    std::fill(state_local.begin(), state_local.end(), 0.0);
    
    double c1 = 0.2 / actuation_period;
    double c2 = 0.2 / actuation_period;
    double c3 = 0.2 / actuation_period;
    double c4 = 0.1 / actuation_period;
    double c5 = 0.1 / actuation_period;
    double c6 = 0.1 / actuation_period;

    /// Initialize auxiliary variables
#if _WITNESS_XYZ_AVG_
    double delta_x, delta_y, delta_z, delta_volume, total_volume_local;
    double l2_x, l2_y, l2_z;
    double avg_v_bulk, avg_w_bulk;
    double l2_avg_u, l2_rmsf_u, l2_rmsf_v, l2_rmsf_w;
    double l2_avg_u_comp, l2_rmsf_u_comp, l2_rmsf_v_comp, l2_rmsf_w_comp;
    double l2_d_avg_u, l2_d_avg_v, l2_d_avg_w;
    double l2_d_rmsf_u, l2_d_rmsf_v, l2_d_rmsf_w;
#else
    int i_index, j_index, k_index;
#endif  /// of _WITNESS_XYZ_AVG_

    /// Calculate state for each witness prove
    for(int twp = 0; twp < num_witness_probes; ++twp) {
        /// Owner rank writes to file
        if( temporal_witness_probes[twp].getGlobalOwnerRank() == my_rank ) {

#if _WITNESS_XYZ_AVG_
            /// Reset auxiliary variables
            total_volume_local = 0.0;
            l2_x = 0.0; l2_y = 0.0; l2_z = 0.0;
            avg_v_bulk    = 0.0; avg_w_bulk     = 0.0; 
            l2_avg_u      = 0.0; l2_rmsf_u      = 0.0; l2_rmsf_v      = 0.0; l2_rmsf_w      = 0.0;
            l2_avg_u_comp = 0.0; l2_rmsf_u_comp = 0.0; l2_rmsf_v_comp = 0.0; l2_rmsf_w_comp = 0.0;
            l2_d_avg_u    = 0.0; l2_d_avg_v     = 0.0; l2_d_avg_w     = 0.0;   
            l2_d_rmsf_u   = 0.0; l2_d_rmsf_v    = 0.0; l2_d_rmsf_w    = 0.0;
            /// Calculate state value from averaging field values along local mesh of the TCP mpi process
            for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
                for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
                    for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                        /// Geometric stuff
                        delta_x = 0.5*( x_field[I1D(i+1,j,k)] - x_field[I1D(i-1,j,k)] ); 
                        delta_y = 0.5*( y_field[I1D(i,j+1,k)] - y_field[I1D(i,j-1,k)] ); 
                        delta_z = 0.5*( z_field[I1D(i,j,k+1)] - z_field[I1D(i,j,k-1)] );
                        delta_volume =  delta_x * delta_y * delta_z;
                        total_volume_local += delta_volume;
                        /// Spatial average
                        l2_x           += std::pow(x_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_y           += std::pow(y_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_z           += std::pow(z_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_avg_u       += std::pow(avg_u_field[I1D(i,j,k)],  2.0) * delta_volume;
                        avg_v_bulk     += avg_v_field[I1D(i,j,k)] * delta_volume;
                        avg_w_bulk     += avg_w_field[I1D(i,j,k)] * delta_volume;
                        l2_rmsf_u      += std::pow(rmsf_u_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_rmsf_v      += std::pow(rmsf_v_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_rmsf_w      += std::pow(rmsf_w_field[I1D(i,j,k)], 2.0) * delta_volume;
#if _RL_CONTROL_IS_SUPERVISED_
                        l2_avg_u_comp  += std::pow(avg_u_reference_field[I1D(i,j,k)],  2.0) * delta_volume;
                        l2_rmsf_u_comp += std::pow(rmsf_u_reference_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_rmsf_v_comp += std::pow(rmsf_v_reference_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_rmsf_w_comp += std::pow(rmsf_w_reference_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_d_avg_u     += std::pow(avg_u_field[I1D(i,j,k)]  - avg_u_reference_field[I1D(i,j,k)],  2.0) * delta_volume;
                        l2_d_avg_v     += std::pow(avg_v_field[I1D(i,j,k)]  - 0.0,  2.0) * delta_volume;
                        l2_d_avg_w     += std::pow(avg_w_field[I1D(i,j,k)]  - 0.0,  2.0) * delta_volume;
                        l2_d_rmsf_u    += std::pow(rmsf_u_field[I1D(i,j,k)] - rmsf_u_reference_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_d_rmsf_v    += std::pow(rmsf_v_field[I1D(i,j,k)] - rmsf_v_reference_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_d_rmsf_w    += std::pow(rmsf_w_field[I1D(i,j,k)] - rmsf_w_reference_field[I1D(i,j,k)], 2.0) * delta_volume;
#else   /// _RL_CONTROL_IS_SUPERVISED_ 0
                        l2_avg_u_comp  += std::pow(avg_u_previous_field[I1D(i,j,k)],  2.0) * delta_volume;
                        l2_rmsf_u_comp += std::pow(rmsf_u_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_rmsf_v_comp += std::pow(rmsf_v_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_rmsf_w_comp += std::pow(rmsf_w_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_d_avg_u     += std::pow(avg_u_field[I1D(i,j,k)]  - avg_u_previous_field[I1D(i,j,k)],  2.0) * delta_volume;
                        l2_d_avg_v     += std::pow(avg_v_field[I1D(i,j,k)]  - 0.0,  2.0) * delta_volume;
                        l2_d_avg_w     += std::pow(avg_w_field[I1D(i,j,k)]  - 0.0,  2.0) * delta_volume;
                        l2_d_rmsf_u    += std::pow(rmsf_u_field[I1D(i,j,k)] - rmsf_u_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_d_rmsf_v    += std::pow(rmsf_v_field[I1D(i,j,k)] - rmsf_v_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
                        l2_d_rmsf_w    += std::pow(rmsf_w_field[I1D(i,j,k)] - rmsf_w_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
#endif  /// _RL_CONTROL_IS_SUPERVISED_
                    }
                }
            }
            l2_x           = std::sqrt( l2_x / total_volume_local);
            l2_y           = std::sqrt( l2_y / total_volume_local);
            l2_z           = std::sqrt( l2_z / total_volume_local);
            l2_avg_u       = std::sqrt( l2_avg_u  / total_volume_local);
            l2_rmsf_u      = std::sqrt( l2_rmsf_u / total_volume_local);
            l2_rmsf_v      = std::sqrt( l2_rmsf_v / total_volume_local);
            l2_rmsf_w      = std::sqrt( l2_rmsf_w / total_volume_local);
            l2_avg_u_comp  = std::sqrt( l2_avg_u_comp  / total_volume_local);
            l2_rmsf_u_comp = std::sqrt( l2_rmsf_u_comp / total_volume_local);
            l2_rmsf_v_comp = std::sqrt( l2_rmsf_v_comp / total_volume_local);
            l2_rmsf_w_comp = std::sqrt( l2_rmsf_w_comp / total_volume_local);
            l2_d_avg_u     = std::sqrt( l2_d_avg_u  / total_volume_local);
            l2_d_avg_v     = std::sqrt( l2_d_avg_v  / total_volume_local);
            l2_d_avg_w     = std::sqrt( l2_d_avg_w  / total_volume_local);
            l2_d_rmsf_u    = std::sqrt( l2_d_rmsf_u / total_volume_local);
            l2_d_rmsf_v    = std::sqrt( l2_d_rmsf_v / total_volume_local);
            l2_d_rmsf_w    = std::sqrt( l2_d_rmsf_w / total_volume_local);
            /// Update state
            state_local[state_local_size2_counter]   = c1 * ( l2_d_avg_u  / l2_avg_u )  * std::copysign(1.0, l2_avg_u  - l2_avg_u_comp);     /// std::copysign(1.0, l2_avg_u - l2_avg_u_comp) = +1.0 if l2_avg_u >= l2_avg_u_comp, -1.0 otherwise
            state_local[state_local_size2_counter+1] = c2 * ( l2_d_avg_v )              * std::copysign(1.0, avg_v_bulk);
            state_local[state_local_size2_counter+2] = c3 * ( l2_d_avg_w )              * std::copysign(1.0, avg_w_bulk);
            state_local[state_local_size2_counter+3] = c4 * ( l2_d_rmsf_u / l2_rmsf_u ) * std::copysign(1.0, l2_rmsf_u - l2_rmsf_u_comp);
            state_local[state_local_size2_counter+4] = c5 * ( l2_d_rmsf_v / l2_rmsf_v ) * std::copysign(1.0, l2_rmsf_v - l2_rmsf_v_comp);
            state_local[state_local_size2_counter+5] = c6 * ( l2_d_rmsf_w / l2_rmsf_w ) * std::copysign(1.0, l2_rmsf_w - l2_rmsf_w_comp);
            state_local[state_local_size2_counter+6] = 2.0 * ( l2_x / L_x ) - 1.0;      /// range (-1,1)
            state_local[state_local_size2_counter+7] = 2.0 * ( l2_y / L_y ) - 1.0;
            state_local[state_local_size2_counter+8] = 2.0 * ( l2_z / L_z ) - 1.0;

#else /// _WITNESS_XYZ_AVG_ 0
            /// Get local indices i, j, k
            i_index = temporal_witness_probes[twp].getLocalIndexI(); 
            j_index = temporal_witness_probes[twp].getLocalIndexJ(); 
            k_index = temporal_witness_probes[twp].getLocalIndexK();
            /// Calculate state values
#if _RL_CONTROL_IS_SUPERVISED_
            state_local[state_local_size2_counter]   = ( avg_u_field[I1D(i_index,j_index,k_index)]  - avg_u_reference_field[I1D(i_index,j_index,k_index)] )  / ( avg_u_field[I1D(i_index,j_index,k_index)] );
            state_local[state_local_size2_counter+1] = ( rmsf_u_field[I1D(i_index,j_index,k_index)] - rmsf_u_reference_field[I1D(i_index,j_index,k_index)] ) / ( rmsf_u_field[I1D(i_index,j_index,k_index)] );
            state_local[state_local_size2_counter+2] = ( rmsf_v_field[I1D(i_index,j_index,k_index)] - rmsf_v_reference_field[I1D(i_index,j_index,k_index)] ) / ( rmsf_v_field[I1D(i_index,j_index,k_index)] );
            state_local[state_local_size2_counter+3] = ( rmsf_w_field[I1D(i_index,j_index,k_index)] - rmsf_w_reference_field[I1D(i_index,j_index,k_index)] ) / ( rmsf_w_field[I1D(i_index,j_index,k_index)] );
#else   /// _RL_CONTROL_IS_SUPERVISED_ 0
            state_local[state_local_size2_counter]   = ( avg_u_field[I1D(i_index,j_index,k_index)]  - avg_u_previous_field[I1D(i_index,j_index,k_index)] )   / ( avg_u_field[I1D(i_index,j_index,k_index)] );
            state_local[state_local_size2_counter+1] = ( rmsf_u_field[I1D(i_index,j_index,k_index)] - rmsf_u_previous_field[I1D(i_index,j_index,k_index)] )  / ( rmsf_u_field[I1D(i_index,j_index,k_index)] );
            state_local[state_local_size2_counter+2] = ( rmsf_v_field[I1D(i_index,j_index,k_index)] - rmsf_v_previous_field[I1D(i_index,j_index,k_index)] )  / ( rmsf_v_field[I1D(i_index,j_index,k_index)] );
            state_local[state_local_size2_counter+3] = ( rmsf_w_field[I1D(i_index,j_index,k_index)] - rmsf_w_previous_field[I1D(i_index,j_index,k_index)] )  / ( rmsf_w_field[I1D(i_index,j_index,k_index)] );
#endif  /// _RL_CONTROL_IS_SUPERVISED_
            state_local[state_local_size2_counter+4] = 2.0 * ( x_field[I1D(i_index,j_index,k_index)] / L_x ) - 1.0;
            state_local[state_local_size2_counter+5] = 2.0 * ( y_field[I1D(i_index,j_index,k_index)] / L_y ) - 1.0;
            state_local[state_local_size2_counter+6] = 2.0 * ( z_field[I1D(i_index,j_index,k_index)] / L_y ) - 1.0;

#endif /// _WITNESS_XYZ_AVG_

            /// Update local state counter
            state_local_size2_counter += state_dim;
        }
    }

    /// Check local state is updated correctly
    if (state_local_size2_counter != state_local_size2) {
        cerr << "Mismatch between state_local_size2_counter (" << state_local_size2_counter << ") != state_local_size2 (" << state_local_size2 << ")" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
}   

///////////////////////////////////////////////////////////////////////////////
/// Calculate reward local value (local to single mpi process, which corresponds to RL environment)
/// updates attribute 'reward_local'
void myRHEA::calculateReward() {
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    // Reward weight coefficients
    double b_param = 1.0;
    double c1 = 0.1 / actuation_period;
    double c2 = 0.02 / actuation_period;
    double c3 = 0.02 / actuation_period;
    double c4 = 0.02 / actuation_period;
    double c5 = 0.01 / actuation_period;
    double c6 = 0.01 / actuation_period;

    /// Initialize variables
    double l2_avg_u      = 0.0;
    double l2_rmsf_u     = 0.0;
    double l2_rmsf_v     = 0.0;
    double l2_rmsf_w     = 0.0;
    double l2_d_avg_u    = 0.0;
    double l2_d_avg_v    = 0.0;
    double l2_d_avg_w    = 0.0;
    double l2_d_rmsf_u   = 0.0;
    double l2_d_rmsf_v   = 0.0;
    double l2_d_rmsf_w   = 0.0;
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
                total_volume_local += delta_volume;
#if _RL_CONTROL_IS_SUPERVISED_
                /// Supervised error
                l2_d_avg_u         += std::pow(avg_u_field[I1D(i,j,k)]  - avg_u_reference_field[I1D(i,j,k)], 2.0)  * delta_volume;
                l2_d_avg_v         += std::pow(avg_v_field[I1D(i,j,k)]  - 0.0, 2.0)  * delta_volume;
                l2_d_avg_w         += std::pow(avg_w_field[I1D(i,j,k)]  - 0.0, 2.0)  * delta_volume;
                l2_d_rmsf_u        += std::pow(rmsf_u_field[I1D(i,j,k)] - rmsf_u_reference_field[I1D(i,j,k)], 2.0) * delta_volume;
                l2_d_rmsf_v        += std::pow(rmsf_v_field[I1D(i,j,k)] - rmsf_v_reference_field[I1D(i,j,k)], 2.0) * delta_volume;
                l2_d_rmsf_w        += std::pow(rmsf_w_field[I1D(i,j,k)] - rmsf_w_reference_field[I1D(i,j,k)], 2.0) * delta_volume;
#else   /// _RL_CONTROL_IS_SUPERVISED_ 0
                /// Unsupervised error
                l2_d_avg_u         += std::pow(avg_u_field[I1D(i,j,k)]  - avg_u_previous_field[I1D(i,j,k)], 2.0)  * delta_volume;
                l2_d_avg_v         += std::pow(avg_v_field[I1D(i,j,k)]  - 0.0, 2.0)  * delta_volume;
                l2_d_avg_w         += std::pow(avg_w_field[I1D(i,j,k)]  - 0.0, 2.0)  * delta_volume;
                l2_d_rmsf_u        += std::pow(rmsf_u_field[I1D(i,j,k)] - rmsf_u_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
                l2_d_rmsf_v        += std::pow(rmsf_v_field[I1D(i,j,k)] - rmsf_v_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
                l2_d_rmsf_w        += std::pow(rmsf_w_field[I1D(i,j,k)] - rmsf_w_previous_field[I1D(i,j,k)], 2.0) * delta_volume;
#endif
                l2_avg_u           += std::pow(avg_u_field[I1D(i,j,k)],  2.0) * delta_volume;
                l2_rmsf_u          += std::pow(rmsf_u_field[I1D(i,j,k)], 2.0) * delta_volume;
                l2_rmsf_v          += std::pow(rmsf_v_field[I1D(i,j,k)], 2.0) * delta_volume;
                l2_rmsf_w          += std::pow(rmsf_w_field[I1D(i,j,k)], 2.0) * delta_volume;
            }
        }
    }
    l2_avg_u     = std::sqrt( l2_avg_u     / total_volume_local);
    l2_rmsf_u    = std::sqrt( l2_rmsf_u    / total_volume_local);
    l2_rmsf_v    = std::sqrt( l2_rmsf_v    / total_volume_local);
    l2_rmsf_w    = std::sqrt( l2_rmsf_w    / total_volume_local);    
    l2_d_avg_u   = std::sqrt( l2_d_avg_u   / total_volume_local);
    l2_d_avg_v   = std::sqrt( l2_d_avg_v   / total_volume_local);
    l2_d_avg_w   = std::sqrt( l2_d_avg_w   / total_volume_local);
    l2_d_rmsf_u  = std::sqrt( l2_d_rmsf_u  / total_volume_local);
    l2_d_rmsf_v  = std::sqrt( l2_d_rmsf_v  / total_volume_local);
    l2_d_rmsf_w  = std::sqrt( l2_d_rmsf_w  / total_volume_local);
    reward_local = ( b_param - (   c1 * ( ( l2_d_avg_u )  / l2_avg_u ) \
                                 + c2 * ( ( l2_d_avg_v ) ) \
                                 + c3 * ( ( l2_d_avg_w ) ) \
                                 + c4 * ( ( l2_d_rmsf_u ) / l2_rmsf_u ) \
                                 + c5 * ( ( l2_d_rmsf_v ) / l2_rmsf_v ) \
                                 + c6 * ( ( l2_d_rmsf_w ) / l2_rmsf_w ) ) ) * 1.0; /// * ( (current_time - initial_episode_time) / episode_period );
    /// Debugging
    cout << "[myRHEA::calculateReward] [Rank " << my_rank << "] Local reward: " << reward_local << ", with reward terms: "
         << c1 * ( ( l2_d_avg_u )  / l2_avg_u )  << " " 
	     << c2 * ( ( l2_d_avg_v ) ) << " "
	     << c3 * ( ( l2_d_avg_w ) ) << " "
	     << c4 * ( ( l2_d_rmsf_u ) / l2_rmsf_u ) << " "
	     << c5 * ( ( l2_d_rmsf_v ) / l2_rmsf_v ) << " "
         << c6 * ( ( l2_d_rmsf_w ) / l2_rmsf_w ) << endl;
    if (my_rank == 0) {
        cout << "[myRHEA::calculateReward] Reward temporal coefficient: " << (current_time - initial_episode_time) / episode_period << endl;
    }

}

///////////////////////////////////////////////////////////////////////////////
/// (if _RL_CONTROL_IS_SUPERVISED_ 0) Update avg_u_previous_field, rmsf_u_previous_field, rmsf_v_previous_field, rmsf_w_previous_field used in calculateState and calculateReward
void myRHEA::updatePreviousActuationFields() {
#if _RL_CONTROL_IS_SUPERVISED_
    /// do nothing
#else
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
        t_action            = argv[4];  // 0.01 
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
