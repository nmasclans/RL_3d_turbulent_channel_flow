#include "myRHEA.hpp"

#ifdef _OPENACC
#include <openacc.h>
#endif
#include <numeric>

using namespace std;

////////// COMPILATION DIRECTIVES //////////
/// TODO: move these env parameters to an .h file for the _ACTIVE_CONTROL_BODY_FORCE_ to be included by SmartRedisManager.cpp & .h files 
#define _FEEDBACK_LOOP_BODY_FORCE_ 0				/// Activate feedback loop for the body force moving the flow
#define _ACTIVE_CONTROL_BODY_FORCE_ 1               /// Activate active control for the body force
#define _FIXED_TIME_STEP_ 1                         /// Activate fixed time step

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

const double fixed_time_step = 1.0e-5;				/// Fixed time step
const int cout_precision = 10;		                /// Output precision (fixed) 

#if _FEEDBACK_LOOP_BODY_FORCE_
/// Estimated uniform body force to drive the flow
double controller_output = tau_w/delta;			    /// Initialize controller output
double controller_error  = 0.0;			        	/// Initialize controller error
double controller_K_p    = 1.0e-1;		        	/// Controller proportional gain
#endif

#if _ACTIVE_CONTROL_BODY_FORCE_
/// eigen-values barycentric map coordinates - corners of realizable region
double EPS         = numeric_limits<double>::epsilon();
/* Baricentric map coordinates, source: https://en.wikipedia.org/wiki/Barycentric_coordinate_system
 * Transform xmap -> lambda: lambda0,1 = Tinv * (xmap0,1 - t); lambda2 = 1 - lambda0 - lambda1
 * Transform lambda -> xmap: xmap0,1   = T * lambda0,1 + t   
 *                                     = lambda0 * x1c + lambda1 * x2c + lambda2 * x3c
 * Realizability Condition (lambda): 0<=lambda_i<=1, sum(lambda_i)=1
 * Realizability Condition (xmap):   xmap coord inside barycentric map triangle, defined by x1c, x2c, x3c */
vector<double> x1c = {1.0, 0.0};                // corner x1c
vector<double> x2c = {0.0, 0.0};                // corner x2c
vector<double> x3c = {0.5, sqrt(3.0) / 2.0};    // corner x3c
vector<double> t   = {x3c[0], x3c[1]};
vector<vector<double>> T = {
    {x1c[0] - x3c[0], x2c[0] - x3c[0] },        // row 1, T[0][:]
    {x1c[1] - x3c[1], x2c[1] - x3c[1]},         // row 2, T[1][:]
};
double Tdet = T[0][0] * T[1][1] - T[0][1] * T[1][0];
vector<vector<double>> Tinv = {
    {  T[1][1] / Tdet, - T[0][1] / Tdet},       // row 1, Tinv[0][:]
    {- T[1][0] / Tdet,   T[0][0] / Tdet},       // row 2, Tinv[1][:]
};
/// Kronecker delta
vector<vector<double>> Deltaij = {
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},
};
#endif

////////// myRHEA CLASS //////////

myRHEA::myRHEA(const string name_configuration_file, const string tag, const string restart_data_file, const string t_action, const string t_episode, const string t_begin_control, const string db_clustered) : FlowSolverRHEA(name_configuration_file) {

#if _ACTIVE_CONTROL_BODY_FORCE_
    DeltaRxx_field.setTopology(topo, "DeltaRxx");
    DeltaRxy_field.setTopology(topo, "DeltaRxy");
    DeltaRxz_field.setTopology(topo, "DeltaRxz");
    DeltaRyy_field.setTopology(topo, "DeltaRyy");
    DeltaRyz_field.setTopology(topo, "DeltaRyz");
    DeltaRzz_field.setTopology(topo, "DeltaRzz");
    action_mask.setTopology(topo, "action_mask");
    timers->createTimer( "rl_smartredis_communications" );
    timers->createTimer( "update_rl_DeltaRij" );
    timers->createTimer( "update_rl_control_term" );

    initRLParams(tag, restart_data_file, t_action, t_episode, t_begin_control, db_clustered);
    initSmartRedis();
#endif

};


void myRHEA::initRLParams(const string &tag, const string &restart_data_file, const string &t_action, const string &t_episode, const string &t_begin_control, const string &db_clustered) {

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
    }

    /// String arguments
    this->tag                = tag;
    this->restart_data_file  = restart_data_file;            // updated variable from previously defined value in FlowSolverRHEA::readConfigurationFile
    /// Double arguments
    try {
        // Set actuation attributes 
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
    this->witness_file = "witness8.txt";
    this->control_cubes_file = "cubeControl8.txt";
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
    first_actuation = true;
    avg_u_field_local = 0.0;
    avg_u_field_local_previous = 0.0;
    action_global.resize(action_global_size2);
    action_global_previous.resize(action_global_size2);
    action_global_instant.resize(action_global_size2);
    state_local.resize(state_local_size2);
    std::fill(action_global.begin(), action_global.end(), 0.0);
    std::fill(action_global_previous.begin(), action_global_previous.end(), 0.0);
    std::fill(action_global_instant.begin(), action_global_instant.end(), 0.0);
    std::fill(state_local.begin(), state_local.end(), 0.0);

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
    manager = new SmartRedisManager(state_local_size2, action_global_size2, n_rl_envs, tag, db_clustered);

    /// Write step type = 1
    manager->writeStepType(1, "ensemble_" + tag + ".step_type");

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


    /// Check if new action is needed
    if (current_time > begin_actuation_time) {

        
        /// TODO: it seems to me with the couts that this loop is made 3 times per time instant (i get x3 couts than expected)... why?
        if (my_rank == 0 && first_actuation) {
            first_actuation = false;    /// just cout once, for the first actuation
            calculateReward();          /// only for initializing 'avg_u_field_local_previous' non-zero for the next reward calculation
            cout << endl << endl << "[myRHEA::calculateSourceTerms] RL control is activated at current time (" << scientific << current_time << ") " << "> time begin control (" << scientific << begin_actuation_time << ")" << endl;
        } 
        /// TODO: remove cout for less debugging
        if (my_rank == 0) {
            cout << "[myRHEA::calculateSourceTerms] RL control applied at time " << current_time << endl;
        }
        
        /// Check if new action is needed
        if (current_time - previous_actuation_time >= actuation_period) {
            
            timers->start( "rl_smartredis_communications" );

            /// Logging
            if (my_rank == 0) {
                cout << "[myRHEA::calculateSourceTerms] Performing SmartRedis communications (state, action, reward) at time instant " << current_time << endl;
            }

            /// Save old action values and time - useful for interpolating to new action
            /// TODO: use action_global and action_global_previous for something (output writing) or delete them
            action_global_previous  = action_global;     // a copy is made
            previous_actuation_time = previous_actuation_time + actuation_period;

            // SmartRedis communications 
            updateState();
            manager->writeState(state_local, state_key);
            calculateReward();                              /// update reward_local attribute
            manager->writeReward(reward_local, reward_key);
            manager->readAction(action_key);
            action_global = manager->getActionGlobal();     /// action_global: vector<double> of size action_global_size2 = n_rl_envs (currently only 1 action variable per rl env.)
            manager->writeTime(current_time, time_key);
            /// Update step size (from 1) to 0 if the next time that we require actuation value is the last one
            if (current_time + 2.0 * actuation_period > final_time) {
                manager->writeStepType(0, step_type_key);
            }

            timers->stop( "rl_smartredis_communications" );

        }

        timers->start( "update_rl_DeltaRij" );

        /// Calculate smooth action 
        smoothControlFunction();        /// updates 'action_global_instant'

        /// Initialize variables
        double Rkk, thetaZ, thetaY, thetaX, xmap1, xmap2; 
        double DeltaRkk, DeltaThetaZ, DeltaThetaY, DeltaThetaX, DeltaXmap1, DeltaXmap2; 
        double d_DeltaRxx_x, d_DeltaRxy_x, d_DeltaRxz_x, d_DeltaRxy_y, d_DeltaRyy_y, d_DeltaRyz_y, d_DeltaRxz_z, d_DeltaRyz_z, d_DeltaRzz_z;
        double delta_x, delta_y, delta_z;
        double Rkk_inv, Akk;
        bool   isNegligibleAction, isNegligibleRkk;
        bool actionIsApplied = false;
        size_t action_idx;
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
                        /// TODO: implement RL action for several variables!
                        action_idx = static_cast<size_t>(action_mask[I1D(i,j,k)]) - 1;
                        DeltaRkk    = action_global_instant[action_idx];
                        DeltaThetaZ = 0.0;
                        DeltaThetaY = 0.0;
                        DeltaThetaX = 0.0;
                        DeltaXmap1  = 0.0;
                        DeltaXmap2  = 0.0;

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
                            actionIsApplied = true;

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

                            /// Build perturbed Rij d.o.f.
                            Rkk    += DeltaRkk;
                            thetaZ += DeltaThetaZ;
                            thetaY += DeltaThetaY;
                            thetaX += DeltaThetaX;
                            xmap1  += DeltaXmap1;
                            xmap2  += DeltaXmap2;

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

        timers->stop( "update_rl_DeltaRij" );

        /// Check action is applied and non-neglegible in some control point for each RL environment / control cube / mpi process
        /// TODO: remove cout if not necessary in future debugging
        /// if (!actionIsApplied) {
        ///     cout << "Warning: Rank " << my_rank << " applied at time: " << current_time << endl;
        /// }               
        
        /// Calculate and incorporate perturbation load F = \partial DeltaRij / \partial xj
        timers->start( "update_rl_control_term" );
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
                    f_rhou_field[I1D(i,j,k)] += ( -1.0 ) * rho_field[I1D(i,j,k)] * ( d_DeltaRxx_x + d_DeltaRxy_y + d_DeltaRxz_z );
                    f_rhov_field[I1D(i,j,k)] += ( -1.0 ) * rho_field[I1D(i,j,k)] * ( d_DeltaRxy_x + d_DeltaRyy_y + d_DeltaRyz_z );
                    f_rhow_field[I1D(i,j,k)] += ( -1.0 ) * rho_field[I1D(i,j,k)] * ( d_DeltaRxz_x + d_DeltaRyz_y + d_DeltaRzz_z );
                    f_rhoE_field[I1D(i,j,k)] += 0.0;
                }
            }
        }
        timers->stop( "update_rl_control_term" );

    } else {

        if (my_rank == 0) {
            cout << "[myRHEA::calculateSourceTerms] RL Control is NOT applied yet, as current time (" << scientific << current_time << ") " << "< time begin control (" << scientific << begin_actuation_time << ")" << endl;
        }

    }
#endif

    /// Update halo values
    //f_rhou_field.update();
    //f_rhov_field.update();
    //f_rhow_field.update();
    //f_rhoE_field.update();

};

void myRHEA::temporalHookFunction() {

    /// Custom temporal hook function

};

void myRHEA::calculateTimeStep() {

#if _FIXED_TIME_STEP_
    /// Set new time step
    delta_t = fixed_time_step;
#else
    FlowSolverRHEA::calculateTimeStep();
#endif

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
        /// TODO: delete cout if not necessary for future debugging
        /// cout << "Candidate witness probes:" << endl;
        /// for (int i=0; i<num_witness_probes; i++) {
        ///     cout << twp_x_positions.at(i) << ", " << twp_y_positions.at(i) << ", " << twp_z_positions.at(i) << endl;
        /// }
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

    // Check successfull broadcast, TODO: delete cout if not necessary for future debugging 
    /// if (my_rank != 0) {
    ///     cout << "Mpi with rank " << my_rank << " has received " << num_witness_probes << " witness probes" << endl;
    ///     for (int i=0; i<num_witness_probes; i++) {
    ///         cout << twp_x_positions.at(i) << ", " << twp_y_positions.at(i) << ", " << twp_z_positions.at(i) << endl;
    ///     }
    /// }
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
            /// TODO: delete cout if not used in future debugging
            /// int i_index, j_index, k_index;
            /// /// Get local indices i, j, k
		    /// i_index = temporal_witness_probes[twp].getLocalIndexI(); 
		    /// j_index = temporal_witness_probes[twp].getLocalIndexJ(); 
		    /// k_index = temporal_witness_probes[twp].getLocalIndexK();
            /// /// Get coordinates of witness probe
            /// cout << "Witness probe: " << twp << " , mpi rank: " << my_rank << ", coord: " 
            ///      << x_field[I1D(i_index,j_index,k_index)] << " " 
            ///      << y_field[I1D(i_index,j_index,k_index)] << " " 
            ///      << z_field[I1D(i_index,j_index,k_index)] << endl;
        }
    }
    this->state_local_size2 = state_local_size2_counter;  // each mpi process updates attribute 'state_local_size2'
    cout << "Rank " << my_rank << " has " << state_local_size2 << " local witness points" << endl;
    cout.flush();
    MPI_Barrier(MPI_COMM_WORLD); /// TODO: only here for cout debugging purposes, delete if wanted

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

    /// Check successfull data reading and broadcast, TODO: delete cout if not necessary for future debugging
    /// cout << "Mpi proc. with rank " << my_rank << " has received control cubes vertices:" << endl;
    /// for (int i = 0; i < num_control_cubes; ++i) {
    ///     cout << "   Cube " << i + 1 << " vertices:" << endl;
    ///     for (int j = 0; j < 4; ++j) {                       /// loop along the 4 vertices
    ///         cout << "      Vertex " << j + 1 << ": "
    ///                 << control_cubes_vertices[i][j][0] << ", "
    ///                 << control_cubes_vertices[i][j][1] << ", "
    ///                 << control_cubes_vertices[i][j][2] << endl;
    ///     }
    /// }
    /// cout.flush();
    /// MPI_Barrier(MPI_COMM_WORLD); /// TODO: only here for cout debugging purposes, delete if wanted

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
                        /// log point information, TODO: delete cout if not necessary for future debugging 
                        /// cout << "[myRHEA::getControlCubes] rank: " << my_rank << ", cube: " << icube << ", action mask: " << action_mask[I1D(i,j,k)] << ", found control point: " 
                        ///      << x_field[I1D(i,j,k)] << ", " << y_field[I1D(i,j,k)] << ", " << z_field[I1D(i,j,k)] << endl;
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
            action_global_size2++;
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
    cout << "Rank " << my_rank << " has RL environment / mpi process domain boundaries: "  
         << "x in (" << x_field[I1D(topo->iter_common[_INNER_][_INIX_],0,0)] << ", " << x_field[I1D(topo->iter_common[_INNER_][_ENDX_],0,0)] << "), "
         << "y in (" << y_field[I1D(0,topo->iter_common[_INNER_][_INIY_],0)] << ", " << y_field[I1D(0,topo->iter_common[_INNER_][_ENDY_],0)] << "), "
         << "z in (" << z_field[I1D(0,0,topo->iter_common[_INNER_][_INIZ_])] << ", " << z_field[I1D(0,0,topo->iter_common[_INNER_][_ENDZ_])] << ")" << endl;
    
    /// Check 1: num. of control cubes == num. mpi processes, and mpi processes must be distributed only along y-coordinate
    this->n_rl_envs = action_global_size2;
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

    /// Logging
    if (my_rank == 0) {
        cout << "[myRHEA::getControlCubes] Total number of control points: " << num_control_points << endl;
        cout << "[myRHEA::getControlCubes] Action global size (num. cubes with at least 1 control point): " << action_global_size2 << endl;
        cout << "[myRHEA::getControlCubes] Action global size corresponds to the Number of RL Environments: " << n_rl_envs << endl; 
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
            /// Get state data: averaged u-velocity
            state_local[state_local_size2_counter] = avg_u_field[I1D(i_index,j_index,k_index)];
            state_local_size2_counter += 1;
            /// Logging, TODO: remove logging lines if not used in the future
            /// cout << "Rank " << my_rank << " has witness point #" << twp << " at coord: [" 
            ///      << x_field[I1D(i_index,j_index,k_index)] << ", " 
            ///      << y_field[I1D(i_index,j_index,k_index)] << ", " 
            ///      << z_field[I1D(i_index,j_index,k_index)] << "], "
            ///      << "and u_field value: " << avg_u_field[I1D(i_index,j_index,k_index)] << endl;
        }
    }
    
    /// Logging // TODO: remove logging if not necessary for future debugging
    cout << "[myRHEA::calculateSourceTerms] Rank " << my_rank << " has local state of size " << state_local_size2 << " and values: ";
    for (int ii = 0; ii<state_local_size2; ii++) {  
        cout << state_local[ii] << " ";
    }
    cout << endl << flush;
}

///////////////////////////////////////////////////////////////////////////////

/// Calculate reward local value (local to single mpi process, which corresponds to RL environment)
/// -> updates attributes: reward_local, avg_u_field_local, avg_u_field_local_previous
void myRHEA::calculateReward() {

    /// Initialize variables
    int points_counter_local = 0;
    avg_u_field_local = 0.0; 

    /// Calculate avg_u_field_rl_envs, the temporal-average of u_field space-averaged over each rl environment
    for(int i = topo->iter_common[_INNER_][_INIX_]; i <= topo->iter_common[_INNER_][_ENDX_]; i++) {
        for(int j = topo->iter_common[_INNER_][_INIY_]; j <= topo->iter_common[_INNER_][_ENDY_]; j++) {
            for(int k = topo->iter_common[_INNER_][_INIZ_]; k <= topo->iter_common[_INNER_][_ENDZ_]; k++) {
                avg_u_field_local += avg_u_field[I1D(i,j,k)];
                points_counter_local++;       // TODO: perhaps fill this counter just one in the initRLParams, more computationally efficient
            }
        }
    }
    avg_u_field_local /= points_counter_local;
    reward_local = std::abs(avg_u_field_local - avg_u_field_local_previous); 

    /// Store avg_u_field_local for next reward calculation
    avg_u_field_local_previous = avg_u_field_local;

    /// Logging
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    stringstream ss;
    ss << "[myRHEA::calculateReward] Rank " << my_rank << " has num. points: " << points_counter_local
       << ", local avg_u_field: " << avg_u_field_local
       << ", local reward: " << reward_local;
    cout << ss.str() << endl;
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
    string configuration_file, tag, restart_data_file, t_action, t_episode, t_begin_control, db_clustered;
    if (argc >= 8 ) {
        /// Extract and validate input arguments
        configuration_file  = argv[1];
        tag                 = argv[2];
        restart_data_file   = argv[3];
        t_action            = argv[4];  // 0.0001 
        t_episode           = argv[5];  // 1.0
        t_begin_control     = argv[6];  // 0.0
        db_clustered        = argv[7];  // False
    } else {
        cerr << "Proper usage: RHEA.exe configuration_file.yaml <tag> <restart_data_file> <t_action> <t_episode> <t_begin_control> <db_clustered>" << endl;
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    /// Construct my RHEA
    myRHEA my_RHEA( configuration_file, tag, restart_data_file, t_action, t_episode, t_begin_control, db_clustered );

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
