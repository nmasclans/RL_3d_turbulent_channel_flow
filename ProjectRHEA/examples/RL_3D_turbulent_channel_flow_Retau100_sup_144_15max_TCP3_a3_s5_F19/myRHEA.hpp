#ifndef _MY_RHEA_
#define _MY_RHEA_

////////// INCLUDES //////////
#include "src/FlowSolverRHEA.hpp"
#include "client.h"
#include "SmartRedisManager.hpp"
#include <vector>

////////// USING //////////
using namespace std;

////////// myRHEA CLASS //////////
class myRHEA : public FlowSolverRHEA {
   
    public:

        ////////// CONSTRUCTORS & DESTRUCTOR //////////
        myRHEA(const std::string name_configuration_file, 
               const std::string tag="", 
               const std::string restart_data_file="", 
               const std::string t_action="", 
               const std::string t_episode="", 
               const std::string t_begin_control="",
               const std::string db_clustered="",
               const std::string global_step="");               /// Parametrized constructor
        virtual ~myRHEA() {};									/// Destructor

	////////// SOLVER METHODS //////////
        
        /// Execute (aggregated method) RHEA
        void execute() override;

        /// Set initial conditions: u, v, w, P and T ... needs to be modified/overwritten according to the problem under consideration
        void setInitialConditions() override;

        /// Calculate rhou, rhov, rhow and rhoE source terms ... needs to be modified/overwritten according to the problem under consideration
        void calculateSourceTerms() override;

        /// Temporal hook function ... needs to be modified/overwritten according to the problem under consideration
        void temporalHookFunction() override;
	
 	    /// Calculate fixed time step
        void calculateTimeStep() override;

        /// Output current solver state data, in dedicated RL directory '$RL_CASE_PATH/rhea_exp/output_data'
        void outputCurrentStateDataRL();

        /// Advance conserved variables in time
        void timeAdvanceConservedVariables() override;

        /// Output temporal point probes data
        void outputTemporalPointProbesData() override;

        /// Correct streamwise bulk velocity (with u_field) and/or check RL early episode termination (with avg_u_field)
        void manageStreamwiseBulkVelocity();

    protected:

        // RL field variables
        DistributedArray rl_f_rhou_field;
        DistributedArray rl_f_rhov_field;
        DistributedArray rl_f_rhow_field;
        DistributedArray rl_f_rhou_field_prev_step;    /// only if _TEMPORAL_SMOOTHING_RL_ACTION_ 1
        DistributedArray rl_f_rhov_field_prev_step;    /// only if _TEMPORAL_SMOOTHING_RL_ACTION_ 1
        DistributedArray rl_f_rhow_field_prev_step;    /// only if _TEMPORAL_SMOOTHING_RL_ACTION_ 1
        DistributedArray rl_f_rhou_field_curr_step;    /// only if _TEMPORAL_SMOOTHING_RL_ACTION_ 1
        DistributedArray rl_f_rhov_field_curr_step;    /// only if _TEMPORAL_SMOOTHING_RL_ACTION_ 1
        DistributedArray rl_f_rhow_field_curr_step;    /// only if _TEMPORAL_SMOOTHING_RL_ACTION_ 1
        DistributedArray d_DeltaRxj_j_field;           /// 3-D field of d_DeltaRxj_j_field
        DistributedArray d_DeltaRyj_j_field;           /// 3-D field of d_DeltaRyj_j_field
        DistributedArray d_DeltaRzj_j_field;           /// 3-D field of d_DeltaRzj_j_field
        DistributedArray avg_u_reference_field;        /// only if _RL_CONTROL_IS_SUPERVISED_ 1
        DistributedArray rmsf_u_reference_field;       /// only if _RL_CONTROL_IS_SUPERVISED_ 1
        DistributedArray rmsf_v_reference_field;       /// only if _RL_CONTROL_IS_SUPERVISED_ 1
        DistributedArray rmsf_w_reference_field;       /// only if _RL_CONTROL_IS_SUPERVISED_ 1
        DistributedArray avg_u_previous_field;         /// only if _RL_CONTROL_IS_SUPERVISED_ 0
        DistributedArray rmsf_u_previous_field;        /// only if _RL_CONTROL_IS_SUPERVISED_ 0
        DistributedArray rmsf_v_previous_field;        /// only if _RL_CONTROL_IS_SUPERVISED_ 0
        DistributedArray rmsf_w_previous_field;        /// only if _RL_CONTROL_IS_SUPERVISED_ 0

        /// Reward calculation
        double l2_err_avg_u_previous;                  /// only if _RL_CONTROL_IS_SUPERVISED_ 1
        double l2_rl_f_previous;                       /// only if _RL_CONTROL_IS_SUPERVISED_ 1

        /// Witness points
        std::string witness_file;
        std::vector<TemporalPointProbe> temporal_witness_probes;
        std::vector<double> twp_x_positions;        /// only used if _WITNESS_XZ_PLANES_ 0
        std::vector<double> twp_y_positions;
        std::vector<double> twp_z_positions;        /// only used if _WITNESS_XZ_PLANES_ 0
        int num_witness_probes;

        /// Control points
        std::string control_file;
        std::vector<TemporalPointProbe> temporal_control_probes;
        std::vector<double> tcp_x_positions;
        std::vector<double> tcp_y_positions;
        std::vector<double> tcp_z_positions;
        int num_control_probes;

        /// SmartRedis
        SmartRedisManager *manager;              /// TODO: should these vars be 'protected' or 'private'?
        std::string tag;
        std::string global_step;
        std::string time_key;
        std::string step_type_key;
        std::string state_key;
        std::string action_key;
        std::string reward_key;
        double actuation_period;
        double begin_actuation_time;
        double previous_actuation_time;
        bool db_clustered;
        bool first_actuation_time_done;
        bool first_actuation_period_done;
        bool last_communication;
        int n_rl_envs;
        int state_local_size2;
        int action_global_size2;
        double reward_local;
        std::vector<double> action_global;
        std::vector<double> state_local;

        // RL early episode termination variable
        bool rl_early_episode_termination;            /// only if _RL_EARLY_EPISODE_TERMINATION_FUNC_U_BULK_

        void initRLParams(const string &tag, const string &restart_data_file, const string &t_action, const string &t_episode, const string &t_begin_control, const string &db_clustered, const string &global_step);
        void initSmartRedis();
        void readWitnessPoints();
        void preproceWitnessPoints();
        void readControlPoints();
        void preproceControlPoints();
        void initializeFromRestart();           /// override FlowSolverRHEA::initializeFromRestart method
        void updateState();
        void calculateReward();

    private:

};

#endif /*_MY_RHEA_*/
