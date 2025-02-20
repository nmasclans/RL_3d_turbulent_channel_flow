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
        void outputCurrentStateDataRL( std::string path);

        /// Advance conserved variables in time
        void timeAdvanceConservedVariables() override;

        /// Output temporal point probes data
        void outputTemporalPointProbesData() override;

        /// Apply correction to streamwise bulk velocity (instantaneous vel.)
        void correctStreamwiseBulkVelocity();

    protected:

        // RL variables
        DistributedArray rl_f_rhou_field;
        DistributedArray rl_f_rhov_field;
        DistributedArray rl_f_rhow_field;
        DistributedArray rl_f_rhou_field_aux;          /// only if _SPACE_AVERAGE_RL_ACTION_ 1
        DistributedArray rl_f_rhov_field_aux;          /// only if _SPACE_AVERAGE_RL_ACTION_ 1
        DistributedArray rl_f_rhow_field_aux;          /// only if _SPACE_AVERAGE_RL_ACTION_ 1
        DistributedArray rl_f_rhou_field_prev_step;    /// only if _TEMPORAL_SMOOTHING_RL_ACTION_ 1
        DistributedArray rl_f_rhov_field_prev_step;    /// only if _TEMPORAL_SMOOTHING_RL_ACTION_ 1
        DistributedArray rl_f_rhow_field_prev_step;    /// only if _TEMPORAL_SMOOTHING_RL_ACTION_ 1
        DistributedArray rl_f_rhou_field_curr_step;    /// only if _TEMPORAL_SMOOTHING_RL_ACTION_ 1
        DistributedArray rl_f_rhov_field_curr_step;    /// only if _TEMPORAL_SMOOTHING_RL_ACTION_ 1
        DistributedArray rl_f_rhow_field_curr_step;    /// only if _TEMPORAL_SMOOTHING_RL_ACTION_ 1
        DistributedArray DeltaRxx_field;               /// 3-D field of DeltaRxx
        DistributedArray DeltaRxy_field;               /// 3-D field of DeltaRxy
        DistributedArray DeltaRxz_field;               /// 3-D field of DeltaRxz
        DistributedArray DeltaRyy_field;               /// 3-D field of DeltaRyy
        DistributedArray DeltaRyz_field;               /// 3-D field of DeltaRyz
        DistributedArray DeltaRzz_field;               /// 3-D field of DeltaRzz
        DistributedArray avg_u_reference_field;        /// only if _RL_CONTROL_IS_SUPERVISED_ 1
        DistributedArray rmsf_u_reference_field;       /// only if _RL_CONTROL_IS_SUPERVISED_ 1
        DistributedArray rmsf_v_reference_field;       /// only if _RL_CONTROL_IS_SUPERVISED_ 1
        DistributedArray rmsf_w_reference_field;       /// only if _RL_CONTROL_IS_SUPERVISED_ 1
        DistributedArray avg_u_previous_field;         /// only if _RL_CONTROL_IS_SUPERVISED_ 0
        DistributedArray rmsf_u_previous_field;        /// only if _RL_CONTROL_IS_SUPERVISED_ 0
        DistributedArray rmsf_v_previous_field;        /// only if _RL_CONTROL_IS_SUPERVISED_ 0
        DistributedArray rmsf_w_previous_field;        /// only if _RL_CONTROL_IS_SUPERVISED_ 0

        /// Witness points
        std::string witness_file;
        std::vector<TemporalPointProbe> temporal_witness_probes;
        std::vector<double> twp_x_positions;        /// only used if _WITNESS_XZ_PLANES_ 0
        std::vector<double> twp_y_positions;
        std::vector<double> twp_z_positions;        /// only used if _WITNESS_XZ_PLANES_ 0
        int num_witness_probes;

        /// Cubic control regions
        DistributedArray action_mask;
        std::vector<std::array<std::array<double, 3>, 4>> control_cubes_vertices; /// tensor size [num_control_cubes, num_coord, num_vertices] = [unknown, 3, 4] 
        std::vector<double> control_cubes_y_central;
        std::string control_cubes_file; 
        int num_control_cubes;
        int num_control_points;

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
        int n_rl_envs;
        int state_local_size2;                   /// or nwitPar
        int action_global_size2;                 /// or nRectangleControl
        double reward_local;
        std::vector<double> action_global;
        std::vector<double> state_local;

        void initRLParams(const string &tag, const string &restart_data_file, const string &t_action, const string &t_episode, const string &t_begin_control, const string &db_clustered, const string &global_step);
        void initSmartRedis();
        void readWitnessPoints();
        void preproceWitnessPoints();
        void readControlCubes();
        void getControlCubes();
        void initializeFromRestart();           /// override FlowSolverRHEA::initializeFromRestart method
        void updateState();
        void calculateReward();

    private:

        /// Eigen-decomposition
        void symmetricDiagonalize(const vector<vector<double>> &A, vector<vector<double>> &Q, vector<vector<double>> &D);
        void eigenDecomposition2Matrix(const vector<vector<double>> &D, const vector<vector<double>> &Q, vector<vector<double>> &A);
        void sortEigenDecomposition(vector<vector<double>> &Q, vector<vector<double>> &D);
        void matrixMultiplicate(const vector<vector<double>> &A, const vector<vector<double>> &B, vector<vector<double>> &C);
        
        /// Rij d.o.f. transformations
        void truncateAndNormalizeEigVal(vector<double> &lambda);
        void enforceRealizability(double &Rkk, double &phi1, double &phi2, double &phi3, double &xmap1, double &xmap2);
        void eigVect2eulerAngles(const vector<vector<double>> &Q, double &phi1, double &phi2, double &phi3);
        void eulerAngles2eigVect(const double &phi1, const double &phi2, const double &phi3, vector<vector<double>> &Q);        
        void eigVal2barycentricCoord(const vector<double> &lambda, double &xmap1, double &xmap2);
        void eigValMatrix2barycentricCoord(const vector<vector<double>> &D, double &xmap1, double &xmap2);
        void barycentricCoord2eigVal(const double &xmap1, const double &xmap2, vector<double> &lambda);
        void barycentricCoord2eigValMatrix(const double &xmap1, const double &xmap2, vector<vector<double>> &D);
        void Rijdof2matrix(const double &Rkk, const vector<vector<double>> &D, const vector<vector<double>> &Q, vector<vector<double>> &R);

        /// Helper functions
        double myDotProduct(const array<double,3> &v1, const array<double,3> &v2) {return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];}
        double myNorm(const array<double,3> &v){return std::sqrt(myDotProduct(v,v));}

};

#endif /*_MY_RHEA_*/
