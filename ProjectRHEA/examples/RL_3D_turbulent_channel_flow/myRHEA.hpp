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
               const std::string db_clustered="");              /// Parametrized constructor
        virtual ~myRHEA() {};									/// Destructor

	////////// SOLVER METHODS //////////
        
        /// Set initial conditions: u, v, w, P and T ... needs to be modified/overwritten according to the problem under consideration
        void setInitialConditions();

        /// Calculate rhou, rhov, rhow and rhoE source terms ... needs to be modified/overwritten according to the problem under consideration
        void calculateSourceTerms();

        /// Temporal hook function ... needs to be modified/overwritten according to the problem under consideration
        void temporalHookFunction();
	
 	    /// Calculate fixed time step
        void calculateTimeStep();

        /// Output current solver state data, in dedicated RL directory 'rhea_exp/output_data'
        void outputCurrentStateDataRL();

    protected:

        // RL variables
        DistributedArray DeltaRxx_field;        /// 3-D field of DeltaRxx
        DistributedArray DeltaRxy_field;        /// 3-D field of DeltaRxy
        DistributedArray DeltaRxz_field;        /// 3-D field of DeltaRxz
        DistributedArray DeltaRyy_field;        /// 3-D field of DeltaRyy
        DistributedArray DeltaRyz_field;        /// 3-D field of DeltaRyz
        DistributedArray DeltaRzz_field;        /// 3-D field of DeltaRzz

        /// Witness points
        std::string witness_file;
        std::vector<TemporalPointProbe> temporal_witness_probes;
        std::vector<double> twp_x_positions;
        std::vector<double> twp_y_positions;
        std::vector<double> twp_z_positions;
        int num_witness_probes;

        /// Cubic control regions
        DistributedArray action_mask;
        std::vector<std::array<std::array<double, 3>, 4>> control_cubes_vertices; /// tensor size [num_control_cubes, num_coord, num_vertices] = [unknown, 3, 4] 
        std::string control_cubes_file; 
        int num_control_cubes;
        int num_control_points;

        /// SmartRedis
        SmartRedisManager *manager;              /// TODO: should these vars be 'protected' or 'private'?
        std::string tag;
        std::string time_key;
        std::string step_type_key;
        std::string state_key;
        std::string action_key;
        std::string reward_key;
        double actuation_period;
        double begin_actuation_time;
        double previous_actuation_time;
        bool db_clustered;
        bool first_actuation;
        int n_rl_envs;
        int state_local_size2;                   /// or nwitPar
        int action_global_size2;                 /// or nRectangleControl
        int num_points_local;
        double avg_u_field_local;
        double avg_u_field_local_previous;
        double reward_local;
        std::vector<double> action_global;
        std::vector<double> action_global_previous;
        std::vector<double> action_global_instant;
        std::vector<double> state_local;

        void initRLParams(const string &tag, const string &restart_data_file, const string &t_action, const string &t_episode, const string &t_begin_control, const string &db_clustered);
        void initSmartRedis();
        void readWitnessPoints();
        void preproceWitnessPoints();
        void readControlCubes();
        void getControlCubes();
        void initializeFromRestart();           /// override FlowSolverRHEA::initializeFromRestart method
        void updateState();
        void calculateReward();
        void smoothControlFunction();
        void calculateInnerSizeTopo();

    private:

        /// Eigen-decomposition
        void symmetricDiagonalize(const vector<vector<double>> &A, vector<vector<double>> &Q, vector<vector<double>> &D);
        void eigenDecomposition2Matrix(const vector<vector<double>> &D, const vector<vector<double>> &Q, vector<vector<double>> &A);
        void sortEigenDecomposition(vector<vector<double>> &Q, vector<vector<double>> &D);
        void matrixMultiplicate(const vector<vector<double>> &A, const vector<vector<double>> &B, vector<vector<double>> &C);
        
        /// Rij d.o.f. transformations
        void truncateAndNormalizeEigVal(vector<double> &lambda);
        void enforceRealizability(double &Rkk, double &thetaZ, double &thetaY, double &thetaX, double &xmap1, double &xmap2);
        void eigVect2eulerAngles(const vector<vector<double>> &Q, double &thetaZ, double &thetaY, double &thetaX);
        void eulerAngles2eigVect(const double &thetaZ, const double &thetaY, const double &thetaX, vector<vector<double>> &Q);        
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
