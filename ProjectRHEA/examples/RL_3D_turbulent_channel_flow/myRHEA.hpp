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

        /// SmartRedis
        SmartRedisManager manager;              /// TODO: should these vars be 'protected' or 'private'?
        std::string tag;
        std::string restart_data_file;
        double t_action;
        double t_episode;
        double t_begin_control;
        bool db_clustered;
        int n_rl_envs;
        int state_local_size;                   /// or nwitPar
        int action_global_size;                 /// or nRectangleControl

        void initRLParams(const string &tag, const string &restart_data_file, const string &t_action, const string &t_episode, const string &t_begin_control, const string &db_clustered);
        void initSmartRedis();
        void readWitnessPoints();



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

};

#endif /*_MY_RHEA_*/
