#include "myRHEA.hpp"

#ifdef _OPENACC
#include <openacc.h>
#endif
#include <numeric>
#include <algorithm>
#include <cmath>        // std::pow

using namespace std;

////////// COMPILATION DIRECTIVES //////////
/// TODO: move these env parameters to an .h file for the _ACTIVE_CONTROL_BODY_FORCE_ to be included by SmartRedisManager.cpp & .h files 
#define _FEEDBACK_LOOP_BODY_FORCE_ 0				/// Activate feedback loop for the body force moving the flow
#define _ACTIVE_CONTROL_BODY_FORCE_ 1               /// Activate active control for the body force
#define _FIXED_TIME_STEP_ 1                         /// Activate fixed time step
#define _REGULARIZE_RL_ACTION_ 1                    /// Activate regularization for RL action or RL source term w.r.t. momentum equation rhs 
#define _RL_CONTROL_IS_SUPERVISED_ 1

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

#if _FEEDBACK_LOOP_BODY_FORCE_
/// Estimated uniform body force to drive the flow
double controller_output = tau_w/delta;			    /// Initialize controller output
double controller_error  = 0.0;			        	/// Initialize controller error
double controller_K_p    = 1.0e-1;		        	/// Controller proportional gain
#endif

#if _ACTIVE_CONTROL_BODY_FORCE_
int action_dim = 6;
/// eigen-values barycentric map coordinates - corners of realizable region
double EPS     = numeric_limits<double>::epsilon();
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
#if _RL_CONTROL_IS_SUPERVISED_
// Reference profiles (_ALL_ coordinates, including boundaries)
double rmsf_u_reference_profile[] = {   /// only inner points
    0.51150471, 1.45835662, 2.11404174, 2.44017584, 2.54703401,
    2.53763372, 2.47299457, 2.38526539, 2.29087569, 2.19823627, 2.11135341,
    2.03182011, 1.9603063,  1.89640688, 1.83874786, 1.78562372, 1.73589045,
    1.68926942, 1.64582258, 1.60539581, 1.56771906, 1.53270985, 1.50052829,
    1.47134978, 1.44525996, 1.42221959, 1.40212976, 1.38502714, 1.37109622,
    1.36058096, 1.35369685, 1.35057738, 1.35123427, 1.3555985,  1.36360569,
    1.37521577, 1.39033546, 1.40873267, 1.43011968, 1.4542445,  1.4809011,
    1.50987913, 1.54098663, 1.57436436, 1.6104301,  1.64961678, 1.69243293,
    1.73938057, 1.79090939, 1.84761624, 1.91010811, 1.97861765, 2.05301811,
    2.13364007, 2.22065781, 2.31311318, 2.40742388, 2.49523502, 2.55948262,
    2.56722702, 2.4570923,  2.12806339, 1.47013439, 0.51661591,
};
double rmsf_v_reference_profile[] = {   /// only inner points
    0.01650155, 0.08181275, 0.16729574, 0.24822572, 0.32499139,
    0.39233852, 0.45178113, 0.50177154, 0.54370385, 0.57736409, 0.60390731,
    0.62361946, 0.63748449, 0.6460212,  0.65020967, 0.65063157, 0.64800714,
    0.64280504, 0.63565974, 0.62707176, 0.61758074, 0.60754187, 0.59728562,
    0.58702418, 0.57698958, 0.56738207, 0.55846834, 0.55051633, 0.54382089,
    0.53863505, 0.53516008, 0.5334955,  0.53365564, 0.53558278, 0.5391632,
    0.54424361, 0.55062734, 0.5581099,  0.56645285, 0.57543321, 0.58480146,
    0.59433624, 0.60376919, 0.61284912, 0.62120998, 0.6285256,  0.6343766,
    0.63843548, 0.64019427, 0.63920571, 0.6346971,  0.62610772, 0.61264217,
    0.59382854, 0.56853613, 0.53636736, 0.49601822, 0.44754995, 0.38951674,
    0.32335627, 0.24746871, 0.1671392,  0.08202398, 0.01678448,
};
double rmsf_w_reference_profile[] = {   /// only inner points
    0.20571631, 0.41576762, 0.54813711, 0.63775117, 0.70223838,
    0.74644498, 0.77708618, 0.79799458, 0.81192009, 0.8205753,  0.824915,
    0.82531996, 0.82188959, 0.81515366, 0.8056561,  0.7941513,  0.78086151,
    0.7658519,  0.74924255, 0.73137574, 0.71271056, 0.6938646,  0.67540599,
    0.65780804, 0.64143427, 0.62666719, 0.61382452, 0.60310661, 0.59457335,
    0.58818177, 0.58387553, 0.58174093, 0.58199005, 0.58478415, 0.59008226,
    0.59765905, 0.60728489, 0.6188314,  0.63219725, 0.64709561, 0.66309653,
    0.6797968,  0.69692535, 0.71425501, 0.73158266, 0.74859145, 0.76483269,
    0.77968062, 0.79265889, 0.80346704, 0.81196701, 0.81778646, 0.82069221,
    0.82020542, 0.81579751, 0.80669046, 0.79225442, 0.77132631, 0.74095518,
    0.69659095, 0.63190866, 0.54316025, 0.41282767, 0.20502068,
};
#endif
#endif


////////// myRHEA CLASS //////////
