/// Code extracted from Fortran SOD2D_GITLAB/lib_sod2d/sources/mod_smartredis.f90 from branch 82-smartredis-integration
/// Code transformed to C++ class

#ifndef SMARTREDIS_H
#define SMARTREDIS_H

#include <smartredis/cpp_client.h> /// SmartRedis::Client
#include <mpi.h>
#include <vector>
#include <string>

class SmartRedisManager {
public:
    SmartRedisManager();
    ~SmartRedisManager();

    void init(int state_local_size2, int action_global_size2, int n_pseudo_envs2, const std::string& tag, bool db_clustered);
    void finalize();
    void writeState(const std::vector<double>& state_local, const std::string& key);
    void readAction(const std::string& key);
    void writeReward(const std::vector<double>& Ftau_neg, const std::string& key);
    void writeStepType(int step_type, const std::string& key);
    void writeTime(double time, const std::string& key);

private:
    SmartRedis::Client client;
    std::vector<int> state_sizes;
    std::vector<int> state_displs;
    std::vector<double> action_global;
    std::vector<double> action_global_previous;
    int state_local_size;
    int state_global_size;
    int action_global_size;
    int step_type_mod;
    int n_pseudo_envs;
};

#endif // SMARTREDIS_H