/*
Code extracted from Fortran SOD2D_GITLAB/lib_sod2d/sources/mod_smartredis.f90 from branch 82-smartredis-integration
Code transformed to C++ class
*/ 

#ifndef SMARTREDISMANAGER_H
#define SMARTREDISMANAGER_H

#include "client.h"
#include <vector>
#include <string>
#include <mpi.h>

class SmartRedisManager {
public:
    SmartRedisManager();
    SmartRedisManager(int state_local_size2, int action_global_size2, int n_pseudo_envs2, const std::string& tag, bool db_clustered);
    ~SmartRedisManager();
    
    void writeState(const std::vector<double>& state_local, const std::string& key);
    void readState(const std::string& key);
    void readAction(const std::string& key);
    void writeReward(const double reward, const std::string& key);
    void writeStepType(const int step_type, const std::string& key);
    void writeTime(const double time, const std::string& key);

    std::vector<double> getStateGlobal();
    std::vector<double> getActionGlobal();

    void printDatabaseContent();

private:
                 
    std::unique_ptr<SmartRedis::Client> client; // Use a unique pointer for conditional initialization
    std::vector<int> state_sizes;
    std::vector<int> state_displs;
    std::vector<double> state_global;
    std::vector<double> action_global;
    int state_local_size;
    int state_global_size;
    int action_global_size;
    std::vector<size_t> state_local_size_vec;
    std::vector<size_t> state_global_size_vec;
    std::vector<size_t> action_global_size_vec;
    int step_type_mod;
    int n_pseudo_envs;
    int mpi_size;
    int mpi_rank;
    int read_interval;
    int read_tries;
};

#endif /* SMARTREDISMANAGER_H */