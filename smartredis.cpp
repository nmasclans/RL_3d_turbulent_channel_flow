#include "smartredis.h"
#include <iostream>

/// TODO: C++: The C++ class SmartRedisManager initializes MPI in the constructor and finalizes it in the destructor. This might be redundant if MPI is already initialized elsewhere in your application.

SmartRedisManager::SmartRedisManager() {
    /* MPI Initialized in myRHEA main
    // Constructor can initialize MPI or other necessary components
    MPI_Init(NULL, NULL);
    */
}

SmartRedisManager::~SmartRedisManager() {
    /* MPI Finalized in myRHEA main
    // Destructor should finalize MPI and clean up resources
    MPI_Finalize();
    */
    /// Finalize client
    finalize()
}

void SmartRedisManager::init(int state_local_size2, int action_global_size2, int n_pseudo_envs2, const std::string& tag, bool db_clustered) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    state_sizes.resize(mpi_size);
    state_displs.resize(mpi_size);
    action_global.resize(action_global_size2, 0.0);
    action_global_previous.resize(action_global_size2, 0.0);

    // Gather the individual sizes to get total size and offsets in root process (0)
    /*  sendbuf(const void*) = state_local_size2 : pointer to the data that each process sends to the root process,
            in this case the size of local state array of each mpi process
        sendcount(int) = 1 : number of elements to send from each process
        sendtype(MPI_Datatype) = MPI_INT : MPI datatype for integers 
        recvbuf(void*) = state_sizes.data() : pointer to the buffer where the gathered data will be stored in the root process,
            which will hold the gathered sizes from all processes
        recvcount(int) = 1: number of elements the root process will receive from each process 
        recvtype(MPI_Datatype) = MPI_int: datatype of the elements received by the root process
        root(int) = 0 : rank of the root process that will receive the gathered data
        comm(MPI_Comm) = MPI_COMM_WORLD: communicator that defines the group of process involved in the communication, 
            which for MPI_COMM_WORLD includes all MPI process in the application
    */
    MPI_Gather(&state_local_size2, 1, MPI_INT, state_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute displacements - global state idxs corresponding to the local state of each MPI process
    if (mpi_rank == 0) {
        int state_counter = 0;
        for (int i = 0; i < mpi_size; ++i) {
            state_displs[i] = state_counter;
            state_counter += state_sizes[i];
        }
    }

    /// Store local parameters (input arguments of each process) to class variables
    n_pseudo_envs = n_pseudo_envs2;
    action_global_size = action_global_size2;
    state_local_size = state_local_size2;
    /// Sum the local state_local_size of each process in MPI_COMM_WORLD to obtain state_global_size, 
    /// with result state_global_size distributed to all processes in MPI_COMM_WORLD
    MPI_Allreduce(&state_local_size, &state_global_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    /// Init client (only root process!), and write global state and action sizes into DB.
    if (mpi_rank == 0) {
        /// Initialize client
        /// if db_slustered, the client is initialize to utilize a SmartRedis Orchestrator in cluster configuration
        try {
            client.Initialize(db_clustered);
        } catch (const SmartRedis::Exception& ex) {
            std::cerr << "Error putting tensor" << ex.what() << std::endl; 
        }

        /// Write global state and action sizes into DB
        /* (SmartRedis C++ API)
        https://www.craylabs.org/docs/api/smartredis_api.html
        void put_tensor(const std::string &name, 
                        const void *data, 
                        const std::vector<size_t> &dims,
                        const SRTensorType type, 
                        const SRMemoryLayout mem_layout)
        */
        try {
            client.put_tensor("state_size", &state_global_size, {1}, SRTensorType::INT64, SRMemoryLayout::CONTIGUOUS);
            client.put_tensor("action_size", action_global_size, {1}, SRTensorType::INT64, SRMemoryLayout::CONTIGUOUS);
        } catch (const SmartRedis::Exception& ex) {
            std::cerr << "Error putting tensor" << ex.what() << std::endl; 
        }
    }
}

/* TODO: continue checking work from here!
void SmartRedisManager::finalize() {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank == 0) {
        client.~Client();
    }
}

void SmartRedisManager::writeState(const std::vector<double>& state_local, const std::string& key) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    std::vector<double> state_global(state_global_size);
    
    // Gather the local states into a global state
    MPI_Gatherv(state_local.data(), state_local_size, MPI_DOUBLE,
                state_global.data(), state_sizes.data(), state_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Write global state into DB
    if (mpi_rank == 0) {
        client.PutTensor(key, state_global);
    }
}

void SmartRedisManager::readAction(const std::string& key) {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0) {
        bool exists = false;
        int interval = 100;
        int tries = 1000;

        int found = client.PollTensor(key, interval, tries, exists);
        if (found != 0 || !exists) {
            std::cerr << "Error in SmartRedis readAction. Actions array not found." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, found);
        }
        client.GetTensor(key, action_global);
        client.DeleteTensor(key);
    }

    // Broadcast rank 0 global action array to all processes
    MPI_Bcast(action_global.data(), action_global_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void SmartRedisManager::writeReward(const std::vector<double>& Ftau_neg, const std::string& key) {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0) {
        client.PutTensor(key, Ftau_neg);
    }
}

void SmartRedisManager::writeStepType(int step_type, const std::string& key) {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0) {
        std::vector<int64_t> step_type_tensor = {step_type};
        client.PutTensor(key, step_type_tensor);
    }
    step_type_mod = step_type;
}

void SmartRedisManager::writeTime(double time, const std::string& key) {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0) {
        std::vector<double> time_tensor = {time};
        client.PutTensor(key, time_tensor);
    }
}