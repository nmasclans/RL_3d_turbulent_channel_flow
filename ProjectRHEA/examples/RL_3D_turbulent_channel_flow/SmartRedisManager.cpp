#include "SmartRedisManager.hpp"

/// Default constructor
SmartRedisManager::SmartRedisManager() : client(nullptr) {
}

/// Constructor
SmartRedisManager::SmartRedisManager(int state_local_size2, int action_global_size2, int n_pseudo_envs2, const std::string& tag, bool db_clustered)
    : client(nullptr) 
{

    /// Store input arguments (input arguments of each process) to class variables
    n_pseudo_envs      = n_pseudo_envs2;
    state_local_size   = state_local_size2;
    action_global_size = action_global_size2;
    state_local_size_vec   = {static_cast<size_t>(state_local_size2)};
    action_global_size_vec = {static_cast<size_t>(action_global_size2)};

    /// Set read parameters
    read_interval = 100;
    read_tries = 1000;

    /// Get mpi_size & mpi_rank
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    state_sizes.resize(mpi_size);                           // vector<int>
    state_displs.resize(mpi_size);                          // vector<int>
    action_global.resize(action_global_size, 0.0);          // vector<double>
    state_global.resize(state_global_size, 0.0);            // vector<double>

    /* Gather the individual sizes to get total size and offsets in root process (0)
        sendbuf(const void*) = state_local_size2 : pointer to the data that each process sends to the root process,
            in this case the size of local state array of each mpi process
        sendcount(int) = 1 : number of elements to send from each process
        sendtype(MPI_Datatype) = MPI_UNSIGNED_LONG : MPI datatype for unsigned long 
        recvbuf(void*) = state_sizes.data() : pointer to the buffer where the gathered data will be stored in the root process,
            which will hold the gathered sizes from all processes
        recvcount(int) = 1: number of elements the root process will receive from each process 
        recvtype(MPI_Datatype) = MPI_INT: datatype of the elements received by the root process
        root(int) = 0 : rank of the root process that will receive the gathered data
        comm(MPI_Comm) = MPI_COMM_WORLD: communicator that defines the group of process involved in the communication, 
            which for MPI_COMM_WORLD includes all MPI process in the application */
    MPI_Gather(&state_local_size, 1, MPI_INT, state_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD); // update state_sizes

    // Compute displacements - global state idxs corresponding to the local state of each MPI process
    if (mpi_rank == 0) {
        int state_counter = 0;
        for (int i = 0; i < mpi_size; ++i) {
            state_displs[i] = state_counter;
            state_counter += state_sizes[i];
        }
    }

    /*  Calculate state_global_size: sum the local state_local_size of each process in MPI_COMM_WORLD, and distribute state_global_size result to all processes in MPI_COMM_WORLD
        void MPI::Comm::Allreduce(const void* sendbuf, void* recvbuf,
                                  int count, const MPI::Datatype& datatype, 
                                  const MPI::Op& op) const=0 */
    MPI_Allreduce(&state_local_size, &state_global_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); // update state_global_size
    state_global_size_vec = {static_cast<size_t>(state_global_size)};

    /// Init client (only root process!), and write global state and action sizes into DB.
    if (mpi_rank == 0) {
        /// Initialize client, the client is initialized to utilize a SmartRedis Orchestrator in cluster configuration if db_clustered
        ///     Client(bool cluster, const std::string& logger_name = "default");
        try {
            client = std::make_unique<SmartRedis::Client>(db_clustered, tag);
        } catch (const SmartRedis::Exception& ex) {
            std::cerr << "Error initializing SmartRedis client: " << ex.what() << std::endl; 
            return;
        }

        /// Write global state and action sizes into database:
        try {
            /* void Client::put_tensor(const std::string& name,
                                       void* data,
                                       const std::vector<size_t>& dims,
                                       const SRTensorType type,
                                       const SRMemoryLayout mem_layout) */
            client->put_tensor("state_size",  &state_global_size,  {1}, SRTensorType::SRTensorTypeInt64, SRMemoryLayout::SRMemLayoutContiguous);
            client->put_tensor("action_size", &action_global_size, {1}, SRTensorType::SRTensorTypeInt64, SRMemoryLayout::SRMemLayoutContiguous);
        } catch (const SmartRedis::Exception& ex) {
            std::cerr << "Error putting tensor: " << ex.what() << std::endl; 
            return;
        }
    }

    // Debugging
    if (mpi_rank==0)
        std::cout << "SmartRedis manager constructed" << std::endl << std::endl;
}

/// Destructor
SmartRedisManager::~SmartRedisManager() {
    /// MPI Finalized in myRHEA main
    /// No need to explicitly call client.~Client(), the destructor for client is automatically called.
}

void SmartRedisManager::writeState(const std::vector<double>& state_local, const std::string& key) {
    /* Build global state: gather the local states into a global state
        int MPI_Gatherv(const void *sendbuf, 
                        int sendcount, 
                        MPI_Datatype sendtype,
                        void *recvbuf, 
                        const int recvcounts[], 
                        const int displs[],
                        MPI_Datatype recvtype, 
                        int root, MPI_Comm comm) */
    state_global.resize(state_global_size);
    MPI_Gatherv(state_local.data(), state_local_size, MPI_DOUBLE, 
                state_global.data(), state_sizes.data(), state_displs.data(), 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // /// Write global state into database
    if (mpi_rank == 0) {
        /* void Client::put_tensor(const std::string& name,
                                   void* data,
                                   const std::vector<size_t>& dims,
                                   const SRTensorType type,
                                   const SRMemoryLayout mem_layout) */
        client->put_tensor(key, state_global.data(), state_global_size_vec, SRTensorType::SRTensorTypeDouble, SRMemoryLayout::SRMemLayoutContiguous);
        std::cout << "State written" << std::endl;
    }
}

/// Read state from database -> updates variable 'state_global'
void SmartRedisManager::readState(const std::string& key) {
    if (mpi_rank == 0) {
        /* bool Client::poll_tensor(const std::string& name, 
                                    int poll_frequency_ms, 
                                    int num_tries) */
        bool found = client->poll_tensor(key, read_interval, read_tries);
        if (!found) {
            std::cerr << "ERROR in readState, state array not found." << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* void Client::get_tensor(const std::string& name,
                                   void*& data,
                                   std::vector<size_t>& dims,
                                   SRTensorType& type,
                                   const SRMemoryLayout mem_layout) */
        void* get_data_ptr;
        std::vector<size_t> get_dims;
        SRTensorType get_SRTensorType;
        try {
            client->get_tensor(key, get_data_ptr, get_dims, get_SRTensorType, SRMemoryLayout::SRMemLayoutContiguous);
        } catch (const SmartRedis::RuntimeException& e) {
            std::cerr << "SmartRedis RuntimeException: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        /// Check received state data
        if (get_data_ptr == nullptr) {
            std::cerr << "ERROR in readState, data pointer is null for key " << key << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (get_dims != state_global_size_vec) {
            std::cerr << "ERROR in readState, dimensions mismatch for key " << key << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (get_SRTensorType != SRTensorType::SRTensorTypeDouble) {
            std::cerr << "ERROR in readState, unexpected tensor type for key " << key << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /// Total elements of state (required in case action has >1 dimension)
        size_t total_elements = 1;
        for (const size_t& dim : state_global_size_vec) {
            total_elements *= dim;
        }

        /// Update state data
        for (size_t i=0; i<total_elements; i++) {
            state_global[i] = ((double*)get_data_ptr)[i];
        }

        /// Delete action data from database
        //  void Client::delete_tensor(const std::string& name)
        client->delete_tensor(key);
    }

    /* Broadcast rank 0 global state array to all processes
        void MPI::Comm::Bcast(void* buffer, 
                              int count,
	                          const MPI::Datatype& datatype, 
                              int root) const = 0 */
    MPI_Bcast(state_global.data(), state_global_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        std::cout << "State read (and deleted)" << std::endl;
    }
}

/// Read action from database -> updates variable 'action_global'
void SmartRedisManager::readAction(const std::string& key) {
    if (mpi_rank == 0) {
        /* bool Client::poll_tensor(const std::string& name, 
                                    int poll_frequency_ms, 
                                    int num_tries) */
        bool found = client->poll_tensor(key, read_interval, read_tries);
        if (!found) {
            std::cerr << "ERROR in readAction, actions array not found." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* void Client::get_tensor(const std::string& name,
                                   void*& data,
                                   std::vector<size_t>& dims,
                                   SRTensorType& type,
                                   const SRMemoryLayout mem_layout) */
        void* get_data_ptr;
        std::vector<size_t> get_dims;
        SRTensorType get_SRTensorType;
        try {
            client->get_tensor(key, get_data_ptr, get_dims, get_SRTensorType, SRMemoryLayout::SRMemLayoutContiguous);
        } catch (const SmartRedis::RuntimeException& e) {
            std::cerr << "SmartRedis RuntimeException: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        /// Check received action data
        if (get_data_ptr == nullptr) {
            std::cerr << "ERROR in readAction, data pointer is null for key " << key << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (get_dims != action_global_size_vec) {
            std::cerr << "ERROR in readAction, dimensions mismatch for key " << key << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (get_SRTensorType != SRTensorType::SRTensorTypeDouble) {
            std::cerr << "ERROR in readAction, unexpected tensor type for key " << key << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /// Total elements of action (required in case action has >1 dimension)
        size_t total_elements = 1;
        for (const size_t& dim : action_global_size_vec) {
            total_elements *= dim;
        }

        /// Update action data
        for (size_t i=0; i<total_elements; i++) {
            action_global[i] = ((double*)get_data_ptr)[i];
        }

        /// Delete action data from database
        //  void Client::delete_tensor(const std::string& name)
        client->delete_tensor(key);
    }

    /* Broadcast rank 0 global action array to all processes
        void MPI::Comm::Bcast(void* buffer, 
                              int count,
	                          const MPI::Datatype& datatype, 
                              int root) const = 0 */
    MPI_Bcast(action_global.data(), action_global_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        std::cout << "Action read (and deleted)" << std::endl;
    }
}

void SmartRedisManager::writeReward(const double reward, const std::string& key) {
    if (mpi_rank == 0) {
        std::vector<double> reward_tensor = {reward};
        client->put_tensor(key, reward_tensor.data(), {1}, SRTensorType::SRTensorTypeDouble, SRMemoryLayout::SRMemLayoutContiguous);
    }
}

/// TODO: necessary method? delete if not used
void SmartRedisManager::writeStepType(const int step_type, const std::string& key) {
    if (mpi_rank == 0) {
        std::vector<int64_t> step_type_tensor = {step_type};
        client->put_tensor(key, step_type_tensor.data(), {1}, SRTensorType::SRTensorTypeInt64, SRMemoryLayout::SRMemLayoutContiguous);
    }
}

void SmartRedisManager::writeTime(const double time, const std::string& key) {
    if (mpi_rank == 0) {
        std::vector<double> time_tensor = {time};
        client->put_tensor(key, time_tensor.data(), {1}, SRTensorType::SRTensorTypeDouble, SRMemoryLayout::SRMemLayoutContiguous);
    }
}

/// Get methods
std::vector<double> SmartRedisManager::getStateGlobal(){
    return state_global;
}

std::vector<double> SmartRedisManager::getActionGlobal(){
    return action_global;
}

/// Check I: test SmartRedis, RedisAI and Redis installation and compilation
void testSmartRedisClient() {
    std::cout << "Testing SmartRedis..." << std::endl;
  
    /// Create smartredis client
    try {
        SmartRedis::Client client(false);
        std::cout << "SmartRedis test completed." << std::endl;
    } catch (const SmartRedis::RuntimeException& e) {
        std::cerr << "SmartRedis RuntimeException: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception occurred." << std::endl;
    }
}

/// Check II: Print database contents
void SmartRedisManager::printDatabaseContent() {
    if (mpi_rank==0){
        try{
            std::vector<std::string> tensor_keys = {"state_size", "action_size", "state_key", "action_key",
                                                    "reward_key", "step_type_key", "time_key"};
            std::cout << "Database contents:" << std::endl;
            for (const auto& key : tensor_keys) {
                if (client->tensor_exists(key)){
                    // Get tensor
                    void* data;
                    std::vector<size_t> dims;
                    size_t total_elements = 1;
                    SRTensorType type;
                    client->get_tensor(key, data, dims, type, SRMemoryLayout::SRMemLayoutContiguous);
                    // Verify data pointer is not null
                    if (data == nullptr) {
                        std::cerr << "Error: data pointer is null for key " << key << std::endl;
                        continue;
                    }
                    // Print tensor details
                    std::cout << "Tensor key: " << key << std::endl;
                    std::cout << "Tensor dimensions: ";
                    for (const auto& dim : dims) {
                        std::cout << dim << " ";
                        total_elements *= dim;
                    }
                    std::cout << std::endl;
                    std::cout << "Tensor #elements: " << total_elements << std::endl;
                    // Print tensor data based on type (concatenated data, use total_elements to go through data elements)
                    std::cout << "Data ";
                    if (type==SRTensorType::SRTensorTypeDouble) {
                        std::cout << "(type SRTensorTypeDouble): ";
                        for (size_t i = 0; i < total_elements; ++i)
                            std::cout << ((double*)data)[i] << " ";
                    } else if (type==SRTensorType::SRTensorTypeFloat) {
                        std::cout << "(type SRTensorTypeFloat): ";
                        for (size_t i = 0; i < total_elements; ++i)
                            std::cout << ((float*)data)[i] << " ";
                    } else if (type==SRTensorType::SRTensorTypeInt64) {
                        std::cout << "(type SRTensorTypeInt64): ";
                        for (size_t i = 0; i < total_elements; ++i)
                            std::cout << ((int*)data)[i] << " ";
                    // Add cases for other tensor types as needed
                    } else {
                        std::cout << "Unsupported tensor type" << std::endl;
                    }
                    std::cout << std::endl;
                } else {
                    std::cout << "Tensor key '" << key << "' does not exist" << std::endl;
                }
            }
            std::cout << std::endl;
        } catch (const SmartRedis::Exception& ex) {
            std::cerr << "Error retrieving database contents: " << ex.what() << std::endl;
        }
    }
}