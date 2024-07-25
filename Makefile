# Environment paths
RHEA_PATH    = /home/jofre/Nuria/flowsolverrhea

EXECUTABLE   = RHEA.exe
MAIN         = myRHEA.cpp
PROJECT_PATH = $(RHEA_PATH)
SRC_DIR      = $(PROJECT_PATH)/src
CXX          = mpicxx
# CPU FLAGS
#CXXFLAGS     = -O3 -Wall -std=c++0x -Wno-unknown-pragmas -Wno-unused-variable -I$(PROJECT_PATH)
CXXFLAGS     = -Ofast -Wall -std=c++17 -Wno-unknown-pragmas -Wno-unused-variable -I$(PROJECT_PATH) # -std=c++17 required for redisai compilation
# CPU-GPU FLAGS
#CXXFLAGS     = -fast -acc -ta=tesla:managed -Minfo=accel -O3 -Wall -std=c++0x -I$(PROJECT_PATH)
#CXXFLAGS     = -fast -acc -ta=tesla,pinned -Minfo=accel -O3 -Wall -std=c++0x -I$(PROJECT_PATH)
# UBUNTU - LINUX
INC_LIB_YAML        =
INC_DIR_YAML        =
INC_LIB_HDF5        = -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi 
INC_DIR_HDF5        = -I/usr/include/hdf5/openmpi
INC_LIB_SMARTREDIS  = -L/apps/smartredis/0.4.0/lib
INC_DIR_SMARTREDIS  = -I/apps/smartredis/0.4.0/include
INC_INSTALL_REDISAI = /apps/redisai/1.2.5/ # dir. of dynamic redisai.so
# MAC - OS X
#INC_LIB_YAML = -L/usr/local/lib
#INC_DIR_YAML = -I/usr/local/include
#INC_LIB_HDF5 =
#INC_DIR_HDF5 =
# TODO: include SmartRedis & RedisAi for Mac system

# !! THE LINES BELOW SHOULD NOT BE MODIFIED !! #

OBJS = $(SRC_DIR)/*.cpp SmartRedisManager.cpp
INC_LIB = $(INC_LIB_YAML) $(INC_LIB_HDF5) $(INC_LIB_SMARTREDIS)
INC_DIR = $(INC_DIR_YAML) $(INC_DIR_HDF5) $(INC_DIR_SMARTREDIS) -I$(INC_INSTALL_REDISAI)
LDFLAGS      = -lyaml-cpp -lhdf5 -lsmartredis -Wl,-rpath,$(INC_INSTALL_REDISAI)

$(EXECUTABLE): $(OBJS)
	$(CXX) $(MAIN) $(CXXFLAGS) $(OBJS) -o $@ $(INC_LIB) $(INC_DIR) $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) $(EXECUTABLE) *.o
