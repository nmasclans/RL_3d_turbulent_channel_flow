//TODO: remove file

# Define the executable and source
EXECUTABLE = test
MAIN = test.cpp

# Dynamically define base directory
BASE_DIR = $(shell pwd)

# Define directories for SmartRedis and RedisAI
SMARTREDIS_DIR = smartredis-0.4.0-f2003/install
REDISAI_DIR = redisai-1.2.5/install-cpu

# Define paths for include directories and libraries
INC_DIR_SMARTREDIS = $(BASE_DIR)/$(SMARTREDIS_DIR)/include
INC_DIR_REDISAI = $(BASE_DIR)/$(REDISAI_DIR)
LIB_DIR_SMARTREDIS = $(BASE_DIR)/$(SMARTREDIS_DIR)/lib
LIB_DIR_REDISAI = $(BASE_DIR)/$(REDISAI_DIR)

# Compiler and flags
CXX = mpicxx
CXXFLAGS = -Ofast -Wall -std=c++0x -Wno-unknown-pragmas -Wno-unused-variable \
           -I$(INC_DIR_SMARTREDIS) -I$(INC_DIR_REDISAI)
LDFLAGS = -L$(LIB_DIR_SMARTREDIS) -L$(LIB_DIR_REDISAI) \
          -lsmartredis -lredisai

# Build target
$(EXECUTABLE): $(MAIN)
	$(CXX) $(MAIN) $(CXXFLAGS) -o $@ $(LDFLAGS)

# Clean target
.PHONY: clean
clean:
	$(RM) $(EXECUTABLE)
