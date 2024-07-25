#!/bin/bash
#
# REDISAI installation for SMART-SOD2D
#
# Bernat Font, Arnau Miro (c) 2023

# Parameters
VERS=1.2.5
FOLDER="redisai-${VERS}"
PREFIX="/apps/redisai/${VERS}"
URL="https://github.com/RedisAI/RedisAI"

# Modules to be loaded (depend on the machine)
# module purge
# module load gcc cmake

## Installation workflow
# Clone github repo
git clone --recursive ${URL} ${FOLDER}
cd $FOLDER
git checkout v${VERS}
bash get_deps.sh
mkdir build && cd build
cmake .. -DBUILD_TORCH=OFF
make -j $(getconf _NPROCESSORS_ONLN)
make install

# Copy 'install' directory
cd ..
sudo mkdir -p ${PREFIX}
sudo cp -r install-cpu/* $PREFIX/

# redisAI has dependency of Redis - install Redis
# sudo apt update
# sudo apt upgrade
sudo apt install redis-server

# Give execution permissions to redisai.so file
sudo chmod +x $PREFIX/redisai.so