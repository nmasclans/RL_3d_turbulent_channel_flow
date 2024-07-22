#!/bin/bash
#
# SMARTREDIS installation for SMART-RHEA

# Parameters
VERS=0.4.0
BRANCH="f2003"
FOLDER="smartredis-${VERS}-${BRANCH}"
PREFIX="/apps/smartredis/${VERS}"
URL="https://github.com/ashao/SmartRedis"

# Modules to be loaded (depend on the machine)
# module purge
# module load gcc nvhpc cmake python

## Installation workflow
# Clone github repo
git clone ${URL} --depth=1 --branch ${BRANCH} ${FOLDER}
cd $FOLDER
make clobber
make deps
mkdir build && cd build
cmake .. \
	-DSR_PYTHON=ON \
	-DSR_FORTRAN=ON \
#	-DCMAKE_CXX_COMPILER=nvc++ \	# NVIDIA compilers (NVHPC), if RHEA is intended to run in CPUs + GPUs
#	-DCMAKE_C_COMPILER=nvc \		# NVIDIA compilers (NVHPC), if RHEA is intended to run in CPUs + GPUs
#	-DCMAKE_Fortran_COMPILER=nvfortran
	-DCMAKE_CXX_COMPILER=g++ \		# GCC compiler, if RHEA is intended to run on CPUs only
	-DCMAKE_C_COMPILER=gcc \		# GCC compiler, if RHEA is intended to run on CPUs only
	-DCMAKE_Fortran_COMPILER=gfortran
make -j $(getconf _NPROCESSORS_ONLN)
make install

# Copy 'install' directory
cd ..
sudo mkdir -p ${PREFIX}
sudo cp -r install/* $PREFIX/
