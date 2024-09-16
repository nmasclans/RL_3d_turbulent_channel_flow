The basic ingredients necessary to run the current framework are: flow solver RHEA, SmartRedis, SmartSim, and TF-Agents. 

Both RHEA and SmartRedis need to be compiled.

The instructions for compiling flow solver RHEA are found in flowsolverrhea repository: https://gitlab.com/ProjectRHEA/flowsolverrhea.

SmartRedis is dynamically linked during the compilation of RHEA. Installation scripts can also be found in [install_smartredis.sh](install_smartredis.sh) and [install_redisai.sh](install_redisai.sh).

__Attention:__  RHEA and SmartRedis need to be compiled with the same compiler so that they can be linked afterwards. 
From RHEA Makefile, `mpicxx` compiler is used to compile RHEA.
To check which compiler is `mpicxx` specifically, we do:
```
$ mpicxx -show
g++ -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi
```
The output we are seeing from the `mpicxx -show` command indicates that mpicxx is using `g++` as the underlying C++ compiler and linking it with the OpenMPI libraries.


### Python environment

Conda environment `smartrhea-env` works well for training, but experiences an error when saving the model, due to an incompatibility issue between `tensorflow-probability` and `tf-agents`.
To solve this issue, `smartrhea-env-v2` installs a different version of these libraries.


