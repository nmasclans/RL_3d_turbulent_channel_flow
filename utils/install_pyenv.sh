python3 -m venv smartrhea-env
source smartrhea-env/bin/activate
export CC=gcc CXX=g++ NO_CHECKS=1
pip install smartsim==0.4.2 # TODO: THIS REQUIRES PYTHON 3.11 INSTALLED, AND TRITON HAS PYTHON 3.12!
pip install smartredis
pip install scipy==1.6.0
pip install matplotlib==3.7.0
pip install numpy==1.20.0
pip install tensorflow==2.8.0
pip install tf_agents==0.10.0
pip install tensorflow-probability=0.14.1
smart build --device cpu --no_pt --no_tf