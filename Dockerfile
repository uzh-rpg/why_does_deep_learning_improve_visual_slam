FROM ubuntu:20.04

# Own dependecies
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-importlib-metadata python3-more-itertools python3-zipp
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libglew-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libopencv-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libyaml-cpp-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y cmake
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libboost-all-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libsuitesparse-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libeigen3-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libssl-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libhdf5-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libhdf5-cpp-103

# Reference : https://github.com/jahaniam/orbslam3_docker/blob/main/Dockerfile.cpu
# Pangolin dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git-all libgl1-mesa-dev libpython3-dev apt-transport-https ca-certificates software-properties-common

# Build Pangolin
RUN cd /tmp && git clone https://github.com/stevenlovegrove/Pangolin && \
    cd Pangolin && git checkout v0.6 && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-std=c++11 .. && \
    make -j$nproc && make install && \
    cd / && rm -rf /tmp/Pangolin

# create user with sudo permissions
RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y sudo
ARG USERNAME=giovanni
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME

# Set work directory to repo
WORKDIR /home/$USERNAME/orbslam3_docker
