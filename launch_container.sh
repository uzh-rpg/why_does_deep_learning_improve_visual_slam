#!/bin/sh
xhost local:root

# UI permisions
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

xhost +local:docker

docker run --privileged --rm -it \
    --volume /home/giovanni/orbslam3_docker/:/home/giovanni/orbslam3_docker/:rw \
    --volume /home/giovanni/datasets/TartanAirMono_grayscale/:/datasets/TartanAirMono_grayscale/:ro \
    --volume /home/giovanni/datasets/UZH_FPV/:/datasets/UZH_FPV/:ro \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --net=host \
    --privileged \
    --user $(id -u):$(id -g) \
    orbslam3
    bash
