#!/bin/bash
xhost +
docker run --gpus all --rm -it \
--shm-size=6gb \
-v `pwd`:/project \
-v /data:/data \
-v /scratch:/scratch \
-v /Datasets:/Datasets \
-v /tmp/.X11-unix:/tmp/.X11-unix \
--net=host \
-e DISPLAY=$DISPLAY \
-h $HOSTNAME \
--privileged \
mvits
