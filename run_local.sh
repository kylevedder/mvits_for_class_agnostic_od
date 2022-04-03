#!/bin/bash
docker run --gpus all --rm -it \
 -v `pwd`:/project \
 -v /data:/data \
 -v /scratch:/scratch \
 -v /Datasets:/Datasets \
 mvits
