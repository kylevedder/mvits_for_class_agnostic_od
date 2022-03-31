#!/bin/bash

nvidia-docker run --rm -it \
 -v `pwd`:/project \
 -v /data:/data \
 -v /scratch:/scratch \
 -v /Datasets:/Datasets \
 mvits