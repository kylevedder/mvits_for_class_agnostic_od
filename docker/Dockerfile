FROM nvidia/cudagl:11.1-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]
# Set the timezone info because otherwise tzinfo blocks install 
# flow and ignores the non-interactive frontend command 🤬🤬🤬
RUN ln -snf /usr/share/zoneinfo/America/New_York /etc/localtime && echo "/usr/share/zoneinfo/America/New_York" > /etc/timezone

# Core system packages
RUN apt-get update --fix-missing
RUN apt install -y software-properties-common wget curl gpg gcc git make

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN apt install -y apt-utils

# Additional dev packages
RUN apt install -y --no-install-recommends libssl-dev libmodule-install-perl libboost-all-dev libopenblas-dev
RUN apt install -y locate nano

RUN conda update conda -y

ENV TORCH_CUDA_ARCH_LIST="Ampere;Turing;Pascal"
ENV FORCE_CUDA="1"
ENV CACHEBUST="1"
RUN conda install python=3.8 pytorch::pytorch=1.8.1 cudatoolkit=11.1 cudatoolkit-dev=11.1 torchvision -c pytorch -c nvidia -c conda-forge

# Clone the repo in order to install dependencies and compile the custom CUDA ops;
# The actual codebase itself is going to be mounted as a volume by the host under /project.
RUN git clone https://github.com/kylevedder/mvits_for_class_agnostic_od.git /mvits
WORKDIR /mvits
RUN pip install -r requirements.txt
WORKDIR /mvits/models/ops
RUN ./make.sh

# Download precomputed weights for the model
RUN pip install gdown
WORKDIR /weights
RUN gdown https://drive.google.com/uc?id=1eYMmOLx3wNX4u5vGOWgVTwr0vLH1MyIr

# Download precomputed resnet weights used by the model in torchhub.
RUN mkdir -p /root/.cache/torch/hub/checkpoints/
RUN wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O /root/.cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth

WORKDIR /project
ENV PYTHONPATH="./:$PYTHONPATH"
RUN echo "python inference/main_mvit_multi_query.py -m mdef_detr -i /Datasets/coco/val2017 -c /weights/MDef_DETR_r101_epoch20.pth -tq_list '[all objects,all entities,all visible entities and objects,all obscure entities and objects]'" >> ~/.bash_history