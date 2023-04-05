FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN apt-get update && apt-get install -y vim openssh-server git
RUN curl -O -L https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz && \
    tar xvzf openmpi-4.0.1.tar.gz && \
    cd openmpi-4.0.1 && \
    ./configure --prefix=/usr/local && \
    make all && make install
COPY nccl-repo-ubuntu1804-2.6.4-ga-cuda10.0_1-1_amd64.deb /nccl.deb
RUN dpkg -i /nccl.deb 

RUN git clone https://github.com/NVIDIA/nccl && cd nccl && make -j $(nproc)

COPY libnccl2_2.6.4-1+cuda10.0_amd64.deb libnccl-dev_2.6.4-1+cuda10.0_amd64.deb /
RUN dpkg -i libnccl2_2.6.4-1+cuda10.0_amd64.deb && dpkg -i libnccl-dev_2.6.4-1+cuda10.0_amd64.deb

# RUN cp /nccl/build/include/nccl.h /usr/local/cuda/include/nccl.h && \
#     ln -s /nccl/build/lib/libnccl.so.2 /usr/local/cuda/nccl/lib/libnccl.so.2

RUN echo "deb http://dk.archive.ubuntu.com/ubuntu/ xenial main" >> /etc/apt/sources.list && \
    echo "deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe" >> /etc/apt/sources.list && \
    apt update && apt install -y gcc-4.9 && apt install -y g++-4.9 && cd /usr/bin/ && rm -r gcc && \
    ln -sf gcc-4.9 gcc && rm -r g++ && ln -sf g++-4.9 g++

RUN python -m pip install sklearn && HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=/nccl/build pip install horovod==0.19.0

CMD [ "/bin/bash" ]