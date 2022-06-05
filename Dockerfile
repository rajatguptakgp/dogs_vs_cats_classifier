FROM centos:latest

RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-Linux-* \
    && sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Linux-*

RUN yum -y update  \
    && yum -y install gcc unzip\
    && yum -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local/ \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3.8 \
    && conda update conda \
    && conda clean --all --yes \
    && rpm -e --nodeps curl bzip2 \
    && yum -y install mesa-libGL

RUN conda create -y -n my_env_prod python=3.8

RUN mkdir /app \ 
    && cd /app

WORKDIR /app

COPY input.zip . 
RUN unzip input.zip \ 
    && rm input.zip

COPY catdog_CV-0.0.1.tar.gz .
RUN tar -xzf catdog_CV-0.0.1.tar.gz \
    && rm catdog_CV-0.0.1.tar.gz \
    && cd catdog_CV-0.0.1 \ 
    && mv * ../ \
    && cd .. \ 
    && rm -rf catdog_CV-0.0.1

RUN /bin/bash -c "source activate my_env_prod \
                  && pip install -r requirements.txt protobuf==3.20.0 flask graphviz"

EXPOSE 7000
ENTRYPOINT ["tail", "-f", "/dev/null"]