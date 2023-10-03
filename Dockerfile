FROM ubuntu:22.04

# Setup workdir
WORKDIR /experiment/

# Setup dependencies
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install git subversion openjdk-8-jdk curl unzip build-essential cpanminus python3.10 python3.10-distutils -y
RUN apt-get install nvidia-driver-535 nvidia-dkms-535 -y
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Copy files
WORKDIR /experiment/
COPY RepairLLaMa-Lora-7B-MegaDiff/ /experiment/RepairLLaMa-Lora-7B-MegaDiff/
COPY CodeLlama-7b-hf/ /experiment/CodeLlama-7b-hf/
COPY flacoco.jar ./
COPY requirements.txt ./
RUN mkdir -p /experiment/java_tools/
COPY java_tools/ /experiment/java_tools/
COPY main.py ./

# Install python dependencies
RUN pip3 install -r requirements.txt
