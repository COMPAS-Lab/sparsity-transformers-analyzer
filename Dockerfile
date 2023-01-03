FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get -y install unzip python3 python3-pip python3-venv vim 
RUN python3 -m venv /opt/nlp
ENV PATH="/opt/nlp/bin:$PATH"

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install scipy scikit-learn
RUN pip3 install datasets accelerate evaluate

WORKDIR /app/transformers
COPY ./transformers .
RUN pip3 install -e .

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /workspace
