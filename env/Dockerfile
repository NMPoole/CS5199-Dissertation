FROM nvcr.io/nvidia/pytorch:21.05-py3

ARG local_uid
ARG local_user

RUN adduser --uid ${local_uid} --gecos "" --disabled-password ${local_user}

WORKDIR /data/${local_user}

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN apt update && apt install -y python3-openslide

USER ${local_user}

ENV PATH="/data/${local_user}/.local/bin:${PATH}"

COPY ./ ./

RUN pip install --user --upgrade pip
RUN pip install --user --no-cache-dir -r requirements.txt
