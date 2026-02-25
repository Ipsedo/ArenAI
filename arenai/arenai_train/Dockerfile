FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu22.04
LABEL authors="Ipsedo"

CMD mkdir /opt/arenai
COPY ../* /opt/arenai

CMD cd /opt/arenai/arenai_train \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make

ENTRYPOINT ["top", "-b"]
