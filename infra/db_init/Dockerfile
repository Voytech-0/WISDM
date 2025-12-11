FROM python:3.11-slim

RUN apt-get update

WORKDIR /db_init

COPY db_init/requirements-db-init.txt .
RUN pip install -r requirements-db-init.txt
ENV PYTHONPATH=/db_init/

COPY data/wisdm-dataset/raw/ data/wisdm-dataset/raw/
COPY db_init/ db_init/
COPY configs/ configs/
