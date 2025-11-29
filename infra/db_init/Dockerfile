FROM python:3.11-slim

RUN apt-get update

WORKDIR /db_init

COPY requirements.txt .
RUN pip install -r requirements.txt
ENV PYTHONPATH=/db_init/

COPY data/wisdm-dataset/raw/ data/wisdm-dataset/raw/
COPY db_init db_init
COPY infra/db_init/start.sh /start.sh
COPY configs/ configs/
RUN chmod +x /start.sh
CMD ["/start.sh"]

