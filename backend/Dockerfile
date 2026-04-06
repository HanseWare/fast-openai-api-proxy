FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY app /app
COPY configs /app/configs

ENV FOAP_CONFIG_DIR=configs
ENV FOAP_LOGLEVEL=INFO
ENV FOAP_HOST=0.0.0.0
ENV FOAP_PORT=8000

EXPOSE 8000

CMD uvicorn main:app --host $FOAP_HOST --port $FOAP_PORT