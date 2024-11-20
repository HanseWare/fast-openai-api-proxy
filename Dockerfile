FROM python:3.11-slim
# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt /app
COPY LICENSE /app
COPY README.md /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY *.py /app/

COPY configs /app/configs

ENV FOAP_CONFIG_DIR=configs
ENV FOAP_LOGLEVEL=INFO
ENV FOAP_HOST=0.0.0.0
ENV FOAP_PORT=8000
# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application
#CMD ["python", "/app/app.py"]
CMD uvicorn app:app --host $FOAP_HOST --port $FOAP_PORT