# Dockerfile: PySpark + Jupyter environment for ETL and EDA

FROM openjdk:11-slim AS base

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy files into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set Spark-related environment variables
ENV JAVA_HOME=/usr/local/openjdk-11
ENV PATH="$JAVA_HOME/bin:$PATH"

# Expose Jupyter port
EXPOSE 8888

# Default command: launch JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
