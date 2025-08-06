# Use Ubuntu with Python
FROM python:3.10-slim

# Set environment
ARG ENV=dev
ENV ENV_MODE=$ENV
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the environment config
COPY environments/$ENV_MODE /app/environments

# Copy the app
COPY . /app

# Expose Jupyter port only
EXPOSE 8888

# Start only Jupyter Lab
CMD jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
