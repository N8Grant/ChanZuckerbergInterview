FROM python:3.12-slim

# Gpt
RUN apt-get update && apt-get install -y \
    libgl1 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    libxcb-xinerama0 \
    build-essential \
    qtbase5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what we need (because .dockerignore protects us)
COPY . .

ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1

RUN pip install .
# gpt
RUN pip install --prefer-binary PyQt5

ENTRYPOINT ["python", "-m", "chanzuck.cli"]
