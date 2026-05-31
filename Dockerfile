FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install system deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (JAX CPU, Keras Core + KerasNLP, and helpers)
RUN pip install --no-cache-dir "jax[cpu]" keras-core==0.14.0 keras-nlp==0.29.0 kagglehub==1.0.1 tokenizers sentencepiece h5py numpy

# Copy workspace (we will mount your project when running the container instead of baking all files)
COPY . /workspace

CMD ["python", "scripts/run_in_jax.py"]
