#!/bin/bash

# Set non-interactive frontend
export DEBIAN_FRONTEND=noninteractive

# Activate the llama_factory environment
# Uncomment the following line if you need to create the environment
# conda create --name demodiff_inf python=3.11 -y
# conda activate demodiff_inf

# Function to get the current conda environment name
get_current_conda_env() {
    echo "current: $CONDA_DEFAULT_ENV"
}

# Get the current conda environment
current_env=$(basename "$CONDA_PREFIX")

# Check if the current environment is 'llamole'
if [[ "$current_env" != *"demodiff_inf"* ]]; then
    echo "current: $CONDA_DEFAULT_ENV"
    echo "Please activate one of demodiff_inf before running this script."
    echo "You can activate an environment using one of these commands:"
    echo "conda activate demodiff_inf"
    exit 1
fi

echo "Running in conda environment: $current_env"

# "pandas>=2.0.0" \
# Install packages using pip
pip install --no-cache-dir \
    "rdkit==2023.9.6" \
    pyarrow \
    "pandas" \
    "matplotlib" \
    "numpy" \
    scipy \
    pyyaml \
    "imageio==2.26.0" \
    pandarallel \
    pydantic \

# Install PyTDC
pip install PyTDC

# Install PyTorch
pip install --no-cache-dir torch

# Install PyTorch Geometric and related packages
pip install --no-cache-dir torch_geometric

# cd /afs/crc.nd.edu/group/dmsquare/vol2/gliu7/developing/demodiff/developing/downstream/baseline/mol-opt
# pip install -e .

mamba install -c conda-forge openbabel

# https://github.com/liugangcode/torch-molecule
# cd /afs/crc.nd.edu/group/dmsquare/vol2/gliu7/developing/demodiff/developing/downstream/baseline/torch-molecule
# pip install -e .

# for tdc
# pip install scikit-learn==1.2.2

pip install openai

echo "Installation complete!"