#!/bin/bash

# Set the environment name
ENV_NAME="cs646project"

# Create a new conda environment
conda create -n $ENV_NAME python=3.8 -y

# Activate the environment
conda activate $ENV_NAME

# Install PyTorch with GPU support
conda install -c conda-forge pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -y

# Install other required packages
conda install -c conda-forge transformers=4.8.1 -y
conda install -c conda-forge timm=0.4.9 -y
conda install -c conda-forge ruamel.yaml -y
conda install -c conda-forge opencv -y

# Confirm installation
echo "Environment $ENV_NAME has been created and packages installed."
