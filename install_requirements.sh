#!/bin/bash

# Install PyTorch and torchvision with compatible versions
conda install -y pytorch==2.1.1 torchvision==0.16.1 -c pytorch -c nvidia

# Install other required packages via pip
pip install pandas==2.1.3 \
            matplotlib==3.8.2 \
            pyyaml==6.0.1 \
            dotmap==1.3.30 \
            tqdm==4.66.1 \
            wandb==0.15.0 \
            einops==0.7.0 \
            openpyxl==3.1.2 \
            scikit-learn==1.3.0 \
            scipy==1.11.4 \
            kornia==0.7.0 \
            opencv-python==4.9.0.80 \
            xgboost==2.0.3 \
            scikit-image==0.22.0  # Compatible version with Python 3.9
