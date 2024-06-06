# Image Quality Assessment in Teledermatology

## Overview
This repository contains scripts and models for assessing image quality in teledermatology.

## Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Training Script:

	•	root: Path to the dataset folder.
	•	num_distortions: Number of distortions to use.
	•	batch_size: Batch size for DataLoader.
	•	num_workers: Number of workers for DataLoader.
	•	model_type: ‘xgb_reg’, ‘xgb_cls’, ‘mlp_reg’, ‘mlp_cls’.
	•	sweep: Set to true for hyperparameter sweep.
	•	sweep_count: Number of sweeps to perform.
	•	sweep_config: Hyperparameter sweep configuration.
	•	logging: Configuration for logging, including use of wandb.

Test Script

	•	root: Path to the dataset folder.
	•	batch_size: Batch size for DataLoader.
	•	num_workers: Number of workers for DataLoader.
	•	model_path: Path to the model .pkl file.
	•	data_type: ‘s’ for synthetic, ‘a’ for authentic.

Inference Script

	•	model_path: Path to the model .pkl file.
	•	images_path: Path to the folder containing images.
	•	csv_path: Path to save the output CSV file.
	•	batch_size: Batch size for DataLoader.
	•	num_workers: Number of workers for DataLoader.

SSIM Script

	•	original_path: Path to the folder containing original images.
	•	distorted_path: Path to the folder containing distorted images.
	•	csv_path: Path to save the output CSV file.
	•	batch_size: Batch size for processing (not used in current implementation).
	•	num_workers: Number of workers for processing (not used in current implementation).

ARNIQA Script

	•	root: Root folder containing the images to be evaluated.
	•	regressor_dataset: Dataset used to train the regressor.
	•	output_csv: Output CSV file to save the quality predictions.

Single Image Inference Script

	•	image_path: Path to the image to be evaluated.
	•	model_path: Path to the model .pkl file.