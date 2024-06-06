# Automated Image Quality Assessment in Teledermatology

## Overview

### Abstract

This research aims to develop and evaluate automated methods for assessing image quality in teledermatology, addressing the critical issue of image quality in remote dermatological consultations. Poor image quality can lead to misdiagnosis or require multiple exchanges between patients and dermatologists to get clearer images, causing delays in diagnosis and treatment. This research explores different image quality assessment (IQA) techniques and their applicability in teledermatology, focusing on improving diagnostic accuracy by ensuring good quality images are available to dermatologists. Diagnostic accuracy refers to the ability of dermatologists to correctly identify and diagnose skin conditions based on the images they receive.

The methodology involved a detailed review of existing IQA methods, followed by the development of a synthetic distortion pipeline to create a wide range of training datasets from good quality dermatological images. These images were gathered from the Fitzpatrick17k and SCIN datasets. For feature extraction, the ARNIQA backbone was used, and various machine learning models, including XGBRegressor, XGBClassifier, MLP Regressor, and MLP Classifier, were trained to assess image quality based on seven key criteria: lighting, background, field of view, orientation, focus, resolution, and color calibration. The models were trained on synthetic distortions and validated on both synthetically distorted and real-world teledermatology images. Performance metrics such as Mean Absolute Error (MAE), R-squared (R2), Spearman's Rank Order Correlation Coefficient (SRCC), and Cohen's Kappa were used for evaluation.

The results showed that the automated IQA methods can assess image quality in the teledermatology domain, closely matching human evaluations and providing reliable feedback on image quality. The final model achieved good performance across various distortions, improving the reliability and effectiveness of teledermatology services. This research highlights the potential of automated IQA systems to improve diagnostic accuracy and patient care in remote dermatological consultations.

## Usage

<details>
<summary><h3>Getting Started</h3></summary>

#### Installation

I recommend using the [**Anaconda**](https://www.anaconda.com/) package manager to avoid dependency/reproducibility problems. For Linux systems, you can find a conda installation guide [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

1. Clone the repository

```sh
git clone https://github.com/Schoggi-Mimi/bachelor-thesis
```

2. Install Python dependencies

```sh
conda create -n ARNIQA -y python=3.10
conda activate ARNIQA
cd ARNIQA
chmod +x requirements.sh
./requirements.sh
```

#### Data Preparation

For the filtered good quality images, contact me and place them under the `data` directory.

For the original images, they can be downloaded from here:
1. [**SCIN**](https://github.com/google-research-datasets/scin)
2. [**Fitzpatrick17k**](https://github.com/mattgroh/fitzpatrick17k)

At the end, the directory structure should look like this:

```
├── data
|    ├── COMB
|    |   ├── embeddings
|    ├── F17K
|    |   ├── embeddings
|    ├── SCIN
|    |    ├── embeddings
|    ├── test_70
|    |    ├── distorted
|    |    ├── embeddings
|    ├── test_200
|    |    ├── embeddings
|    |    ├── scores.json
|    ├── ood
```

</details>

<details>
<summary><h3>Single Image Inference</h3></summary>
To get the quality score of a single image, run the following command:

```sh
python single_image_inference.py --config_path config.yaml
```

Parameters:
- `--image_path`: Path to the image to be evaluated.
- `--model_path`: Path to the model.

</details>

<details>
<summary><h3>Training</h3></summary>

To train the model, run the following command:

```sh
python train.py --config_path config.yaml
```

### Config File Explanation

Parameters:
- `--root`: Path to the dataset folder.
- `--num_distortions`: Number of distortions to use.
- `--batch_size`: Batch size for DataLoader.
- `--num_workers`: Number of workers for DataLoader.
- `--model_type`: Model type to use ('xgb_reg', 'xgb_cls', 'mlp_reg', 'mlp_cls').
- `--sweep`: Set to true for hyperparameter sweep.
- `--sweep_count`: Number of sweeps to perform.
- `--logging`: Configuration for logging, including the use of wandb.

**Note:** For logging, make sure to set `project` and `entity` under `wandb` in the config file.

</details>

<details>
<summary><h3>Testing</h3></summary>
To manually test a model, run the following command:

```sh
python test.py --config_path config.yaml
```

### Config File Explanation

Parameters:
- `--root`: Path to the dataset folder.
- `--batch_size`: Batch size for DataLoader.
- `--num_workers`: Number of workers for DataLoader.
- `--model_path`: Path to the model .pkl file.
- `--data_type`: 's' for synthetic, 'a' for authentic.

**Note:** If `data_type == 'a'`, the script will test the authentic test set (change also `root` to `test_200`). If `data_type == 's'`, the script will test the synthetic dataset (change `root` to `test_70`).

</details>

## Additional Scripts

### Inference Script

**Parameters:**
- `--model_path`: Path to the model .pkl file.
- `--images_path`: Path to the folder containing images.
- `--csv_path`: Path to save the output CSV file.
- `--batch_size`: Batch size for DataLoader.
- `--num_workers`: Number of workers for DataLoader.

### SSIM Script

**Parameters:**
- `--original_path`: Path to the folder containing original images.
- `--distorted_path`: Path to the folder containing distorted images.
- `--csv_path`: Path to save the output CSV file.
- `--batch_size`: Batch size for processing (not used in current implementation).
- `--num_workers`: Number of workers for processing (not used in current implementation).

### ARNIQA Script

**Parameters:**
- `--root`: Root folder containing the images to be evaluated.
- `--regressor_dataset`: Dataset used to train the regressor.
- `--output_csv`: Output CSV file to save the quality predictions.

### Single Image Inference Script

**Parameters:**
- `--image_path`: Path to the image to be evaluated.
- `--model_path`: Path to the model .pkl file.
```