# Automated Image Quality Assessment in Teledermatology

## Overview

### Abstract

This research focuses on developing and evaluating automated methods to assess image quality in the context of teledermatology. Teledermatology, a growing field within telemedicine, allows patients to receive dermatological consultations remotely by sending photos of their skin conditions to dermatologists. However, the success of these remote consultations largely depends on the quality of the images provided. Poor-quality images can lead to misdiagnosis or require patients to resend images, causing delays and frustration. This research examines different image quality assessment (IQA) techniques for teledermatology. The goal is to make sure that only good quality images are sent to dermatologists, thereby improving their ability to make accurate medical diagnoses. By implementing these IQA techniques, the aim is to simplify the process, reduce back-and-forth communication, and make teledermatology more efficient and reliable.

The methodology involved a detailed review of existing IQA methods, followed by the development of a synthetic distortion pipeline to create a wide range of training datasets from good quality dermatological images. These images were gathered from the Fitzpatrick17k and SCIN datasets. For feature extraction, the state-of-the-art approach called ARNIQA was used, and different machine learning models, including XGBRegressor, XGBClassifier, MLP Regressor, and MLP Classifier, were trained to assess image quality based on seven dermatology quality criteria: lighting, background, field of view, orientation, focus, resolution, and color calibration. The models were trained on synthetic distortions and validated on both synthetically distorted and real-world dermatology images. Performance metrics such as Mean Absolute Error (MAE), R-squared ($R^{2}$), Spearmans Rank Order Correlation Coefficient (SRCC), and Cohens Kappa were used for evaluation.

The results showed that the automated IQA methods can assess image quality in the context of teledermatology, closely matching human evaluations and therefore providing reliable feedback on image quality. The final model achieved good performance across multiple dermatology quality criteria, improving the reliability and effectiveness of teledermatology services. This research highlights the potential of automated IQA systems to improve the accuracy of diagnoses and patient care in remote dermatological consultations.

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
conda create -n IQA -y python=3.10
conda activate IQA
chmod +x install_requirements.sh
./install_requirements.sh
```

#### Data Preparation

For the filtered good quality images, contact me and place them under the `datas` directory.

For the original images, they can be downloaded from here:
1. [**SCIN**](https://github.com/google-research-datasets/scin)
2. [**Fitzpatrick17k**](https://github.com/mattgroh/fitzpatrick17k)

At the end, the directory structure should look like this:

```
├── datas
|    ├── COMB
|    |   ├── embeddings
|    |   ├── ... (950 good quality images)
|    ├── F17K
|    |   ├── embeddings
|    |   ├── ... (475 good quality images)
|    ├── SCIN
|    |   ├── embeddings
|    |   ├── ... (475 good quality images)
|    ├── test_70 (synthetic test set)
|    |   ├── distorted
|    |   ├── embeddings
|    |   ├── ... (70 good quality images)
|    ├── test_200 (authentic test set)
|    |   ├── embeddings
|    |   ├── scores.json
|    |   ├── ... (200 images)
|    ├── ood
```

</details>

<details>
<summary><h3>Single Image Inference</h3></summary>
To perform inference on a single image, run the following command:

```sh
python single_image_inference.py --config_path config.yaml
```

**Parameters:**
- `--image_path`: Path to the image to be evaluated.
- `--model_path`: Path to the model.

</details>

<details>
<summary><h3>Training</h3></summary>

To train the model, run the following command:

```sh
python train.py --config_path config.yaml
```

**Parameters:**
- `--root`: Path to the dataset folder.
- `--num_distortions`: Number of distortions to use.
- `--batch_size`: Batch size for DataLoader.
- `--num_workers`: Number of workers for DataLoader.
- `--model_type`: Model type to use ('xgb_reg', 'xgb_cls', 'mlp_reg', 'mlp_cls').
- `--sweep`: Set to true for hyperparameter sweep.
- `--sweep_count`: Number of sweeps to perform.
- `--model_save`: Set to true to save the trained model.
- `--model_save_path`: Path to save the final model.
- `--plot_results`: Set to true to enable plotting of results.
- `--logging`: Configuration for logging, including the use of wandb.

**Note:** For logging, make sure to set `project` and `entity` under `wandb` in the config file.

</details>

<details>
<summary><h3>Testing</h3></summary>
To test the model and print the radar plot, run the following command:

```sh
python test.py --config_path config.yaml
```

**Parameters:**
- `--root`: Path to the dataset folder.
- `--batch_size`: Batch size for DataLoader.
- `--num_workers`: Number of workers for DataLoader.
- `--model_path`: Path to the model .pkl file.
- `--data_type`: 's' for synthetic, 'a' for authentic.

**Note:** If `data_type == 'a'`, the script will test the authentic test set (change also `root` to `test_200`). If `data_type == 's'`, the script will test the synthetic dataset (change `root` to `test_70`).

</details>

<details>
<summary><h3>Additional Scripts</h3></summary>

### Inference Script

This script is used to perform inference on a folder containing images and save the results in a CSV file.

**Parameters:**
- `--model_path`: Path to the model .pkl file.
- `--images_path`: Path to the folder containing images.
- `--csv_path`: Path to save the output CSV file.
- `--batch_size`: Batch size for DataLoader.
- `--num_workers`: Number of workers for DataLoader.

### Structural Similarity Index Measure (SSIM) Script

This script is used to calculate the SSIM between two folders containing original and distorted images. The scores are saved in a CSV file. The scores are inverted to match the quality scores used in the research.

**Parameters:**
- `--original_path`: Path to the folder containing original images.
- `--distorted_path`: Path to the folder containing distorted images.
- `--csv_path`: Path to save the output CSV file.
- `--batch_size`: Batch size for processing (not used in current implementation).
- `--num_workers`: Number of workers for processing (not used in current implementation).

### ARNIQA Script

This script runs the ARNIQA model on a folder containing images and saves the scores in a CSV file. The score are inverted to match the quality scores used in the research.

**Parameters:**
- `--root`: Root folder containing the images to be evaluated.
- `--regressor_dataset`: Dataset used to train the regressor.
- `--output_csv`: Output CSV file to save the quality predictions.
</details>

<details>
<summary><h3>Playground Notebooks</h3></summary>

#### `create_distortions.ipynb`
This notebook is used to create synthetic distortions based on the seven dermatology quality criteria. It allows you to select a folder containing good quality images and apply various distortions to create a wide range of training datasets. The distorted images are saved in the same folder for later use in training and evaluation.

#### `create_labels.ipynb`
In this notebook, you can manually assign distortion scores to a range of images. It prompts you to rate each of the seven dermatology criteria from 0 (no distortion) to 1 (high distortion). The scores are saved in a JSON file for later use in training and evaluation.

#### `create_plots.ipynb`
This notebook is used for generating several plots to visualize model performance and comparison. It includes tools for plotting results that are crucial for understanding how well the models are performing.

</details>
