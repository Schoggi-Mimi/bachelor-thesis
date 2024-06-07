Here’s a detailed readme that elaborates on your project structure, usage, and execution instructions:

---

# Automated Image Quality Assessment in Teledermatology

## Overview

### Abstract

This research focuses on developing and evaluating automated methods to assess image quality in the context of teledermatology. Teledermatology, a growing field within telemedicine, allows patients to receive dermatological consultations remotely by sending photos of their skin conditions to dermatologists. However, the success of these remote consultations largely depends on the quality of the images provided. Poor-quality images can lead to misdiagnosis or require patients to resend images, causing delays and frustration. This research examines different image quality assessment (IQA) techniques for teledermatology. The goal is to make sure that only good quality images are sent to dermatologists, thereby improving their ability to make accurate medical diagnoses. By implementing these IQA techniques, the aim is to simplify the process, reduce back-and-forth communication, and make teledermatology more efficient and reliable.

The methodology involved a detailed review of existing IQA methods, followed by the development of a synthetic distortion pipeline to create a wide range of training datasets from good quality dermatological images. These images were gathered from the Fitzpatrick17k and SCIN datasets. For feature extraction, the state-of-the-art approach called ARNIQA was used, and different machine learning models, including XGBRegressor, XGBClassifier, MLP Regressor, and MLP Classifier, were trained to assess image quality based on seven dermatology quality criteria: lighting, background, field of view, orientation, focus, resolution, and color calibration. The models were trained on synthetic distortions and validated on both synthetically distorted and real-world dermatology images. Performance metrics such as Mean Absolute Error (MAE), R-squared (R$^{2}$), Spearman’s Rank Order Correlation Coefficient (SRCC), and Cohen’s Kappa were used for evaluation.

The results showed that the automated IQA methods can assess image quality in the context of teledermatology, closely matching human evaluations and therefore providing reliable feedback on image quality. The final model achieved good performance across multiple dermatology quality criteria, improving the reliability and effectiveness of teledermatology services. This research highlights the potential of automated IQA systems to improve the accuracy of diagnoses and patient care in remote dermatological consultations.

## Usage

<details>
<summary><h3>Getting Started</h3></summary>

#### Installation

I recommend using the [**Anaconda**](https://www.anaconda.com/) package manager to avoid dependency/reproducibility problems. For Linux systems, you can find a conda installation guide [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

1. **Clone the repository**

   First, clone the repository to your local machine using the following command:

   ```sh
   git clone https://github.com/Schoggi-Mimi/bachelor-thesis
   ```

2. **Create and activate a conda environment**

   Create a new conda environment named `IQA` with Python 3.10:

   ```sh
   conda create -n IQA -y python=3.10
   conda activate IQA
   ```

3. **Install required dependencies**

   Make the installation script executable and run it to install the necessary packages:

   ```sh
   chmod +x install_requirements.sh
   ./install_requirements.sh
   ```

#### Data Preparation

Prepare the dataset by organizing it into the `datas` directory. This directory should contain subdirectories for different datasets and their respective embeddings.

1. **Filtered good quality images**: These images are available upon request and should be placed under the `datas` directory.

2. **Original images**: Download the datasets from the following sources:
   - [**SCIN**](https://github.com/google-research-datasets/scin)
   - [**Fitzpatrick17k**](https://github.com/mattgroh/fitzpatrick17k)

   After downloading, organize the data as follows:

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

To perform inference on a single image and assess its quality, run the following command:

```sh
python single_image_inference.py --config_path config.yaml
```

**Parameters:**
- `--image_path`: Path to the image to be evaluated.
- `--model_path`: Path to the pre-trained model file.

This script will load the specified image and use the model to evaluate its quality based on the selected criteria.

</details>

<details>
<summary><h3>Training</h3></summary>

To train a new model using the provided configuration file, run the following command:

```sh
python train.py --config_path config.yaml
```

**Parameters:**
- `--root`: Path to the dataset folder.
- `--num_distortions`: Number of distortions to apply during training.
- `--batch_size`: Batch size for the DataLoader.
- `--num_workers`: Number of worker threads for the DataLoader.
- `--model_type`: Type of model to train (`xgb_reg`, `xgb_cls`, `mlp_reg`, `mlp_cls`).
- `--sweep`: Set to `true` to enable hyperparameter sweeping.
- `--sweep_count`: Number of sweeps to perform during hyperparameter tuning.
- `--model_save`: Set to `true` to save the trained model.
- `--model_save_path`: Path where the final trained model should be saved.
- `--plot_results`: Set to `true` to enable plotting of results.
- `--logging`: Configuration for logging, including the use of wandb for monitoring training.

**Note:** If using wandb for logging, ensure the `project` and `entity` fields are correctly set in the `config.yaml` file under `wandb`.

This process trains the model on the specified dataset, applying the given distortions, and evaluates its performance based on the chosen criteria.

</details>

<details>
<summary><h3>Testing</h3></summary>

To test the trained model and generate performance metrics and visualizations, use the following command:

```sh
python test.py --config_path config.yaml
```

**Parameters:**
- `--root`: Path to the dataset folder for testing.
- `--batch_size`: Batch size for the DataLoader.
- `--num_workers`: Number of worker threads for the DataLoader.
- `--model_path`: Path to the trained model `.pkl` file.
- `--data_type`: Specify `'s'` for synthetic data or `'a'` for authentic data.

**Note:** The `root` parameter should be set to the appropriate test set directory (`test_200` for authentic or `test_70` for synthetic).

This command runs the test suite on the specified dataset, evaluating the model's performance on either the synthetic or authentic test sets.

</details>

<details>
<summary><h3>Additional Scripts</h3></summary>

### Inference Script

This script performs batch inference on a folder of images, saving the results in a CSV file.

**Parameters:**
- `--model_path`: Path to the trained model `.pkl` file.
- `--images_path`: Path to the directory containing images for batch inference.
- `--csv_path`: Path to save the results in a CSV file.
- `--batch_size`: Batch size for processing images.
- `--num_workers`: Number of worker threads for processing.

### Structural Similarity Index Measure (SSIM) Script

This script calculates the Structural Similarity Index (SSIM) between two sets of images, typically an original and its distorted version. Results are saved in a CSV file.

**Parameters:**
- `--original_path`: Path to the folder with original images.
- `--distorted_path`: Path to the folder with distorted images.
- `--csv_path`: Path to save the SSIM scores in a CSV file.
- `--batch_size`: Batch size for processing (not used in current implementation).
- `--num_workers`: Number of worker threads for processing (not used in current implementation).

### ARNIQA Script

This script runs the ARNIQA model on a set of images and saves the predicted quality scores in a CSV file.

**Parameters:**
- `--root`: Path to the folder containing the images.
- `--regressor_dataset`: Dataset used for training the regressor.
- `--output_csv`: Path to save the predicted quality scores in a CSV file.

</details>

<details>
<summary><h3>Playground Notebooks</h3></summary>

### `create_distortions.ip

ynb`
This notebook allows you to generate synthetic distortions on images based on the seven dermatology quality criteria. You can select a folder of good quality images and apply various distortions to create training datasets. The distorted images are saved in the same folder and are used for training and evaluation.

### `create_labels.ipynb`
In this notebook, you can manually assign distortion scores to images. It prompts you to rate each image on the seven dermatology criteria from 0 (no distortion) to 1 (high distortion). The scores are saved in a JSON file for later use in training and evaluation.

### `create_plots.ipynb`
This notebook provides tools for generating various plots to visualize and compare model performance. It helps in understanding the effectiveness of the models in assessing image quality.

</details>

--- 

This readme provides detailed instructions on how to set up, train, and evaluate your IQA models for teledermatology, along with descriptions of additional scripts and notebooks that support the project.