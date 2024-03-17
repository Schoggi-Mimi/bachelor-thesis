# Synthetically Degraded Datasets

## KADID IQA Database [KADID](http://database.mmsp-kn.de/kadid-10k-database.html)

**Description:** KADID-10k is a large-scale artificially distorted image quality assessment (IQA) database. It consists of two datasets, the Konstanz artificially distorted image quality database (KADID-10k) and the Konstanz artificially distorted image quality set (KADIS-700k). KADID-10k contains 81 pristine images, each degraded by 25 distortions in 5 levels. Through crowdsourcing, subjective IQA study on KADID-10k was conducted, obtaining 30 degradation category ratings (DCRs) per image.

### KADID-10k Database

- **Size:** ~3.1GB
- **Contents:**
  - **"image" folder:** Contains 81 reference images and 10,125 distorted images (81 reference images x 25 types of distortion x 5 levels of distortions), saved as PNG format. Distorted image name format: Ixx_yy_zz.png.
  - **"dmos.csv" file:** Contains the differential mean opinion score (DMOS) and variance for each distorted image. DMOS range: [1, 5].

### KADIS-700 Dataset

- **Size:** ~44.6GB
- **Contents:**
  - **"ref_imgs" folder:** Contains 140,000 reference images.
  - **"dist_imgs" folder:** Empty, intended for saving 700,000 distorted images.
  - **"code_imdistort" folder:** MATLAB code for all 25 types of distortions.
  - **"kadis700k_ref_img.csv" file:** Information for generating 700,000 distorted images.
  - **"generate_kadis700.m" file:** Script to generate 700,000 distorted images.

### Distortion Types
- **Blurs:** Gaussian blur, Lens blur, Motion blur
- **Color distortions:** Color diffusion, Color shift, Color quantization, Color saturation
- **Compression:** JPEG2000, JPEG
- **Noise:** White noise, White noise in color component, Impulse noise, Multiplicative noise, Denoise
- **Brightness change:** Brighten, Darken, Mean shift
- **Spatial distortions:** Jitter, Non-eccentricity patch, Pixelate, Quantization, Color block
- **Sharpness and contrast:** High sharpen, Contrast change


## LIVE IQA Database [Release 2](https://utexas.box.com/v/databaserelease2)

**Description:** The LIVE Image Quality Assessment Database is a dataset created through extensive subjective experiments to obtain scores from human subjects for images distorted with different distortion types. This dataset is valuable for calibrating Quality Assessment algorithms and testing their performance.

### Release 2 Details:

- **Differences from Release 1:**
  - More distortion types:
    - JPEG compressed images (169 images).
    - JPEG2000 compressed images (175 images).
    - Gaussian blur (145 images).
    - White noise (145 images).
    - Bit errors in JPEG2000 bit stream (145 images).
  - More subjects for JPEG distortion.
  - DMOS values instead of MOS values for distorted images.
  - Different processing of raw scores than in Release 1.

## CSIQ Database [CSIQ Dataset](https://s2.smu.edu/~eclarson/csiq.html)

**Description:** The CATEGORICAL SUBJECTIVE IMAGE QUALITY (CSIQ) database consists of 30 original images, each distorted using six different types of distortions at four to five different levels. These images were rated subjectively by 35 different observers, resulting in 5000 subjective ratings reported in the form of DMOS (Differential Mean Opinion Score).


## TID2013 Database [TID2013 WinRAR archive (~913 MB)](https://www.ponomarenko.info/tid2013/tid2013.rar)

**Description:** Tampere Image Database 2013 is designed for the evaluation of full-reference image visual quality assessment metrics. It allows for estimating how a given metric corresponds to mean human perception. The dataset contains 25 reference images and 3000 distorted images (25 reference images x 24 types of distortions x 5 levels of distortions). The Mean Opinion Score (MOS) for each distorted image is provided in the file "mos.txt".

- **Contents:**
  - 25 reference images cropped from Kodak Lossless True Color Image Suite.
  - 3000 distorted images, named in the format "iXX_YY_Z.bmp", indicating the reference image number, distortion type number, and distortion level number.
  - File "mos.txt" contains the Mean Opinion Score for each distorted image, obtained from 971 observers from five countries: Finland, France, Italy, Ukraine, and USA.

# Authentic Degraded Datasets

## FLIVE Database [FLIVE Dataset on GitHub](https://github.com/niu-haoran/FLIVE_Database)

**Description:** FLIVE (Face Livestreaming Image & Video Evaluation) is the largest existing dataset for No-Reference Image Quality Assessment (NRIQA), consisting of nearly 40,000 real-world images.

## Smartphone Photography Attribute and Quality (SPAQ) Database [SPAQ Dataset](https://github.com/h4nwei/SPAQ)

**Description:** The SPAQ database is designed for the perceptual quality assessment of smartphone photography. It comprises 11,125 pictures taken by 66 smartphones, with rich annotations including image quality, image attributes (brightness, colorfulness, contrast, noisiness, and sharpness), and scene category labels. EXIF data for all images are recorded to aid deeper analysis. The database also includes human opinions collected in a controlled laboratory environment. Additionally, the database is utilized to train blind image quality assessment (BIQA) models using baseline and multi-task deep neural networks.
