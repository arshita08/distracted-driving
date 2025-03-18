# Distracted Driver Detection

## Overview
This project develops a machine learning model to detect and classify distracted driving behaviors from images of drivers. Each image, captured inside a car, shows a driver engaged in activities such as texting, talking on the phone, eating, applying makeup, reaching behind, or talking to a passenger. The goal is to predict the likelihood of the specific activity (class) in each image.

### Example Image
![Distracted Driver Example](driver.gif)
*Animated GIF of a driver texting while operating a vehicle.*

## Problem Description
The dataset includes 10 distinct classes to predict:
- `c0`: Safe driving
- `c1`: Texting - right
- `c2`: Talking on the phone - right
- `c3`: Texting - left
- `c4`: Talking on the phone - left
- `c5`: Operating the radio
- `c6`: Drinking
- `c7`: Reaching behind
- `c8`: Hair and makeup
- `c9`: Talking to passenger

## Dataset
- **Training Set**: 22,424 images
- **Holdout Set**: 79,726 images
- **Image Dimensions**: Resized to 64x64 pixels with 3 channels (RGB)
- **Subjects**: 26 unique drivers, with image counts ranging from 346 to 1,237 per subject

### Data Distribution
- **Class Frequency**: Ranges from 1,911 (class `c8`) to 2,489 (class `c0`) images per class.
- **Subject Frequency**: Average of 862 images per subject.

## Methodology
### Model Architecture
- **Model**: 50-layer Residual Network (ResNet50)
- **Initializer**: Glorot Uniform
- **Optimizer**: Adam
- **Loss Function**: Sparse categorical cross-entropy
- **Metrics**: Accuracy

### Data Preprocessing
- Images are loaded, resized to 64x64x3, and converted to numerical arrays.
- Data is normalized by scaling pixel values to [0, 1].
- Labels are mapped to integers (0-9) and reshaped for training.
- Cross-validation uses Leave-One-Group-Out (LOGO) with subjects as groups to prevent overfitting.

### Training Process
- **Batch Size**: 32
- **Epochs**: Varied from 2 to 10
- **Cross-Validation**: LOGO to evaluate performance across subjects

## Results
The model was trained with different epoch counts, yielding the following performance metrics:

| Model | Epochs | Train Accuracy (%) | Dev Accuracy (%) | Bias (%) | Variance (%) |
|-------|--------|---------------------|------------------|----------|--------------|
| Model A | 2      | 27.91               | 21.19            | 72.09    | 6.72         |
| Model B | 5      | 37.83               | 25.79            | 62.17    | 12.04        |
| Model C | 10     | 86.95               | 40.68            | 13.06    | 46.27        |

- **Holdout Loss**: 2.64 (after 10 epochs)
- **Training Loss**: 0.93 (after 10 epochs)
- **Validation Loss**: 3.79 (after 10 epochs)

### Observations
- Training accuracy improves with more epochs but is accompanied by high variance (e.g., 46.27% at 10 epochs).
- High bias and variance suggest underfitting and overfitting issues.
- Resource constraints (RAM, CPU speed) limit hyper-parameter tuning.

## Why High Losses?
The elevated training, validation, and holdout losses are due to:
- Limited resources for extensive hyper-parameter tuning (e.g., grid searches).
- Insufficient model capacity and data augmentation due to computational limitations.

## Improvements
### Addressing High Bias
- Increase epoch count.
- Use a larger batch size (up to the number of examples).
- Deepen the network architecture.
- Increase image resolution (e.g., 128x128 or 256x256).
- Perform grid searches over parameters (batch size, epochs, optimizer, initializer).

### Addressing High Variance
- Augment images to increase dataset size.
- Apply regularization techniques.
- Conduct grid searches over parameters.
- Reduce dev set size for more training examples.
- Investigate and address classes with low accuracy.

## Next Steps
- Retrain the 10-epoch model on the full dataset and evaluate on the holdout set.
- Implement data augmentation and regularization to reduce variance.
- Explore higher-resolution images and deeper architectures if resources allow.
- Perform comprehensive hyper-parameter tuning.

## Usage
### Dependencies
- Python libraries: `numpy`, `pandas`, `tensorflow`, `keras`, `matplotlib`, `scipy`, `sklearn`, `PIL`
- Ensure `resnets_utils.py` is available for utility functions.

### Setup
1. Clone the repository: `git clone https://github.com/arshita08/distracted-driving.git`
2. Install dependencies: `pip install -r requirements.txt` (create `requirements.txt` with listed libraries).
3. Place dataset files (`driver_imgs_list.csv`, `test_file_names.csv`, and image folders) in the `imgs/` directory.

### Running the Code
1. Preprocess images using `CreateImgArray` and normalize with `Rescale`.
2. Build and train the ResNet50 model with provided functions (`identity_block`, `convolutional_block`, `ResNet50`).
3. Evaluate with LOGO cross-validation and predict on the holdout set.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests for any enhancements or bug fixes.
