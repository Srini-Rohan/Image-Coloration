# Pix2Pix-GAN  Image Coloration 

## Overview
This project implements the Pix2Pix Generative Adversarial Network (GAN) for image coloration. Pix2Pix is a type of conditional GAN that can be used for image-to-image translation tasks. In this specific implementation, the Pix2Pix algorithm is applied to convert grayscale images to colored versions.

## Getting Started
### Libraries used
- TensorFlow
- NumPy
- OpenCV
- Matplotlib

### Installation
Clone the repository and install the required dependencies:
```bash
git clone 
cd pix2pix-coloration
pip install -r requirements.txt
```

## Project Structure
The project is organized into three main files:

### 1. Pix2PixGAN.py
This file contains the implementation of the Pix2Pix GAN. The GAN consists of a generator and a discriminator, each defined as separate functions within the Pix2PixGAN class. The generator is responsible for transforming input grayscale images into colorized versions, while the discriminator evaluates the authenticity of the generated images.

### 2. Dataset.py
This file provides utility functions for preparing the dataset. The Dataset function loads and preprocesses color and grayscale images from specified directories. It returns train and test datasets for both color and grayscale images.

### 3. utils.py
The utils.py file contains utility functions for generating and displaying images using the trained model. The generate_images function takes a model, test input, and target images as input and displays the input, ground truth, and predicted images using Matplotlib.

```python
from utils import generate_images
generate_images(model, test_input, target)
```

## Training
The training process is encapsulated in the train_step method of the Pix2PixGAN class. The model is trained using the provided dataset 10 epochs.

```python
from Pix2PixGAN import Pix2PixGAN
from DatasetUtils import Dataset

# Initialize Pix2PixGAN with input size
gan = Pix2PixGAN(input_size=(256, 256, 1))

# Load and preprocess the dataset
(train_color_ds, train_gray_ds), (test_color_ds, test_gray_ds) = Dataset(path='path/to/dataset', size=256)

# Train the model
gan.fit(train_ds=train_color_ds, epochs=10)
```

## Results
### Epoch 0
![Epoch0](https://github.com/Srini-Rohan/Image-Coloration/assets/76437900/3d4a252a-1b26-4ed0-992f-e33df628a3c4)

### Epoch 10
![Results](https://github.com/Srini-Rohan/Image-Coloration/assets/76437900/0467222f-e023-48b6-a484-16cb08cab586)
