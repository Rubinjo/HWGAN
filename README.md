# HWGAN: Handwritten Text Generator
<p align="center">
  <a aria-label="Python_shield" href="https://www.python.org/" target="_blank">
    <img alt="Made with Python" src="http://ForTheBadge.com/images/badges/made-with-python.svg" target="_blank" />
  </a>
</p>

This project was executed as a school assignment at the University of Twente. HWGAN is our own implementation of a handwriting replicator build with a GAN and OCR neural network. The neural networks in this project have been build with [tensorflow](https://github.com/tensorflow/tensorflow).

## Project Overview
- School: University of Twente
- Course: Machine Learning II
- Assignment Type: Open Project
- Group Size: 4

## Setup

1. Use Python 3.6-3.8.
2. Execute the following command to install required packages:

```
pip install -r ./helper/requirements.txt
```

3. For GPU support we recommend to also install CUDA Toolkit 11.0, cuDNN 8.0.4 and NVIDIA GPU Driver 450 or higher ([NVIDIA website](https://developer.nvidia.com/cuda-toolkit)).

## Usage

- For training the OCR and GAN models you use the following command:

```
python train.py
```

- For creating a word you use the following command:

```
python run.py
```

## File Structure

     dataset
         ├── DATA_GUIDE.md                      # How to add custom datasets
         ├── ...                                # Location to add custom datasets
     helper
         ├── requirements.txt                   # Configuration file with all dependencies to install
         ├── split_data.py                      # Split the dataset into letter specific data
         ├── ()                                 # Custom images are split into characters
     models
         ├── gan_model/                         # Holds all files related to the GAN model
             ├── gifs/
                 ├── ...                        # .gif files of the training process
             ├── graphs/
                 ├── ...                        # Loss graphs of the training process
             ├── saved_models/
                 ├── ...                        # Trained discriminator and generator models
             ├── GAN.py                         # GAN model
         ├── ocr_model/                         # Holds all files related to the OCR model
             ├── ocr_model.h5                   # Trained OCR model
             ├── OCR.py                         # OCR model
             ├── ...                            # Stats on the performance
     out
         ├── ...                                # Output of the run.py executable
     train.py                                   # Main executable - Train all models
     run.py                                     # Main executable - Generate given word
     ...                                        # Extra project files
     
## Acknowledgments
This project has already been executed by [ScrabbleGAN](https://github.com/amzn/convolutional-handwriting-gan), which is a more elaborate implementation of the same principal with [pytoch](https://github.com/pytorch/pytorch).
