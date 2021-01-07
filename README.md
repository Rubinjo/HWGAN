# Handwriting-Replicator

## Setup

1. Use Python 3.6-3.8.
2. Execute the following command to install required packages:

```
pip install -r ./helper/requirements.txt
```

3. For GPU support you should also install CUDA Toolkit 11.0, cuDNN 8.0.4 and NVIDIA GPU Driver 450 or higher.

## File Structure

     dataset
         ├── DATA_GUIDE.md                          # How to add custom datasets
         ├── ...                                    # Location to add custom datasets
     helper
         ├── requirements.txt                       # Configuration file with all dependencies to install
         ├── split_data.py                          # Split the dataset into letter specific data
         ├── ()                                     # Custom images are split into characters
     models
         ├── gan_model/
             ├── GAN.py                             # Create and train GAN model
             ├── g_model.h5                         # Trained generator model
             ├── d_model.h5                         # Trained discriminator model
             ├── ...                                # Stats on the performance of the GAN
         ├── ocr_model/
             ├── OCR.py                             # Create and train OCR model
             ├── ocr_model.h5                       # Trained OCR model
             ├── ...                                # Stats on the performance of the OCR
     out
         ├── ...                                    # Output of the executable
     main.py                                        # Core executable - train models
     ...                                            # Extra project files
