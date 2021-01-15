## Dataset guidelines

Different datasets can be used to train our GAN model. Custom datasets are split by an image processing operation in which the images are labelled by our OCR model. Our OCR model itself can however currently not be trained on custom datasets. For more details on the specific implementation of different datasets have a look below.

### Default

1. The model automatically uses the [EMNIST ByMerge dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset) if no custom dataset is specified.

### Custom Dataset

1. Create a main dataset directory `dataset/folder_name/`, folder_name can be any folder name without spaces.
2. Put all images of your custom dataset into the created folder (`dataset/folder_name/`). Multiple subfolders in this main folder (`folder_name`) is also supported and should result in the same outcome.
3. Run the train.py command with `data` as argument, so command: `python train.py -data folder_name`. For more details on run arguments have a look at our [README](../README.md).
