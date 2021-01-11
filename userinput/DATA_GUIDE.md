## Dataset guidelines

### Default

1. The model automatically uses the EMNIST ByMerge dataset if no custom dataset is specified.

### Custom Dataset

1. Create the directory `userinput/your_folder_name/`, your_folder_name can be any folder name without spaces.
2. Put all images of your custom dataset into the created folder (`userinput/your_folder_name/`).
3. Run the train.py command with `your_folder_name` as argument, so command: `python train.py your_folder_name`.
