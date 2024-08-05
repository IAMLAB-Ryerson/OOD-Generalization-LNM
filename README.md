# OOD-Generalization-LNM

# Installation Instructions
Create a new virtual environment: ```conda create -n <your_environment_name> python=3.9.15```

Activate the environment: ```conda activate <your_environment_name>```

This project has only been tested on:

```torch 1.13.0```

```torchvision 0.14.0```

```cuda 11.7```

```cudnn 8.0```

Example installation of the above packages: 

```pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html```

Please make sure these are installed on your system prior to installing additional dependencies with:

 ```pip install -r requirements.txt```

Also, users must add their openslide "bin" directory path to ```config.ini``` due to the tiatoolbox dependency for stain normalization. For more information, visit the "Binaries" section on the OpenSlide website at https://openslide.org/download/#binaries.

# Usage
```inference.py``` has five inputs:

```--cluster_type``` - The cluster type as described in the paper. Pick one from ["resnet_18_hist", "resnet_18_imagenet", "single"].

```--partial_type``` - The partial tumour percentage in the training dataset. Pick one from ["0", "10", "20", "30", "40", "50", "natural"]

```--fold``` - The fold number. Pick one from ["1", "2", "3", "4", "5"]

```--input_directory``` - The directory structure of the input patches. See [Directory Structure](#directory-structure) for more information.

```--output_directory``` - The path to the output directory. [Default: ```./"outputs"```]

# Directory Structure
Before running ```inference.py```, ensure your input directory (--input_directory) is organized as follows:

```
input_directory/
    ├── WSI_1/
    │   ├── patch1
    │   ├── patch2
    │   └── ...
    ├── WSI_2/
    │   ├── patch1
    │   ├── patch2
    │   └── ...
    └── ...
```

Each WSI should have its own subdirectory containing its respective patches.


# Example
Make predictions with fold 1 of the Histopathology-40% model:

```python inference.py --cluster_type resnet_18_hist --partial_type 40 --fold 1 --input_directory "path_to_input_directory" --output_directory "path_to_output_directory"```