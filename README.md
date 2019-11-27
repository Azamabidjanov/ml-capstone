# Machine-Learning Capstone Project.

## Authors: Natalie Kalin, Armando Ocampo, Jay Harrison, Azam Abidjanov

# Recycling Project

## Dataset
https://github.com/garythung/trashnet

## Data Parsing
https://towardsdatascience.com/how-to-build-an-image-classifier-for-waste-sorting-6d11d3c9c478

https://nbviewer.jupyter.org/github/collindching/Waste-Sorter/blob/master/Waste%20sorter.ipynb

## Saad's Website Intro Code

https://github.com/minds-mines/intro-ml-code-samples/tree/complete

## Development Environment - `pipenv`
We are using pipenv to keep our development/dependencies consistent.
Pipenv uses Pipfile/Pipfile.lock to ensure we are using the same version
of packages for our projects. This avoids "I swear it just worked on my machine!" type situations.
Whenever you are introducing a new package, please add it to the pipfile (instructions below).

## Pipenv installation
+ Clone the repo: `git clone`
+ In the repo directory, set your environment and dependencies: `pipenv install --dev`
+ Activate your environment: `pipenv shell`
+ Run python scripts from the activated shell: `python pull-data.py`

## Pipenv: Adding new packages
+ To add a new package (scipy for example) into the pipfile: `pipenv install scipy`

## Fastai Installation (In Pipenv shell)
To run the code that trains a fastai based classifier, please follow steps below after setting your environment with pipenv.
+ fastai is based on PyTorch, so we need to install pytorch first.
  + In the pipenv shell, run the pip command for your system: https://pytorch.org/get-started/locally/
+ Once Pytorch is installed, install fastai: `pip install fastai`
  + If running Windows & MS Visual Studio, you may experience a "io.h No such file or directory" error
  + Fix: https://stackoverflow.com/a/50210015
+ (Optional) If experiencing RAM shortage issues or want to use the GPU(nvidia)
  + Check if PyTorch will use the GPU. In python shell, after importing torch, run `torch.cuda.is_available()`
  + If false, check the cuda version `torch.version.cuda`. Then, check if your graphics card driver is compatible: https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver. Update your driver if necessary.  
