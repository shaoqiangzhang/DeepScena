# DeepScena
Deep clustering of scRNA-seq data

# 1. Installation
## Start by installing the necessary packages
The code runs with Python version 3.9 and Pytorch 1.7.1. Assuming Anaconda, the virtual environment can be installed using:

```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```
You can also follow the commands provided on the Pytorch official website (https://pytorch.org/get-started/previous-versions/) according to your CUDA version. 

```
conda install numpy pandas scikit-learn
```
```
conda install -c conda-forge scanpy python-igraph leidenalg
```
See the requirements.txt file for an overview of the packages in the environment we used to produce our results.
