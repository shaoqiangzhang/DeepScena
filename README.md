# DeepScena
Deep clustering of scRNA-seq data

See the requirements.txt file for an overview of the packages in the environment we used to produce our results.

## Data preprocessing
Before preprocessing data, please install packages "**numpy**","**pandas**", and "**scanpy**". Assuming Anaconda, they can be installed using

```
conda install numpy pandas scikit-learn
```
```
conda install -c conda-forge scanpy python-igraph leidenalg
```
###Run the following command line to preprocess UMI-count data:
```
python preprocess_data.py -p Data/ -i pbmc4340.txt -o pbmcpre.csv
```
or 
```
python preprocess_data.py --filepath Data/ --filename pbmc4340.txt --resultfile pbmcpre.csv
```

If the scRNA-seq expression dataset is read-count, please add a parameter "**-r**" or "**--reads**", such as
```
python preprocess_data.py -p FILEPATH/ -i FILENAME -o OUTPUT.csv -r
```

## Training and clustering
The code of DeepScena runs with Python version 3.9 and Pytorch >=1.7.1. Assuming Anaconda, the virtual environment can be installed using:

```
conda install pytorch==1.7.1 torchvision==0.8.2  cudatoolkit=10.1 -c pytorch
```
You can also follow the commands provided on the Pytorch official website (https://pytorch.org/get-started/previous-versions/) according to your CUDA version. 
