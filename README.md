# DeepScena
A self-supervised deep clustering of large scRNA-seq data

## Install packages
The code of DeepScena runs with Python version 3.9 and Pytorch==1.7.1. And you should have CUDA installed.

Please create a Pytorch environment, install Pytorch and some other packages, such as "**numpy**","**pandas**", "**scikit-learn**", "**scanpy**", and "**cudatoolkit**". See the __requirements.txt__ and __intall.txt__ file for an overview of the packages in the environment we used to produce our results.

## Data preprocessing

### Run the following command line to preprocess UMI-count data:
```
python preprocess_data.py -p Data/ -i pbmc4340.txt -o pbmcpre.csv
```
or 
```
python preprocess_data.py --filepath Data/ --filename pbmc4340.txt --resultfile pbmcpre.csv
```
The program accepts "**.txt**", "**.csv**", and **10x_mtx** files as input file name. 

### If the scRNA-seq data is read-count, please add a parameter "-r" or "--reads" as follows
```
python preprocess_data.py -p FILE_PATH/ -i FILE_NAME -o OUTPUT_FILE.csv -r
```
You can type command "**python preprocess_data.py -h**" for usage help. 

## Training and clustering

```
python runDeepScena.py
```
Before running DeepScena, please modify your file path and names (two files: the preprocessed scRNA-seq data file and its cell-type file), rename your dataset name (e.g. dataset_name = 'pbmc'), set the number of clusters (e.g. number_cluster = 8), and reset some other parameters such as batch size and maximum iterations. 

The latent low-dimensional space will be saved as "dataset_name"+"_uspace.csv" (e.g. pbmc_upsace.csv)， which can be used for visualization.

The clustering result file will be saved as "dataset_name"+"_clusters.csv" (e.g. pbmc_clusters.csv).

## Citation
Please cite our paper if you use the code.
