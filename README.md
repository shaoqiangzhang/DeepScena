# DeepScena
A self-supervised deep clustering of large scRNA-seq data

## Install packages
The code of DeepScena runs with Python version 3.9 and Pytorch==1.7.1.
Please create a Pytorch environment, install Pytorch and some other packages, such as "**numpy**","**pandas**", "**scikit-learn**" and "**scanpy**". 
See the __requirements.txt__ and __intall.txt__ file for an overview of the packages in the environment we used to produce our results.

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
Before running DeepScena, please modify to your file path (two files: the preprocessed scRNA-seq data file and a cell-type file),
set the number of clusters (e.g. number_cluster=8), and reset some other parameters such as batch size and maximum iterations. 

## Citation
Please cite our paper if you use the code.
