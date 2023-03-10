
#Requirement
#-----
#Python 3.6

#Tensorflow 1.14

#Keras 2.2

#transfer csv to h5 file

filename = "./dataset.csv"
adata = sc.read_csv(filename, first_column_names=True)
adata.write('./dataset.h5')

#command:

python zidpkm.py --dataname "dataset" # "dataset" is data name (h5 format) 

