###scGMAI uses the following dependencies:

#python = 3.6
#numpy = 1.16.3
#scipy = 1.4.1
#pandas = 0.25.3
#scikit-learn = 0.22.1
#tensorflow = 1.13.1
#matplotlib = 3.0.3
#R = 3.6


#preprocess data

python Preprocessing.py

#imputation data are obtained by autoencoder. 

python autoencoderRunner.py

#dimension reduction by FastICA. 

python FastICA.py

#GaussianMixture clustering 

python GaussianMixture_clustering.py

