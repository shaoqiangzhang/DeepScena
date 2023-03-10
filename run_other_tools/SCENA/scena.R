## Install the SCENA package

install.packages("devtools")
devtools::install_github("shaoqiangzhang/SCENA")

install.packages("gpuR")

library(SCENA)

Express=read.csv("./dataset.csv", header = T,row.names = 1)

Express=datapreprocess(Express,log=T)

##do clustering in parallel with 5 CPU cores

cc=scena_cpu(Express,T=20,num=6)##"num" is the number of clusters "T" is the the number of matrix iterations

#library(gpuR)
#source('./parallelclust_gpu.R')
## Because the GPU code cannot be called from the installed SCENA package directly, please copy it to your working path and run it using ¡¯source'
#cl <- makeCluster(5)
#parLapply(cl,1:5,K=10,T=50,X1=50,X2=100,X3=150,X4=200,X5=250,Express=Express,select_features_GPU)
#stopCluster(cl)

