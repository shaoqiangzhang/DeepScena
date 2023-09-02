library(scCAN)
data=read.csv("zeisel_GSE60361.csv",header = T,row.names = 1)
 #rows are cells, columns are genes, otherwise transpose using t()
data<-t(data)

label=read.csv("zeisel_labels.csv",header = F,row.names = 1)
celltype=as.vector(label['V2'])
label=celltype$V2

if(max(data)>100) data <- log2(data + 1)
data <- as(data, "dgCMatrix")
result <- scCAN(data,sparse = T,ncores = 10,r.seed = 1)
cluster <- result$cluster

library(mclust)
adjustedRandIndex(label,cluster)

library(NMI)
label2=data.frame(1:3005, label)
cluster2=data.frame(1:3005,cluster)

NMI(label2,cluster2)
