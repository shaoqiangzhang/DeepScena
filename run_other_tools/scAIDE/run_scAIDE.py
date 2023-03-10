import pandas as pd
import numpy as np

#Required Installations
#Python3: tensorflow 1.14, numpy, scipy, scikit_learn, tqdm, seaborn
#install AIDE:  [https://github.com/tinglabs/aide](https://github.com/tinglabs/aide)
#install RPH-kmeans: [https://github.com/tinglabs/rph_kmeans](https://github.com/tinglabs/rph_kmeans)

# the input is configured as n cells (rows) by m genes (columns).
sc_data = pd.read_csv("single_cell_dataset.csv", index_col=0)
sc_data = sc_data.values.astype('float32') # type = np.ndarray

# Configuring AIDE parameters:
from aide import AIDE, AIDEConfig
config = AIDEConfig()

#tune the following 4 parameters with default values
config.pretrain_step_num = 1000 # Pretrain step
config.ae_drop_out_rate = 0.4 # Dropout rate
config.alpha = 12.0 # A parameter that determines the portion of AE loss vs MDS encoder loss
config.early_stop = True# Early stop (maximum step number = 20000, minimum step number = 4000)

# Running AIDE:
encoder = AIDE(name = "sc_test", save_folder = "sc_test_folder")
sc_embedding = encoder.fit_transform(sc_data, config=config)

# save embedding
np.savetxt("~/sc_embedding.txt", sc_embedding)

#run RPH-kmeans
from rph_kmeans import RPHKMeans

K=10 #The number of clusters K for RPH-kmeans in scAIDE is set as that in the original study. 
clt = RPHKmeans(n_init=10, n_clusters=K, max_point = 2000)
clt_labels = clt.fit_predict(sc_embedding)

# Output results
