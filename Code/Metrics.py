import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn import metrics
ari=metrics.adjusted_rand_score
nmi = normalized_mutual_info_score
def acc(y_true, y_pred, num_cluster):

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    print(y_pred.size)
    print(y_true.size)
    assert y_pred.size == y_true.size

    w = np.zeros((num_cluster, num_cluster))
    print(w)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        print(w[y_pred[i], y_true[i]])
    from scipy.optimize import linear_sum_assignment

    ind = linear_sum_assignment(w.max() - w)
    accuracy = 0.0
    for i in ind[0]:
        accuracy = accuracy + w[i, ind[1][i]]
    return accuracy / y_pred.size