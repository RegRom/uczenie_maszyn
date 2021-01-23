#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from strlearn.streams import StreamGenerator
from strlearn.metrics import balanced_accuracy_score
from strlearn.evaluators import TestThenTrain

np.set_printoptions(suppress=True, precision=3)
#%%
X1, Y1 = make_classification(n_features=100, n_redundant=0, n_informative=1, n_clusters_per_class=1, n_samples=10000)

print(X1.shape)
print(Y1.shape)

plt.scatter(X1[:, 0], X1[:, 1], c=Y1)

#%%
stream = StreamGenerator(
    n_chunks=1, 
    n_drifts=0,
    chunk_size=100,
    n_features=10
)

clf = GaussianNB()
metrics = [accuracy_score, balanced_accuracy_score]
evaluator = TestThenTrain(metrics)
