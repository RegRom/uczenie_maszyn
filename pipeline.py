# %%

import numpy as np
import pandas as pd
import data_preparation as prep
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import StratifiedKFold

# %%

datasets = ['yeast3.csv', 
    'yeast-2_vs_8.csv', 
    'yeast-2_vs_4.csv',
    'yeast-0-5-6-7-9_vs_4.csv',
    'wisconsin.csv',
    'vehicle1.csv',
    'vehicle0.csv',
    'shuttle-c2-vs-c4.csv',
    'shuttle-c0-vs-c4.csv',
    'segment0.csv',
    'page-blocks0.csv',
    'new-thyroid1.csv',
    'iris0.csv',
    'haberman.csv',
    'glass6.csv',
    'glass1.csv',
    'glass-0-1-6_vs_2.csv',
    'ecoli3.csv',
    'abalone19.csv',
    'abalone9-18.csv'
]
# Zamiana etykiet tekstowych na liczbowe
# prep.change_label_for_dataset_batch('datasets\\', datasets)

# %%

# Wczytanie przekształconych datasetów do słownika
datasets_list = prep.load_datasets_batch('datasets\\', datasets)

# Dodanie wygenerowanych zbiorów danych do słownika
datasets_list['generated1'] = pd.read_csv('datasets\\11_features_0.035_0.965.csv')
datasets_list['generated2'] = pd.read_csv('datasets\\12_features_0.02_0.98.csv')
datasets_list['generated3'] = pd.read_csv('datasets\\13_features_0.014_0.986.csv')
datasets_list['generated4'] = pd.read_csv('datasets\\14_features_0.011_0.989.csv')
datasets_list['generated5'] = pd.read_csv('datasets\\15_features_0.009_0.991.csv')

# prep.change_labels_to_numeric('datasets\\yeast3.csv')

# %%

print(datasets_list.keys())

X, y = prep.data_label_split(datasets_list['yeast3.csv'].to_numpy())
print(X)

# %%

