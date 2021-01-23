# %%

import numpy as np
import pandas as pd
import data_preparation as prep

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
datasets_list = 

# prep.change_labels_to_numeric('datasets\\yeast3.csv')

# %%

print(datasets_list['yeast3.csv'])

# %%
