import numpy as np
import pandas as pd

def change_labels_to_numeric(filepath):
    data = pd.read_csv(filepath)

    data['Class'] = data['Class'].astype('category')
    data['Class'] = data['Class'].cat.codes

    data.to_csv(filepath)

def change_label_for_dataset_batch(filepath_list):
    for dataset in filepath_list:
        change_labels_to_numeric(dataset)

