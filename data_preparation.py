from operator import index
import numpy as np
import pandas as pd

def change_labels_to_numeric(filepath):
    data = pd.read_csv(filepath)

    # data['Class'] = data['Class'].astype('category')
    # data['Class'] = data['Class'].cat.codes

    data["Class"] = np.where(data["Class"].str.contains("positive"), 1, 0)

    data.to_csv(filepath, index=False)

def change_label_for_dataset_batch(filepath, dataset_names):
    for dataset in dataset_names:
        change_labels_to_numeric(f'{filepath}{dataset}')

def load_datasets_batch(filepath, dataset_names):
    datasets_list = {}
    for dataset in dataset_names:
        print(f'{filepath}{dataset}')
        dataset_loaded = pd.read_csv(filepath_or_buffer=f'{filepath}{dataset}', skiprows=1, header=None)
        datasets_list[f'{dataset}'] = dataset_loaded
    
    return datasets_list
