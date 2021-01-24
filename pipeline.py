# %%

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import data_preparation as prep
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from sklearn.ensemble import VotingClassifier, StackingClassifier
from scipy.stats import wilcoxon
from tabulate import tabulate
import math
from sklearn.linear_model import LogisticRegression

skf = StratifiedKFold(n_splits=5, random_state=444)

# %%

datasets = [
    'glass1.csv',
    'wisconsin.csv',
    'iris0.csv',
    'haberman.csv',
    'vehicle1.csv',
    'vehicle0.csv',
    'new-thyroid1.csv',
    'segment0.csv',
    'glass6.csv',
    'yeast3.csv', 
    'ecoli3.csv',
    'page-blocks0.csv',
    'yeast-2_vs_4.csv',
    'yeast-0-5-6-7-9_vs_4.csv',
    'glass-0-1-6_vs_2.csv',
    'shuttle-c0-vs-c4.csv',
    'abalone9-18.csv',
    'shuttle-c2-vs-c4.csv',
    'yeast-2_vs_8.csv',
    '11_features_0.035_0.965.csv',
    '12_features_0.02_0.98.csv',
    '13_features_0.014_0.986.csv',
    '14_features_0.011_0.989.csv',
    '15_features_0.009_0.991.csv', 
    'abalone19.csv',
]

# %%

# Zamiana etykiet tekstowych na liczbowe
# prep.change_label_for_dataset_batch('datasets\\', datasets)

# # Zakodowanie kolumn z wartościami kategorycznymi jako liczbowe
# prep.label_encode_column('Sex', 'datasets\\abalone19.csv')
# prep.label_encode_column('Sex', 'datasets\\abalone9-18.csv')

# %%

# Wczytanie przekształconych datasetów do słownika
datasets_list = prep.load_datasets_batch('datasets\\', datasets)

# %%
# Funkcja przeprowadzająca eksperyment dla danego zbioru danych i klasyfikatora
# Zwraca wynik balanced_accuracy_score dla uśrednionego wyniku ze wszystkich foldów

def make_experiment_for_dataset(X, y, clf):
    scores = []
    for train_index, test_index in tqdm(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_pred = clf.fit(X_train, y_train).predict(X_test)
        scores.append(balanced_accuracy_score(y_test, y_pred))
    
    return np.mean(scores)


# %%
# Funkcja wykonująca eksperyment dla listy zbiorów danych. Zwraca listę wyników balanced_accuracy_score dla każdego zbioru danych

def make_experiment_for_dataset_list(datasets, clf):
    datasets_scores = []

    for key, value in tqdm(datasets.items()):
        if(type(value) is DataFrame):
            X, y = prep.data_label_split(value.to_numpy())
        else:
            X, y = prep.data_label_split(value)
        
        score = make_experiment_for_dataset(X, y, clf)
        datasets_scores.append(score)
    
    return datasets_scores
        

# %%
# Utworzenie i wytrenowanie poszczególnych algorytmów

svc = SVC(random_state=444)
gpc = GaussianProcessClassifier(random_state=444)
mlp = MLPClassifier(random_state=444)

scores_svc = make_experiment_for_dataset_list(datasets_list, svc)
scores_gpc = make_experiment_for_dataset_list(datasets_list, gpc)
scores_mlp = make_experiment_for_dataset_list(datasets_list, mlp)

# %%
# Utworzenie i wytrenowanie komitetu klasyfikatorów z głosowaniem twardym

ensemble_voting = VotingClassifier(estimators=[('SVC', svc), ('GPC', gpc), ('MLP', mlp)], voting='hard')

scores_ensemble = make_experiment_for_dataset_list(datasets_list, ensemble_voting)

# %%
# Utworzenie i wytrenowanie komitetu klasyfikatorów z głosowaniem miękkim

# ensemble = VotingClassifier(estimators=[('SVC', svc), ('GPC', gpc), ('MLP', mlp)], voting='soft')

# scores_soft_ensemble = make_experiment_for_dataset_list(datasets_list, ensemble)

# %%
# Utworzenie i wytrenowanie komitetu klasyfikatorów typu Stacked

ensemble_stacked = StackingClassifier(estimators=[('SVC', svc), ('GPC', gpc), ('MLP', mlp)], final_estimator=LogisticRegression())

scores_stacked = make_experiment_for_dataset_list(datasets_list, ensemble_stacked)

# %%

all_scores = np.transpose([scores_svc, scores_gpc, scores_mlp, scores_ensemble, scores_stacked])
print(all_scores)

# %%
# Funkcje do przeprowadzania testów Wilcoxona

def round_down(number):
    math.floor(number * 100)/100.0

    return number

def perform_wilcoxon_test(sample1, sample2):
    try:
        stat, pvalue = wilcoxon(sample1, sample2)
    except ValueError:
        stat, pvalue = 1.00
    
    return round(pvalue, 3)

# %%
# Przeprowadzanie testu Wilcoxona dla klasyfikatorów

p_svc_gpc = perform_wilcoxon_test(scores_svc, scores_gpc)
p_svc_mlp = perform_wilcoxon_test(scores_svc, scores_mlp)
p_svc_ensemble = perform_wilcoxon_test(scores_svc, scores_ensemble)
p_gpc_mlp = perform_wilcoxon_test(scores_gpc, scores_mlp)
p_gpc_ensemble = perform_wilcoxon_test(scores_gpc, scores_ensemble)
p_mlp_ensemble = perform_wilcoxon_test(scores_mlp, scores_ensemble)
p_gpc_stacking = perform_wilcoxon_test(scores_gpc, scores_stacked)
p_mlp_stacking = perform_wilcoxon_test(scores_mlp, scores_stacked)
p_ensemble_stacking = perform_wilcoxon_test(scores_gpc, scores_stacked)


# %%
# Przedstawienie wyników testu Wilcoxona

headers = ['SVC-GPC', 'SVC-MLP', 'GPC-MLP', 'SVC-Ensemble', 'GPC-Ensemble', 'MLP-Ensemble', 'SVC-Stacking', 'GPC-Stacking', 'MLP-Stacking']
wilcoxon_scores_rows = [
    [p_svc_gpc, p_svc_mlp, p_gpc_mlp, p_svc_ensemble, p_gpc_ensemble, p_mlp_ensemble, p_gpc_stacking, p_mlp_stacking, p_ensemble_stacking]
]
scores_table = tabulate(wilcoxon_scores_rows, headers=headers, tablefmt="pretty")
print(scores_table)

# %%
# Przedstawienie wyników całościowych

dataset_names_trimmed = list({name.replace('.csv', '') for name in datasets})

def round_list_elements(list):
    newlist = []
    for item in list:
        newitem = round(item, 3)
        newlist.append(newitem)
    return newlist

final_scores = [
    dataset_names_trimmed,
    round_list_elements(scores_svc), 
    round_list_elements(scores_gpc), 
    round_list_elements(scores_mlp), 
    round_list_elements(scores_ensemble), 
    round_list_elements(scores_stacked),
]
final_scores_transposed = np.transpose(final_scores)
summary_row = [
        'Average', 
        round(np.mean(scores_svc), 3), 
        round(np.mean(scores_gpc), 3), 
        round(np.mean(scores_mlp), 3), 
        round(np.mean(scores_ensemble), 3), 
        round(np.mean(scores_stacked), 3),
    ]

final_scores_transposed = np.vstack([final_scores_transposed, summary_row])
print(final_scores_transposed)

# %%

headers = ['Dataset', 'SVC', 'GPC', 'MLP', 'Voting Ensemble', 'Stacking Ensemble']
scores_rows = final_scores_transposed

scores_table = tabulate(scores_rows, headers=headers, tablefmt="pretty")
print(scores_table)

# %%
