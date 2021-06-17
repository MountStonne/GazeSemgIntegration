# -*- coding: utf-8 -*-
"""feature_extract_step50ms_30subjects.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LnHy1M3IeBv-_fa5hJhGPAEfRbNZQfZ8
"""

# # Commented out IPython magic to ensure Python compatibility.
# from google.colab import drive
# drive.mount('/content/drive')
# # %cd /content/drive/My Drive/Colab Notebooks/Research/GazeSemgIntegration/more_features

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as ltb
from sklearn.neural_network import MLPClassifier
from copy import deepcopy
import warnings
import math


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

warnings.filterwarnings('ignore')

subject_number = 11

# static_end = []
# for i in tqdm(range(10,10+subject_number)):
#     current_dynamic = pd.read_csv('data_all_intact/S0' + str(i) + '_dynamic.csv', header=None)

#     for j in range(len(current_dynamic)-1, -1, -1):
#         if current_dynamic.iloc[j,0] == 0:
#             static_end.append(j)
#             break


# static_end_df = pd.DataFrame(static_end)
# static_end_df.to_csv('featured_data/all_intact_featured_step50ms/static_end.csv', index=False)

static_end_df = pd.read_csv('featured_data/all_intact_featured_step50ms/static_end.csv')
static_end = static_end_df.to_numpy().flatten()


"""# WL"""

def wl(self):
    current =0
    diff =0
    for i in range(1, len(self)):
        current = self.iloc[i]
        diff += np.abs(current - self.iloc[i-1])
    return diff

# index variable
'''change name, features'''
name = 'WL'

# start
for i in tqdm(range(30,30+subject_number)):
    
    # read csv files
    current_emg = pd.read_csv('data_all_intact/S0' + str(i) + '_emg.csv', header=None)
    current_grasp = pd.read_csv('data_all_intact/S0' + str(i) + '_grasp.csv', header=None)
    current_grasprepetition = pd.read_csv('data_all_intact/S0' + str(i) + '_grasprepetition.csv', header=None)
    
    current_static_emg = current_emg.iloc[0:static_end[i-10]]
    current_static_grasp = current_grasp.iloc[0:static_end[i-10]]
    current_static_grasprepetition = current_grasprepetition.iloc[0:static_end[i-10]]
    
    # find raw train/test data
    train_index = []
    test_index = []
    for j in tqdm(range(len(current_static_grasprepetition))):
        if current_static_grasprepetition.iloc[j,0] in [1,2,3,4,5,6,7,8,9]:
            train_index.append(j)
        if current_static_grasprepetition.iloc[j,0] in [10,11,12]:
            test_index.append(j)
    
    trainx_raw = current_static_emg.iloc[train_index].reset_index(drop=True)
    trainy_raw = current_static_grasp.iloc[train_index].reset_index(drop=True)

    testx_raw = current_static_emg.iloc[test_index].reset_index(drop=True)
    testy_raw = current_static_grasp.iloc[test_index].reset_index(drop=True)
    
    # add overlapping window to train/test data
    ## 385 means the overlapping window is 200ms
    ## train data

    trainx = []
    trainy = []
    
    for a in tqdm(range(385,len(trainx_raw), 96)):
        current_row = []
        for b in range(0,len(trainx_raw.columns)):
            current_row.append(wl(trainx_raw.iloc[a-385:a+1, b]))
        trainx.append(current_row)
        trainy.append(trainy_raw.iloc[a, 0])

    ## test data
    testx = []
    testy = []

    for a in tqdm(range(385,len(testx_raw),96)):
        current_row = []
        for b in range(0,len(testx_raw.columns)):
            current_row.append(wl(testx_raw.iloc[a-385:a+1, b]))
        testx.append(current_row)
        testy.append(testy_raw.iloc[a, 0])
    
    trainx_df = pd.DataFrame(trainx)
    trainy_df = pd.DataFrame(trainy)
    testx_df = pd.DataFrame(testx)
    testy_df = pd.DataFrame(testy)

    trainx_df.to_csv('featured_data/all_intact_featured_step50ms/S0' + str(i) + '/' + name + '_trainx.csv')
    trainy_df.to_csv('featured_data/all_intact_featured_step50ms/S0' + str(i) + '/' + name + '_trainy.csv')
    testx_df.to_csv('featured_data/all_intact_featured_step50ms/S0' + str(i) + '/' + name + '_testx.csv')
    testy_df.to_csv('featured_data/all_intact_featured_step50ms/S0' + str(i) + '/' + name + '_testy.csv')