#!/usr/bin/env python
# coding: utf-8

# In[7]:


# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/My Drive/Colab Notebooks/Research/GazeSemgIntegration/more_features


# In[1]:


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


# # Index variables

# In[2]:


subject_number = 1


# # Find where the static data end from data file 'SXXX_dynamic.csv'

# In[3]:


static_end = []

for i in tqdm(range(10,10+subject_number)):
    current_dynamic = pd.read_csv('data_all_intact/S0' + str(i) + '_dynamic.csv', header=None)

    for j in range(len(current_dynamic)-1, -1, -1):
        if current_dynamic.iloc[j,0] == 0:
            static_end.append(j)
            break
    


# In[4]:


static_end


# # Analysis performance for different features

# In[5]:


# save accuracy in array 'acc_features'
# save features name in array 'name_features'
acc_features = []
name_features = []


# ## STD--Standard deviation

# In[14]:


# index variable
'''change name, features'''
name = 'STD'

# start
for i in tqdm(range(10,10+subject_number)):
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
    for j in range(len(current_static_grasprepetition)):
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
    trainx = deepcopy(trainx_raw)
    trainy = deepcopy(trainy_raw)
    
    for a in tqdm(range(385,len(trainx_raw))):
        for b in range(0,12):
            trainx.iloc[a,b] = trainx_raw.iloc[a-385:a+1,b].std()

    trainx = trainx.iloc[385:len(trainx),:]
    trainy = trainy.iloc[385:len(trainy),:]
    
    ## test data
    testx = deepcopy(testx_raw)
    testy = deepcopy(testy_raw)

    for a in tqdm(range(385,len(testx_raw))):
        for b in range(0,12):
            testx.iloc[a,b] = testx_raw.iloc[a-385:a+1,b].std()

    testx = testx.iloc[385:len(testx),:]
    testy = testy.iloc[385:len(testy),:]
    
    # train/test model
    lightGBM_model = ltb.LGBMClassifier()
    lightGBM_model.fit(trainx,trainy)
    expected1 = testy.values.flatten()
    predicted1 = lightGBM_model.predict(testx)
    counter = 0
    for i in range(0, len(expected1)):
        if expected1[i] == predicted1[i]:
            counter += 1

    accuracy1 = counter/len(expected1)

acc_features.append(accuracy1)
name_features.append(name)


# In[17]:


# trainx.to_csv('featured_data/'+name+'/trainx.csv')
# trainy.to_csv('featured_data/'+name+'/trainy.csv')
# testx.to_csv('featured_data/'+name+'/testx.csv')
# testy.to_csv('featured_data/'+name+'/testy.csv')


# ## RMS--Root Mean Square

# In[18]:


def rms(self):
    current =0
    square =0
    for i in range(0, len(self)):
        current = self.iloc[i]
        square += current ** 2
    mean = square/len(self)
    root = math.sqrt(mean)
    return root


# In[ ]:


# index variable
'''change name, features'''

name = 'RMS'

# start
for i in tqdm(range(10,10+subject_number)):
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
    for j in range(len(current_static_grasprepetition)):
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
    trainx = deepcopy(trainx_raw)
    trainy = deepcopy(trainy_raw)
    
    for a in tqdm(range(385,len(trainx_raw))):
        for b in range(0,12):
            trainx.iloc[a,b] = rms(trainx_raw.iloc[a-385:a+1,b])

    trainx = trainx.iloc[385:len(trainx),:]
    trainy = trainy.iloc[385:len(trainy),:]
    
    ## test data
    testx = deepcopy(testx_raw)
    testy = deepcopy(testy_raw)

    for a in tqdm(range(385,len(testx_raw))):
        for b in range(0,12):
            testx.iloc[a,b] = rms(testx_raw.iloc[a-385:a+1,b])

    testx = testx.iloc[385:len(testx),:]
    testy = testy.iloc[385:len(testy),:]
    
    # train/test model
    lightGBM_model = ltb.LGBMClassifier()
    lightGBM_model.fit(trainx,trainy)
    expected1 = testy.values.flatten()
    predicted1 = lightGBM_model.predict(testx)
    counter = 0
    for i in range(0, len(expected1)):
        if expected1[i] == predicted1[i]:
            counter += 1

    accuracy1 = counter/len(expected1)

acc_features.append(accuracy1)
name_features.append(name)


# In[ ]:


# trainx.to_csv('featured_data/'+name+'/trainx.csv')
# trainy.to_csv('featured_data/'+name+'/trainy.csv')
# testx.to_csv('featured_data/'+name+'/testx.csv')
# testy.to_csv('featured_data/'+name+'/testy.csv')


# ## IEMG--Integrated EMG

# In[6]:


def iemg(self):
    current =0
    absolute =0
    for i in range(0, len(self)):
        current = self.iloc[i]
        absolute += np.abs(current)
    return absolute


# In[ ]:


# index variable
'''change name, features'''

name = 'IEMG'

# start
for i in tqdm(range(10,10+subject_number)):
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
    for j in range(len(current_static_grasprepetition)):
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
    trainx = deepcopy(trainx_raw)
    trainy = deepcopy(trainy_raw)
    
    for a in tqdm(range(385,len(trainx_raw))):
        for b in range(0,12):
            trainx.iloc[a,b] = iemg(trainx_raw.iloc[a-385:a+1,b])

    trainx = trainx.iloc[385:len(trainx),:]
    trainy = trainy.iloc[385:len(trainy),:]
    
    ## test data
    testx = deepcopy(testx_raw)
    testy = deepcopy(testy_raw)

    for a in tqdm(range(385,len(testx_raw))):
        for b in range(0,12):
            testx.iloc[a,b] = iemg(testx_raw.iloc[a-385:a+1,b])

    testx = testx.iloc[385:len(testx),:]
    testy = testy.iloc[385:len(testy),:]
    
    # train/test model
    lightGBM_model = ltb.LGBMClassifier()
    lightGBM_model.fit(trainx,trainy)
    expected1 = testy.values.flatten()
    predicted1 = lightGBM_model.predict(testx)
    counter = 0
    for i in range(0, len(expected1)):
        if expected1[i] == predicted1[i]:
            counter += 1

    accuracy1 = counter/len(expected1)

acc_features.append(accuracy1)
name_features.append(name)


# In[ ]:


# trainx.to_csv('featured_data/'+name+'/trainx.csv')
# trainy.to_csv('featured_data/'+name+'/trainy.csv')
# testx.to_csv('featured_data/'+name+'/testx.csv')
# testy.to_csv('featured_data/'+name+'/testy.csv')


# In[ ]:





# In[ ]:





# In[ ]:


print(acc_features,name_features)


# In[ ]:


acc_features
name_features


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


# final test commit from Colab


# In[ ]:




