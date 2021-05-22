#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/My Drive/Colab Notebooks/Research/GazeSemgIntegration/more_features


# In[3]:


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

# In[3]:


subject_number = 1


# # Find where the static data end from data file 'SXXX_dynamic.csv'

# In[4]:


static_end = []

for i in tqdm(range(10,10+subject_number)):
    current_dynamic = pd.read_csv('data_all_intact/S0' + str(i) + '_dynamic.csv', header=None)

    for j in range(len(current_dynamic)-1, -1, -1):
        if current_dynamic.iloc[j,0] == 0:
            static_end.append(j)
            break
    


# In[5]:


static_end


# # Analysis performance for different features

# In[6]:


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


# In[20]:


trainx.to_csv('featured_data/'+name+'_trainx.csv')
trainy.to_csv('featured_data/'+name+'_trainy.csv')
testx.to_csv('featured_data/'+name+'_testx.csv')
testy.to_csv('featured_data/'+name+'_testy.csv')


# ## RMS--Root Mean Square

# In[26]:


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


trainx.to_csv('featured_data/'+name+'_trainx.csv')
trainy.to_csv('featured_data/'+name+'_trainy.csv')
testx.to_csv('featured_data/'+name+'_testx.csv')
testy.to_csv('featured_data/'+name+'_testy.csv')


# ## IEMG--Integrated EMG

# In[12]:


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


trainx.to_csv('featured_data/'+name+'_trainx.csv')
trainy.to_csv('featured_data/'+name+'_trainy.csv')
testx.to_csv('featured_data/'+name+'_testx.csv')
testy.to_csv('featured_data/'+name+'_testy.csv')


# ## MAV--Mean Absolute Value

# In[10]:


def mav(self):
    current =0
    absolute =0
    for i in range(0, len(self)):
        current = self.iloc[i]
        absolute += np.abs(current)
    return absolute/len(self)


# In[ ]:


# index variable
'''change name, features'''

name = 'MAV'

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
            trainx.iloc[a,b] = mav(trainx_raw.iloc[a-385:a+1,b])

    trainx = trainx.iloc[385:len(trainx),:]
    trainy = trainy.iloc[385:len(trainy),:]
    
    ## test data
    testx = deepcopy(testx_raw)
    testy = deepcopy(testy_raw)

    for a in tqdm(range(385,len(testx_raw))):
        for b in range(0,12):
            testx.iloc[a,b] = mav(testx_raw.iloc[a-385:a+1,b])

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


trainx.to_csv('featured_data/'+name+'_trainx.csv')
trainy.to_csv('featured_data/'+name+'_trainy.csv')
testx.to_csv('featured_data/'+name+'_testx.csv')
testy.to_csv('featured_data/'+name+'_testy.csv')


# ## WL--Waveform Length

# In[2]:


def wl(self):
    current =0
    diff =0
    for i in range(1, len(self)):
        current = self.iloc[i]
        diff += np.abs(current - self.iloc[i-1])
    return diff


# In[ ]:


# index variable
'''change name, features'''

name = 'WL'

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
            trainx.iloc[a,b] = wl(trainx_raw.iloc[a-385:a+1,b])

    trainx = trainx.iloc[385:len(trainx),:]
    trainy = trainy.iloc[385:len(trainy),:]
    
    ## test data
    testx = deepcopy(testx_raw)
    testy = deepcopy(testy_raw)

    for a in tqdm(range(385,len(testx_raw))):
        for b in range(0,12):
            testx.iloc[a,b] = wl(testx_raw.iloc[a-385:a+1,b])

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


trainx.to_csv('featured_data/'+name+'_trainx.csv')
trainy.to_csv('featured_data/'+name+'_trainy.csv')
testx.to_csv('featured_data/'+name+'_testx.csv')
testy.to_csv('featured_data/'+name+'_testy.csv')


# ## LOG--Log Detector

# In[34]:


def log(self):
    current =0
    ln = 0
    for i in range(0, len(self)):
        current = self.iloc[i]
        ln += np.log(np.abs(current))
    return np.exp(ln/len(self))


# In[ ]:


# index variable
'''change name, features'''

name = 'LOG'

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
            trainx.iloc[a,b] = log(trainx_raw.iloc[a-385:a+1,b])

    trainx = trainx.iloc[385:len(trainx),:]
    trainy = trainy.iloc[385:len(trainy),:]
    
    ## test data
    testx = deepcopy(testx_raw)
    testy = deepcopy(testy_raw)

    for a in tqdm(range(385,len(testx_raw))):
        for b in range(0,12):
            testx.iloc[a,b] = log(testx_raw.iloc[a-385:a+1,b])

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


trainx.to_csv('featured_data/'+name+'_trainx.csv')
trainy.to_csv('featured_data/'+name+'_trainy.csv')
testx.to_csv('featured_data/'+name+'_testx.csv')
testy.to_csv('featured_data/'+name+'_testy.csv')


# ## SSI--Simple Square Integral

# In[36]:


def ssi(self):
    current =0
    square = 0
    for i in range(0, len(self)):
        current = self.iloc[i]
        square += current ** 2
    return square


# In[ ]:


# index variable
'''change name, features'''

name = 'SSI'

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
            trainx.iloc[a,b] = ssi(trainx_raw.iloc[a-385:a+1,b])

    trainx = trainx.iloc[385:len(trainx),:]
    trainy = trainy.iloc[385:len(trainy),:]
    
    ## test data
    testx = deepcopy(testx_raw)
    testy = deepcopy(testy_raw)

    for a in tqdm(range(385,len(testx_raw))):
        for b in range(0,12):
            testx.iloc[a,b] = ssi(testx_raw.iloc[a-385:a+1,b])

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


trainx.to_csv('featured_data/'+name+'_trainx.csv')
trainy.to_csv('featured_data/'+name+'_trainy.csv')
testx.to_csv('featured_data/'+name+'_testx.csv')
testy.to_csv('featured_data/'+name+'_testy.csv')


# ## SKW--Skewness

# In[66]:


def skw(self):
    current =0
    result_sum = 0
    for i in range(0, len(self)):
        current = self.iloc[i]
        result_sum += (current-self.mean())**3/len(self)/self.std()**3
    return result_sum


# In[ ]:


# index variable
'''change name, features'''

name = 'SKW'

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
            trainx.iloc[a,b] = skw(trainx_raw.iloc[a-385:a+1,b])

    trainx = trainx.iloc[385:len(trainx),:]
    trainy = trainy.iloc[385:len(trainy),:]
    
    ## test data
    testx = deepcopy(testx_raw)
    testy = deepcopy(testy_raw)

    for a in tqdm(range(385,len(testx_raw))):
        for b in range(0,12):
            testx.iloc[a,b] = skw(testx_raw.iloc[a-385:a+1,b])

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


# In[68]:


trainx.to_csv('featured_data/'+name+'_trainx.csv')
trainy.to_csv('featured_data/'+name+'_trainy.csv')
testx.to_csv('featured_data/'+name+'_testx.csv')
testy.to_csv('featured_data/'+name+'_testy.csv')


# ## KURT--Kurtosis

# In[69]:


def kurt(self):
    current =0
    result_sum = 0
    for i in range(0, len(self)):
        current = self.iloc[i]
        result_sum += (current-self.mean())**4/len(self)/self.std()**4
    return result_sum


# In[ ]:


# index variable
'''change name, features'''

name = 'KURT'

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
            trainx.iloc[a,b] = kurt(trainx_raw.iloc[a-385:a+1,b])

    trainx = trainx.iloc[385:len(trainx),:]
    trainy = trainy.iloc[385:len(trainy),:]
    
    ## test data
    testx = deepcopy(testx_raw)
    testy = deepcopy(testy_raw)

    for a in tqdm(range(385,len(testx_raw))):
        for b in range(0,12):
            testx.iloc[a,b] = kurt(testx_raw.iloc[a-385:a+1,b])

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


trainx.to_csv('featured_data/'+name+'_trainx.csv')
trainy.to_csv('featured_data/'+name+'_trainy.csv')
testx.to_csv('featured_data/'+name+'_testx.csv')
testy.to_csv('featured_data/'+name+'_testy.csv')


# ## AAC--Average Amplitude Change

# In[1]:


def aac(self):
    current =0
    diff =0
    for i in range(1, len(self)):
        current = self.iloc[i]
        diff += np.abs(current - self.iloc[i-1])
    return diff/len(self)


# In[ ]:


# index variable
'''change name, features'''

name = 'AAC'

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
            trainx.iloc[a,b] = aac(trainx_raw.iloc[a-385:a+1,b])

    trainx = trainx.iloc[385:len(trainx),:]
    trainy = trainy.iloc[385:len(trainy),:]
    
    ## test data
    testx = deepcopy(testx_raw)
    testy = deepcopy(testy_raw)

    for a in tqdm(range(385,len(testx_raw))):
        for b in range(0,12):
            testx.iloc[a,b] = aac(testx_raw.iloc[a-385:a+1,b])

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


trainx.to_csv('featured_data/'+name+'_trainx.csv')
trainy.to_csv('featured_data/'+name+'_trainy.csv')
testx.to_csv('featured_data/'+name+'_testx.csv')
testy.to_csv('featured_data/'+name+'_testy.csv')


# ## DASDV--Difference Absolute Standard Deviation Value

# In[9]:


def dasdv(self):
    current =0
    square =0
    for i in range(1, len(self)):
        current = self.iloc[i]
        square += (current - self.iloc[i-1])**2
    return square/(len(self)-1)


# In[ ]:


# index variable
'''change name, features'''

name = 'DASDV'

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
            trainx.iloc[a,b] = dasdv(trainx_raw.iloc[a-385:a+1,b])

    trainx = trainx.iloc[385:len(trainx),:]
    trainy = trainy.iloc[385:len(trainy),:]
    
    ## test data
    testx = deepcopy(testx_raw)
    testy = deepcopy(testy_raw)

    for a in tqdm(range(385,len(testx_raw))):
        for b in range(0,12):
            testx.iloc[a,b] = dasdv(testx_raw.iloc[a-385:a+1,b])

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


# In[11]:


trainx.to_csv('featured_data/'+name+'_trainx.csv')
trainy.to_csv('featured_data/'+name+'_trainy.csv')
testx.to_csv('featured_data/'+name+'_testx.csv')
testy.to_csv('featured_data/'+name+'_testy.csv')


# In[ ]:





# In[ ]:


print(acc_features,name_features)


# In[14]:


df_acc_features = pd.DataFrame(acc_features)
df_name_features = pd.DataFrame(name_features)


# In[ ]:


df_acc_features.to_csv('analysis_results/acc_features.csv')
df_name_features.to_csv('analysis_results/name_features.csv')


# In[ ]:





# In[12]:


# # function test
# dummy_data = {
#     'signal': [1,2,3,4,5,6,7,8,9,-10]}

# dummy_df = pd.DataFrame(dummy_data)
# dummy_df


# In[13]:


# dasdv(dummy_df)


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:





# In[ ]:




