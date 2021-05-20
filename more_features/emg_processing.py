#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# ## sEMG Processing Function
# ### Add any new functions here such that they follow this behavior:
#     Args:
#         data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
#         window (int)        : the sliding window size (non-overlapping)
# 
#     Returns:
#         pandas.DataFrame    : a dataframe of the processed features with (n_samples//window, n_sensors) shape
# 
# And add it to the "processing_methods" dictionary

# In[3]:


def RMS(data, window=100):
    """
    Root Mean Square (RMS) with a non-overlapping sliding window
    Source: doi.org/10.1016/j.eswa.2012.01.102 (access pdf using mun email)

    Args:
        data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        window (int)        : the sliding window size (non-overlapping)

    Returns:
        pandas.DataFrame    : a dataframe of the RMS features with (n_samples//window, n_sensors) shape
    """
    # select the first n rows that are divisible by the window size and omit the last ones
    data2process = np.array(data)[:(data.shape[0]//window)*window]
    # reshape (reorder) the selected data to (n_samples//window, window, n_sensors)
    data2process = data2process.reshape(data2process.shape[0]//window, window, data2process.shape[1])
    # apply the processing function on each window (on axis=1) to get (n_samples//window, n_sensors)
    processed_data = np.sqrt((1/window) * np.sum(data2process**2, axis=1))
    return pd.DataFrame(data=processed_data, columns=[f'RMS_{i+1}' for i in range(data.shape[1])])


def IEMG(data, window=100):
    """
    Integrated EMG (IEMG) with a non-overlapping sliding window
    Source: doi.org/10.1016/j.eswa.2012.01.102 (access pdf using mun email)

    Args:
        data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        window (int)        : the sliding window size (non-overlapping)

    Returns:
        pandas.DataFrame    : a dataframe of the IEMG features with (n_samples//window, n_sensors) shape
    """
    # select the first n rows that are divisible by the window size and omit the last ones
    data2process = np.array(data)[:(data.shape[0]//window)*window]
    # reshape (reorder) the selected data to (n_samples//window, window, n_sensors)
    data2process = data2process.reshape(data2process.shape[0]//window, window, data2process.shape[1])
    # apply the processing function on each window (on axis=1) to get (n_samples//window, n_sensors)
    processed_data = np.sum(np.abs(data2process), axis=1)
    return pd.DataFrame(data=processed_data, columns=[f'IEMG_{i+1}' for i in range(data.shape[1])])


def MAV(data, window=100):
    """
    Mean Absolute Value (MAV) with a non-overlapping sliding window
    Source: doi.org/10.1016/j.eswa.2012.01.102 (access pdf using mun email)

    Args:
        data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        window (int)        : the sliding window size (non-overlapping)

    Returns:
        pandas.DataFrame    : a dataframe of the MAV features with (n_samples//window, n_sensors) shape
    """
    # select the first n rows that are divisible by the window size and omit the last ones
    data2process = np.array(data)[:(data.shape[0]//window)*window]
    # reshape (reorder) the selected data to (n_samples//window, window, n_sensors)
    data2process = data2process.reshape(data2process.shape[0]//window, window, data2process.shape[1])
    # apply the processing function on each window (on axis=1) to get (n_samples//window, n_sensors)
    processed_data = (1/window) * np.sum(np.abs(data2process), axis=1)
    return pd.DataFrame(data=processed_data, columns=[f'MAV_{i+1}' for i in range(data.shape[1])])


def WL(data, window=100):
    """
    Waveform Length (WL) with a non-overlapping sliding window
    Source: doi.org/10.1016/j.eswa.2012.01.102 (access pdf using mun email)

    Args:
        data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        window (int)        : the sliding window size (non-overlapping)

    Returns:
        pandas.DataFrame    : a dataframe of the WL features with (n_samples//window, n_sensors) shape
    """
    # select the first n rows that are divisible by the window size and omit the last ones
    data2process = np.array(data)[:(data.shape[0]//window)*window]
    # reshape (reorder) the selected data to (n_samples//window, window, n_sensors)
    data2process = data2process.reshape(data2process.shape[0]//window, window, data2process.shape[1])
    # apply the processing function on each window (on axis=1) to get (n_samples//window, n_sensors)
    processed_data = np.sum(np.abs(data2process[:, 1:, :]-data2process[:, :-1, :]), axis=1)
    return pd.DataFrame(data=processed_data, columns=[f'WL_{i+1}' for i in range(data.shape[1])])


def LOG(data, window=100):
    """
    Log Detector (LOG) with a non-overlapping sliding window
    Source: doi.org/10.1016/j.eswa.2012.01.102 (access pdf using mun email)

    Args:
        data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        window (int)        : the sliding window size (non-overlapping)

    Returns:
        pandas.DataFrame    : a dataframe of the LOG features with (n_samples//window, n_sensors) shape
    """
    # select the first n rows that are divisible by the window size and omit the last ones
    data2process = np.array(data)[:(data.shape[0]//window)*window]
    # reshape (reorder) the selected data to (n_samples//window, window, n_sensors)
    data2process = data2process.reshape(data2process.shape[0]//window, window, data2process.shape[1])
    # apply the processing function on each window (on axis=1) to get (n_samples//window, n_sensors)
    processed_data = np.exp((1/window) * np.sum(np.log(np.abs(data2process)), axis=1))
    return pd.DataFrame(data=processed_data, columns=[f'LOG_{i+1}' for i in range(data.shape[1])])


def SSI(data, window=100):
    """
    Simple Square Integral (SSI) with a non-overlapping sliding window
    Source: doi.org/10.1016/j.eswa.2012.01.102 (access pdf using mun email)

    Args:
        data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        window (int)        : the sliding window size (non-overlapping)

    Returns:
        pandas.DataFrame    : a dataframe of the SSI features with (n_samples//window, n_sensors) shape
    """
    # select the first n rows that are divisible by the window size and omit the last ones
    data2process = np.array(data)[:(data.shape[0]//window)*window]
    # reshape (reorder) the selected data to (n_samples//window, window, n_sensors)
    data2process = data2process.reshape(data2process.shape[0]//window, window, data2process.shape[1])
    # apply the processing function on each window (on axis=1) to get (n_samples//window, n_sensors)
    processed_data = np.sum(data2process**2, axis=1)
    return pd.DataFrame(data=processed_data, columns=[f'SSI_{i+1}' for i in range(data.shape[1])])


def VAR(data, window=100):
    """
    Variance of EMG (VAR) with a non-overlapping sliding window
    Source: doi.org/10.1016/j.eswa.2012.01.102 (access pdf using mun email)

    Args:
        data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        window (int)        : the sliding window size (non-overlapping)

    Returns:
        pandas.DataFrame    : a dataframe of the VAR features with (n_samples//window, n_sensors) shape
    """
    # select the first n rows that are divisible by the window size and omit the last ones
    data2process = np.array(data)[:(data.shape[0]//window)*window]
    # reshape (reorder) the selected data to (n_samples//window, window, n_sensors)
    data2process = data2process.reshape(data2process.shape[0]//window, window, data2process.shape[1])
    # apply the processing function on each window (on axis=1) to get (n_samples//window, n_sensors)
    processed_data = (1/(window-1)) * np.sum(data2process**2, axis=1)
    return pd.DataFrame(data=processed_data, columns=[f'VAR_{i+1}' for i in range(data.shape[1])])


def WAMP(data, window=100, thresh=0.2):
    """
    Willison Amplitude (WAMP) with a non-overlapping sliding window
    Source: doi.org/10.1016/j.eswa.2012.01.102 (access pdf using mun email)

    Args:
        data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        window (int)        : the sliding window size (non-overlapping)
        thresh (float)      : the percentage of the maximum signal amplitude to be used as a threshold

    Returns:
        pandas.DataFrame    : a dataframe of the WAMP features with (n_samples//window, n_sensors) shape
    """
    # select the first n rows that are divisible by the window size and omit the last ones
    data2process = np.array(data)[:(data.shape[0]//window)*window]
    # reshape (reorder) the selected data to (n_samples//window, window, n_sensors)
    data2process = data2process.reshape(data2process.shape[0]//window, window, data2process.shape[1])
    # apply the processing function on each window (on axis=1) to get (n_samples//window, n_sensors)
    processed_data = np.sum((np.abs(data2process[:, :-1, :]-data2process[:, 1:, :]) >= 
                                 thresh*np.max(data2process[:, :-1], axis=1, keepdims=True)).astype(int), axis=1)
    return pd.DataFrame(data=processed_data, columns=[f'WAMP_{i+1}' for i in range(data.shape[1])])


def SSC(data, window=100, thresh=0.2):
    """
    Slope Sign Change (SSC) with a non-overlapping sliding window
    Source: doi.org/10.1016/j.eswa.2012.01.102 (access pdf using mun email)

    Args:
        data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        window (int)        : the sliding window size (non-overlapping)
        thresh (float)      : the percentage of the maximum signal amplitude to be used as a threshold

    Returns:
        pandas.DataFrame    : a dataframe of the SSC features with (n_samples//window, n_sensors) shape
    """
    # select the first n rows that are divisible by the window size and omit the last ones
    data2process = np.array(data)[:(data.shape[0]//window)*window]
    # reshape (reorder) the selected data to (n_samples//window, window, n_sensors)
    data2process = data2process.reshape(data2process.shape[0]//window, window, data2process.shape[1])
    # apply the processing function on each window (on axis=1) to get (n_samples//window, n_sensors)
    func = ((data2process[:, 1:-1]-data2process[:, :-2]) * (data2process[:, 1:-1]-data2process[:, 2:]) >= 
                                        thresh*np.max(data2process[:, :-1], axis=1, keepdims=True)).astype(int)
    processed_data = np.sum(func, axis=1)
    return pd.DataFrame(data=processed_data, columns=[f'SSC_{i+1}' for i in range(data.shape[1])])


def SKW(data, window=100):
    """
    Skewness (SKW) with a non-overlapping sliding window
    Source: www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm

    Args:
        data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        window (int)        : the sliding window size (non-overlapping)

    Returns:
        pandas.DataFrame    : a dataframe of the SKW features with (n_samples//window, n_sensors) shape
    """
    # select the first n rows that are divisible by the window size and omit the last ones
    data2process = np.array(data)[:(data.shape[0]//window)*window]
    # reshape (reorder) the selected data to (n_samples//window, window, n_sensors)
    data2process = data2process.reshape(data2process.shape[0]//window, window, data2process.shape[1])
    # apply the processing function on each window (on axis=1) to get (n_samples//window, n_sensors)
    mean = np.mean(data2process, axis=1, keepdims=True)
    stddev = np.std(data2process, axis=1, keepdims=True)
    processed_data = np.sum(((data2process - mean)**3/window)/stddev**3, axis=1)
    return pd.DataFrame(data=processed_data, columns=[f'SKW_{i+1}' for i in range(data.shape[1])])


def KURT(data, window=100):
    """
    Kurtosis (KURT) with a non-overlapping sliding window
    Source: www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm

    Args:
        data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        window (int)        : the sliding window size (non-overlapping)

    Returns:
        pandas.DataFrame    : a dataframe of the KURT features with (n_samples//window, n_sensors) shape
    """
    # select the first n rows that are divisible by the window size and omit the last ones
    data2process = np.array(data)[:(data.shape[0]//window)*window]
    # reshape (reorder) the selected data to (n_samples//window, window, n_sensors)
    data2process = data2process.reshape(data2process.shape[0]//window, window, data2process.shape[1])
    # apply the processing function on each window (on axis=1) to get (n_samples//window, n_sensors)
    mean = np.mean(data2process, axis=1, keepdims=True)
    stddev = np.std(data2process, axis=1, keepdims=True)
    processed_data = np.sum(((data2process - mean)**4/window)/stddev**4, axis=1)
    return pd.DataFrame(data=processed_data, columns=[f'KURT_{i+1}' for i in range(data.shape[1])])


def AAC(data, window=100):
    """
    Average Amplitude Change (AAC) with a non-overlapping sliding window
    Source: doi.org/10.1016/j.eswa.2012.01.102 (access pdf using mun email)

    Args:
        data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        window (int)        : the sliding window size (non-overlapping)

    Returns:
        pandas.DataFrame    : a dataframe of the AAC features with (n_samples//window, n_sensors) shape
    """
    # select the first n rows that are divisible by the window size and omit the last ones
    data2process = np.array(data)[:(data.shape[0]//window)*window]
    # reshape (reorder) the selected data to (n_samples//window, window, n_sensors)
    data2process = data2process.reshape(data2process.shape[0]//window, window, data2process.shape[1])
    # apply the processing function on each window (on axis=1) to get (n_samples//window, n_sensors)
    processed_data = (1/window) * np.sum(np.abs(data2process[:, 1:]-data2process[:, :-1]), axis=1)
    return pd.DataFrame(data=processed_data, columns=[f'ACC_{i+1}' for i in range(data.shape[1])])


def DASDV(data, window=100):
    """
    Difference Absolute Standard Deviation Value (DASDV) with a non-overlapping sliding window
    Source: doi.org/10.1016/j.eswa.2012.01.102 (access pdf using mun email)

    Args:
        data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        window (int)        : the sliding window size (non-overlapping)

    Returns:
        pandas.DataFrame    : a dataframe of the DASDV features with (n_samples//window, n_sensors) shape
    """
    # select the first n rows that are divisible by the window size and omit the last ones
    data2process = np.array(data)[:(data.shape[0]//window)*window]
    # reshape (reorder) the selected data to (n_samples//window, window, n_sensors)
    data2process = data2process.reshape(data2process.shape[0]//window, window, data2process.shape[1])
    # apply the processing function on each window (on axis=1) to get (n_samples//window, n_sensors)
    processed_data = (1/(window-1)) * np.sum((data2process[:, 1:]-data2process[:, :-1])**2, axis=1)
    return pd.DataFrame(data=processed_data, columns=[f'DASDV_{i+1}' for i in range(data.shape[1])])


# In[4]:


processing_methods = {
    'RMS': RMS,
    'IEMG': IEMG,
    'MAV': MAV,
    'WL': WL,
    'LOG': LOG,
    'SSI': SSI,
    'VAR': VAR,
    'WAMP': WAMP,
    'SSC': SSC,
    'SKW': SKW,
    'KURT': KURT,
    'AAC': AAC,
    'DASDV': DASDV,
}


def process_emg(data, feats=list(processing_methods.keys()), window=100):
    """
    takes in sEMG raw data and produces the selected processing features passed to it

    Args:
        data (numpy.array)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        feats (list[str])   : list of features names to be applied from "processing_methods" [default=all]
        window (int)        : the sliding window size (non-overlapping)

    Returns:
        pandas.DataFrame    : a dataframe of the new features with (n_samples//window, n_sensors*len(feats)) shape
    """
    processed_feats = None  # initialize the processed data
    for feat in feats:      # loop over selected processing features
        feat_df = processing_methods[feat](data, window=window)  # apply the processing to the raw data
        # add the processed data to the previous processed features
        processed_feats = feat_df if processed_feats is None else processed_feats.join(feat_df)
    return processed_feats


# ## Functions to easily apply preprocessing and feature extraction techniques in one go

# In[5]:


def cutoff_processing(dataset, sampling_freq=1000, secs=3, repeat_col='repeat'):
    """
    takes in sEMG raw data and performs cut-off cleaning from the end of the repeatiotions

    Args:
        dataset (pandas.DataFrame)  : the unprocessed sEMG data stored as (n_samples, n_sensors) array
        sampling_freq (int)         : the sampling frequency of the sEMG sensors during data collection
        secs (int)                  : the number of seconds to extract from each repeatition from the end
        repeat_col (str)            : the name of the column in the dataframe containing the repeatition numbers

    Returns:
        pandas.DataFrame            : a dataframe of cleaned data with (repeations_num*secs*sampling_freq, n_sensors) shape
    """
    new_data = np.empty(shape=(0, dataset.shape[1]))  # initialize the new data 
    # a mask containing the indexes of the starts of new repeatitions in the dataframe
    ends_mask = (dataset[repeat_col].values[1:] - dataset[repeat_col].values[-1]).astype(bool)
    # the indexes of the repeatitions ends in the dataframe
    repeat_ends = (dataset.iloc[1:][ends_mask].index - dataset.index[0]).tolist() + [dataset.index[-1] - dataset.index[0]]

    for end in repeat_ends:  # loop over the indexes of repeatitions ends
        # select the cleaned data from the end of the repeatition and add it to the data
        new_data = np.vstack([new_data, dataset.iloc[end-(secs*sampling_freq):end].values])
    
    return pd.DataFrame(data=new_data, columns=dataset.columns)  # return a cleaned dataframe with the same columns


def process_columns(df, window=100):
    """
    applies a sliding window processing to the columns that don't include emg columns 

    Args:
        df (pandas.DataFrame)  : the columns to process in a (n_samples, columns_num) df
        window (int)           : the sliding window size (non-overlapping)

    Returns:
        pandas.DataFrame       : a dataframe of the processed columns with (n_samples//window, columns_num) shape
    """
    # initialize the new dataframe with zeros of the correct shape
    processed_df = pd.DataFrame(data=np.zeros(shape=(df.shape[0]//window, df.shape[1])), columns=df.columns)

    for idx, start in enumerate(range(0, df.shape[0], window)):  # loop over the data with the window size
        processed_df.iloc[idx] = df.iloc[start+window//2]  # select the middle row of the window as the new value
    return processed_df


def process_dataset(dataset, sampling_freq=1000, secs=3, feature_range=(-1, 1), feats=list(processing_methods.keys()),                          window=100, emg_cols=None, norm_func=None, repeat_col=None):
    """
    takes in sEMG raw data as a dataframeand applies multiple processing steps to it

    Args:
        dataset (pandas.DataFrame)          : the unprocessed sEMG data stored as (n_samples, n_sensors) dataframe
        sampling_freq (int)                 : the sampling frequency of the sEMG sensors during data collection
        secs (int)                          : the number of seconds to extract from each repeatition from the end
        feature_range (tuple(float, float)) : the range to normalize the sEMG data with
        feats (list[str])                   : list of features names to be applied from "processing_methods" [default=all]
        window (int)                        : the sliding window size (non-overlapping)
        emg_cols (list[str])                : list of the column names containing the sEMG data in the dataframe
        norm_func (func)                    : your custom normalization function func(pd.DataFrame) -> pd.DataFrame
        repeat_col (str)                    : the name of the column in the dataframe containing the repeatition numbers

    Returns:
        pandas.DataFrame                    : new processed dataframe after applying different methods
    """
    # indexes of the column before the first sEMG column and the one after the last sEMG column
    prior_idx = dataset.columns.get_loc(emg_cols[0])
    post_idx = dataset.columns.get_loc(emg_cols[-1]) + 1

    if secs:    # apply cut-off processing to the data
        dataset = cutoff_processing(dataset, secs=secs, sampling_freq=sampling_freq, repeat_col=repeat_col)

    # normalize the data if a function is passed (normalization can be custom -> using repeatition or random rows for test)
    if norm_func:
        dataset.iloc[:] = norm_func(dataset)
    
    new_data = process_emg(dataset[emg_cols], feats=feats, window=window)  # processed sEMG data with new features
    prior_cols = process_columns(dataset.iloc[:, :prior_idx], window=window)    # process rows from cols before sEMG columns
    post_cols = process_columns(dataset.iloc[:, post_idx:], window=window)      # process rows from cols after sEMG columns
    new_df = pd.concat([prior_cols, new_data, post_cols], axis=1)          # concatenate all the columns in correct order

    return new_df


# ## An Example of Using the Functions

# In[6]:


dummy_data = {
    'sub_idx':  [4, 4, 4, 4, 4, 4, 4, 4], 
    'repeat':   [4, 4, 4, 4, 5, 5, 5, 5],
    'signal_1': [-0.015186, -0.02292 , -0.024209, -0.020342, -0.017764, -0.019053, -0.020342, -0.019053],
    'signal_2': [-0.020342, -0.020342, -0.020342, -0.019053, -0.016475, -0.019053, -0.019053, -0.015186],
    'signal_3': [-0.021631, -0.016475, -0.013897, -0.015186, -0.025498, -0.020342, -0.015186, -0.017764],
    'signal_4': [-0.021631, -0.020342, -0.024209, -0.016475, -0.011319, -0.016475, -0.015186, -0.012608],
    'signal_5': [-0.016475, -0.017764, -0.017764, -0.017764, -0.020342, -0.019053, -0.019053, -0.019053],
    'signal_6': [-0.017764, -0.020342, -0.021631, -0.016475, -0.019053, -0.021631, -0.019053, -0.017764],
    'signal_7': [-0.013897, -0.016475, -0.019053, -0.013897, -0.015186, -0.016475, -0.011319, -0.011319],
    'signal_8': [-0.019053, -0.021631, -0.021631, -0.016475, -0.017764, -0.017764, -0.013897, -0.015186],
    'gesture_label': [7, 7, 7, 7, 7, 7, 7, 7]}

dummy_df = pd.DataFrame(dummy_data)
dummy_df


# In[7]:


emg_cols = ['signal_1', 'signal_2', 'signal_3', 'signal_4', 'signal_5', 'signal_6', 'signal_7', 'signal_8']

proc_df  = process_dataset(dummy_df, sampling_freq=3, secs=1, feature_range=(-1, 1), feats=['RMS', 'IEMG', 'MAV', 'WL'],                               window=2, emg_cols=emg_cols, norm_func=None, repeat_col='repeat')
proc_df


# In[ ]:




