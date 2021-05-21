def RMS(data, window=100):
    processed = np.zeros(shape=(int(np.ceil(data.shape[0]/window)), data.shape[1]))
    for idx, start in enumerate(range(0, data.shape[0], window)):
        window_data = data[start:start+window]
        processed[idx] = np.sqrt((1/window_data.shape[0]) * np.sum(window_data**2, axis=0))
    
    return pd.DataFrame(data=processed, columns=[f'RMS_{i+1}' for i in range(data.shape[1])])


def IEMG(data, window=100):
    processed = np.zeros(shape=(int(np.ceil(data.shape[0]/window)), data.shape[1]))
    for idx, start in enumerate(range(0, data.shape[0], window)):
        window_data = data[start:start+window]
        processed[idx] = np.sum(np.abs(window_data), axis=0)
    
    return pd.DataFrame(data=processed, columns=[f'IEMG_{i+1}' for i in range(data.shape[1])])


def MAV(data, window=100):
    processed = np.zeros(shape=(int(np.ceil(data.shape[0]/window)), data.shape[1]))
    for idx, start in enumerate(range(0, data.shape[0], window)):
        window_data = data[start:start+window]
        processed[idx] = (1/window_data.shape[0]) * np.sum(np.abs(window_data), axis=0)
    
    return pd.DataFrame(data=processed, columns=[f'MAV_{i+1}' for i in range(data.shape[1])])


def WL(data, window=100):
    processed = np.zeros(shape=(int(np.ceil(data.shape[0]/window)), data.shape[1]))
    for idx, start in enumerate(range(0, data.shape[0], window)):
        window_data = data[start:start+window]
        processed[idx] = np.sum(np.abs(window_data[1:]-window_data[:-1]), axis=0)
    
    return pd.DataFrame(data=processed, columns=[f'WL_{i+1}' for i in range(data.shape[1])])


def LOG(data, window=100):
    processed = np.zeros(shape=(int(np.ceil(data.shape[0]/window)), data.shape[1]))
    for idx, start in enumerate(range(0, data.shape[0], window)):
        window_data = data[start:start+window]
        processed[idx] = np.exp((1/window_data.shape[0]) * np.sum(np.log(np.abs(window_data)), axis=0))
    
    return pd.DataFrame(data=processed, columns=[f'LOG_{i+1}' for i in range(data.shape[1])])


def SSI(data, window=100):
    processed = np.zeros(shape=(int(np.ceil(data.shape[0]/window)), data.shape[1]))
    for idx, start in enumerate(range(0, data.shape[0], window)):
        window_data = data[start:start+window]
        processed[idx] = np.sum(window_data**2, axis=0)
    
    return pd.DataFrame(data=processed, columns=[f'SSI_{i+1}' for i in range(data.shape[1])])


def VAR(data, window=100):
    processed = np.zeros(shape=(int(np.ceil(data.shape[0]/window)), data.shape[1]))
    for idx, start in enumerate(range(0, data.shape[0], window)):
        window_data = data[start:start+window]
        processed[idx] = (1/(window_data.shape[0]-1)) * np.sum(window_data**2, axis=0)
    
    return pd.DataFrame(data=processed, columns=[f'VAR_{i+1}' for i in range(data.shape[1])])


def WA(data, thresh=0.2, window=100):
    processed = np.zeros(shape=(int(np.ceil(data.shape[0]/window)), data.shape[1]))
    for idx, start in enumerate(range(0, data.shape[0], window)):
        window_data = data[start:start+window]
        processed[idx] = np.sum((np.abs(window_data[:-1]-window_data[1:]) >= 
                                 thresh*np.max(window_data[:-1], axis=0)).astype(int), axis=0)
    
    return pd.DataFrame(data=processed, columns=[f'WA_{i+1}' for i in range(data.shape[1])])


def SSC(data, thresh=0.2, window=100):
    processed = np.zeros(shape=(int(np.ceil(data.shape[0]/window)), data.shape[1]))
    for idx, start in enumerate(range(0, data.shape[0], window)):
        window_data = data[start:start+window]
        func = ((window_data[1:-1]-window_data[:-2]) * 
                (window_data[1:-1]-window_data[2:]) >= thresh*np.max(window_data[:-1], axis=0)).astype(int)
        processed[idx] = np.sum(func, axis=0)
    
    return pd.DataFrame(data=processed, columns=[f'SSC_{i+1}' for i in range(data.shape[1])])


def SKW(data, window=100):
    processed = np.zeros(shape=(int(np.ceil(data.shape[0]/window)), data.shape[1]))
    for idx, start in enumerate(range(0, data.shape[0], window)):
        window_data = data[start:start+window]
        mean = np.mean(window_data, axis=0)
        stddev = np.std(window_data, axis=0)
        processed[idx] = np.sum(((window_data - mean)**3/window_data.shape[0])/stddev**3, axis=0)
    
    return pd.DataFrame(data=processed, columns=[f'SKW_{i+1}' for i in range(data.shape[1])])


def KURT(data, window=100):
    processed = np.zeros(shape=(int(np.ceil(data.shape[0]/window)), data.shape[1]))
    for idx, start in enumerate(range(0, data.shape[0], window)):
        window_data = data[start:start+window]
        mean = np.mean(window_data, axis=0)
        stddev = np.std(window_data, axis=0)
        processed[idx] = np.sum(((window_data - mean)**4/window_data.shape[0])/stddev**4, axis=0)
    
    return pd.DataFrame(data=processed, columns=[f'KURT_{i+1}' for i in range(data.shape[1])])


def AAC(data, window=100):
    processed = np.zeros(shape=(int(np.ceil(data.shape[0]/window)), data.shape[1]))
    for idx, start in enumerate(range(0, data.shape[0], window)):
        window_data = data[start:start+window]
        processed[idx] = (1/window_data.shape[0]) * np.sum(np.abs(window_data[1:]-window_data[:-1]), axis=0)
    
    return pd.DataFrame(data=processed, columns=[f'ACC_{i+1}' for i in range(data.shape[1])])


def DASDV(data, window=100):
    processed = np.zeros(shape=(int(np.ceil(data.shape[0]/window)), data.shape[1]))
    for idx, start in enumerate(range(0, data.shape[0], window)):
        window_data = data[start:start+window]
        processed[idx] = (1/(window_data.shape[0]-1)) * np.sum((window_data[1:]-window_data[:-1])**2, axis=0)
    
    return pd.DataFrame(data=processed, columns=[f'DASDV_{i+1}' for i in range(data.shape[1])])

