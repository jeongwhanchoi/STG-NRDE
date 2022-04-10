import numpy as np
import torch

def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def Add_Window_Horizon_Inc(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    index_expand = 0
    batch_size = 64
    if single:
        while index < end_index:
            X.append(data[0:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            if index > 0:
                if index % batch_size == 0:
                    index_expand = index_expand + batch_size
            X.append(data[index_expand:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    # import pdb; pdb.set_trace()
    # X = np.array(X)
    Y = np.array(Y)
    X = np.array(X)
    # max_len = np.max([len(a) for a in X])
    # X = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in X])
    # Y = np.array(Y, dtype=object)
    return X, Y

if __name__ == '__main__':
    from data.load_raw_data import Load_Sydney_Demand_Data
    path = '../data/1h_data_new3.csv'
    data = Load_Sydney_Demand_Data(path)
    print(data.shape)
    X, Y = Add_Window_Horizon(data, horizon=2)
    print(X.shape, Y.shape)


