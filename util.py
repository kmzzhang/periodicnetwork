import numpy as np
import os
import shutil
from torch.utils.data import Dataset

class dataset3(Dataset):
    def __init__(self, x, y, error):
        self.x = x
        self.y = y
        self.error = error

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.error[i]


def times_to_lags(x, p=None):
    lags = x[:, 1:] - x[:, :-1]
    if p is not None:
        lags = np.c_[lags, x[:, 0] - x[:, -1] + p]
    return lags


def preprocess(X_raw):
    print(X_raw.shape)
    N, L, F = X_raw.shape
    X = np.zeros((N, L-1, F))
    X[:, :, 0] = (times_to_lags(X_raw[:, :, 0]) / X_raw[:, :, 0].max(axis=1)[:,None]) + 0.001
    X[:, :, 1] = times_to_lags(X_raw[:, :, 1])
    means = np.atleast_2d(np.nanmean(X_raw[:, :, 1], axis=1)).T
    scales = np.atleast_2d(np.nanstd(X_raw[:, :, 1], axis=1)).T
    X[:, :, 1] /= scales
    X[:, :, 2] = X[:, :, 1] / X[:, :, 0]
    return X, means, scales


class PreProcessor():
    def __init__(self):
        pass

    @staticmethod
    def dtdfg(X_raw):
        N, L, F = X_raw.shape
        X = np.zeros((N, L - 1, 3))
        X[:, :, 0] = (times_to_lags(X_raw[:, :, 0]) / X_raw[:, :, 0].max(axis=1)[:, None]) + 0.001
        X[:, :, 1] = times_to_lags(X_raw[:, :, 1])
        means = np.atleast_2d(np.nanmean(X_raw[:, :, 1], axis=1)).T
        scales = np.atleast_2d(np.nanstd(X_raw[:, :, 1], axis=1)).T
        X[:, :, 1] /= scales
        X[:, :, 2] = X[:, :, 1] / X[:, :, 0]
        return X, means, scales

    @staticmethod
    def dtfg(X_raw):
        N, L, F = X_raw.shape
        X = np.zeros((N, L - 1, 3))
        X[:, :, 0] = (times_to_lags(X_raw[:, :, 0]) / X_raw[:, :, 0].max(axis=1)[:, None]) + 0.001
        X[:, :, 1] = X_raw[:, :, 1][:, :-1]
        df = times_to_lags(X_raw[:, :, 1])
        means = np.atleast_2d(np.nanmean(X_raw[:, :, 1], axis=1)).T
        scales = np.atleast_2d(np.nanstd(X_raw[:, :, 1], axis=1)).T
        X[:, :, 1] -= means
        X[:, :, 1] /= scales
        X[:, :, 2] = df / X[:, :, 0]
        return X, means, scales

    @staticmethod
    def dtdf(X_raw):
        N, L, F = X_raw.shape
        X = np.zeros((N, L - 1, 2))
        X[:, :, 0] = (times_to_lags(X_raw[:, :, 0]) / X_raw[:, :, 0].max(axis=1)[:, None]) + 0.001
        X[:, :, 1] = times_to_lags(X_raw[:, :, 1])
        means = np.atleast_2d(np.nanmean(X_raw[:, :, 1], axis=1)).T
        scales = np.atleast_2d(np.nanstd(X_raw[:, :, 1], axis=1)).T
        X[:, :, 1] /= scales
        return X, means, scales

    @staticmethod
    def dtf(X_raw, p):
        N, L, F = X_raw.shape
        X = np.zeros((N, L, 2))
        X[:, :, 0] = times_to_lags(X_raw[:, :, 0], p) / p[:, None]
        X[:, :, 1] = X_raw[:, :, 1]
        means = np.atleast_2d(np.nanmean(X_raw[:, :, 1], axis=1)).T
        scales = np.atleast_2d(np.nanstd(X_raw[:, :, 1], axis=1)).T
        X[:, :, 1] -= means
        X[:, :, 1] /= scales
        return X, means, scales

    @staticmethod
    def f(X_raw):
        print(X_raw.shape)
        N, L, F = X_raw.shape
        X = np.zeros((N, L, 1))
        X[:, :, 0] = X_raw[:, :, 1]
        means = np.atleast_2d(np.nanmean(X_raw[:, :, 1], axis=1)).T
        scales = np.atleast_2d(np.nanstd(X_raw[:, :, 1], axis=1)).T
        X[:, :, 0] -= means
        X[:, :, 0] /= scales
        return X, means, scales

    @staticmethod
    def df(X_raw):
        print(X_raw.shape)
        N, L, F = X_raw.shape
        X = np.zeros((N, L - 1, 1))
        X[:, :, 0] = times_to_lags(X_raw[:, :, 1])
        means = np.atleast_2d(np.nanmean(X_raw[:, :, 1], axis=1)).T
        scales = np.atleast_2d(np.nanstd(X_raw[:, :, 1], axis=1)).T
        X[:, :, 0] /= scales
        return X, means, scales

    @staticmethod
    def dfg(X_raw):
        N, L, F = X_raw.shape
        X = np.zeros((N, L - 1, 2))
        dt = (times_to_lags(X_raw[:, :, 0]) / X_raw[:, :, 0].max(axis=1)[:, None]) + 0.001
        X[:, :, 0] = times_to_lags(X_raw[:, :, 1])
        means = np.atleast_2d(np.nanmean(X_raw[:, :, 1], axis=1)).T
        scales = np.atleast_2d(np.nanstd(X_raw[:, :, 1], axis=1)).T
        X[:, :, 0] /= scales
        X[:, :, 1] = X[:, :, 0] / dt
        return X, means, scales

    @staticmethod
    def g(X_raw):
        N, L, F = X_raw.shape
        X = np.zeros((N, L - 1, 1))
        dt = (times_to_lags(X_raw[:, :, 0]) / X_raw[:, :, 0].max(axis=1)[:, None]) + 0.001
        df = times_to_lags(X_raw[:, :, 1])
        means = np.atleast_2d(np.nanmean(X_raw[:, :, 1], axis=1)).T
        scales = np.atleast_2d(np.nanstd(X_raw[:, :, 1], axis=1)).T
        df /= scales
        X[:, :, 0] = df / dt
        return X, means, scales


def cat_list(*lists):
    lists = lists[0]
    ans = []
    for l in lists:
        ans += list(l)
    return ans


def train_test_split(y, train_size=0.33, random_state=0):
    if random_state != -1:
        np.random.seed(random_state)
    labels = np.copy(y)
    np.random.shuffle(labels)
    y_unique = np.unique(y)
    indexes = np.arange(len(y))
    x_split = [np.array(indexes[y == label]) for label in y_unique]
    for i in range(len(y_unique)):
        if random_state != -1:
            np.random.shuffle(x_split[i])
    trains = cat_list(x_split[label][:max(int(train_size * len(x_split[label]) + 0.5), 1)] for label in
                      range(len(y_unique)))
    tests = cat_list(x_split[label][max(int(train_size * len(x_split[label]) + 0.5), 1):] for label in
                     range(len(y_unique)))
    return trains, tests


def create_device(path, ngpu=1, njob=1, ID=None):
    try:
        shutil.rmtree(path)
    except:
        pass
    os.mkdir(path)
    if ngpu == 1:
        if ID is None:
            ID = 0
        with open(path + '/%d_%d' % (ID, 0), 'a'):
            os.utime(path + '/%d_%d' % (ID, 0), None)
        return None
    for i in range(ngpu):
        for j in range(njob):
            with open(path+'/%d_%d'%(i,j), 'a'):
                os.utime(path+'/%d_%d'%(i,j), None)


def get_device(path):
    device = os.listdir(path)[0]
    os.remove(path+'/'+device)
    return device


def return_device(path, device):
    with open(path + '/' + device, 'a'):
        os.utime(path + '/' + device, None)
