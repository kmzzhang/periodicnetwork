import numpy as np
import os
import shutil
from torch.utils.data import Dataset
import copy


# t --> dt. account for periodicity
def times_to_lags(x, p=None):
    lags = x[:, 1:] - x[:, :-1]
    if p is not None:
        lags = np.c_[lags, x[:, 0] - x[:, -1] + p]
    return lags


def preprocess(X_raw, periods, use_error=False):
    N, L, F = X_raw.shape
    out_dim = 3 if use_error else 2
    X = np.zeros((N, L, out_dim))
    X[:, :, 0] = times_to_lags(X_raw[:, :, 0], periods) / periods[:, None]
    X[:, :, 1:out_dim] = X_raw[:, :, 1:out_dim]
    means = np.atleast_2d(np.nanmean(X_raw[:, :, 1], axis=1)).T
    scales = np.atleast_2d(np.nanstd(X_raw[:, :, 1], axis=1)).T
    X[:, :, 1] -= means
    X[:, :, 1] /= scales
    return X, means, scales


class PreProcessor:
    def __init__(self):
        pass

    @staticmethod
    def dtf(X_raw, periods):
        """
        Preprocess data

        Parameters
        ----------
        X_raw: np.ndarray of shape (N, L, 2), where F=0 is time, and F=1 is flux.
                Turn time into time-intervals and normalize flux.
        periods: np.ndarray of shape (N)

        Returns
        -------
        X: np.ndarray of shape (N, L, 2)
        """

        N, L, F = X_raw.shape
        X = np.zeros((N, L, 2))
        X[:, :, 0] = times_to_lags(X_raw[:, :, 0], periods) / periods[:, None]
        X[:, :, 1] = X_raw[:, :, 1]
        means = np.atleast_2d(np.nanmean(X_raw[:, :, 1], axis=1)).T
        scales = np.atleast_2d(np.nanstd(X_raw[:, :, 1], axis=1)).T
        X[:, :, 1] -= means
        X[:, :, 1] /= scales
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
