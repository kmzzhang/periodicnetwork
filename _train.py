# Author: Keming Zhang
# Date: Nov 2020
# arXiv: 2011.01243

import numpy as np
import torch
from torch import from_numpy
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm_notebook
from tqdm import tqdm as tqdm_
import wandb

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dint = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dint = torch.LongTensor


def times_to_lags(x):
    lags = x[:, 1:] - x[:, :-1]
    return lags


# preprocess into (dt, f) representation.
# for variable length training only
def preprocess(X_raw):
    N, F, L = X_raw.shape
    X = torch.zeros((N, 2, L-1))
    X[:, 0, :] = times_to_lags(X_raw[:, 0, :]) / torch.max(X_raw[:, 0, :], dim=1)[0][:,None]
    X[:, 1, :] = X_raw[:, 1, :][:,:-1]
    means = X_raw[:, 1, :].mean(dim=1).float()
    scales = X_raw[:, 1, :].std(dim=1).float()
    X[:, 1, :] -= means[:, None]
    X[:, 1, :] /= scales[:, None]
    return X, means, scales


# allow random cyclic permutation on the fly
# as data augmentation for the non-invariant networks
def permute(x):
    seq_length = x.shape[2]
    for i in range(x.shape[0]):
        start = np.random.randint(0, seq_length - 1)
        x[i] = torch.cat((x[i, :, start:], x[i, :, :start]), dim=1)
    return x


def train(model, optimizer, train_loader, val_loader, test_loader, n_epoch, eval_after=1e5, patience=10, min_lr=0.00001,
          filename='model', save=False, monitor='accuracy', print_every=-1, early_stopping_limit=1e5,
          threshold=0.1, use_tqdm=False, jupyter=False, scales_all=None, clip=-1, retrain=False, decay_type='plateau',
          log=False, perm=True):
    if jupyter:
        tqdm = tqdm_notebook
    else:
        tqdm = tqdm_
    mean_x, std_x, aux_mean, aux_std = 1, 1, 1, 1
    if scales_all is not None:
        mean_x, std_x, aux_mean, aux_std = from_numpy(scales_all).float()
        mean_x = mean_x[:-1]
        std_x = std_x[:-1]
    softmax = torch.nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()
    min_val_loss = 1e9
    max_accuracy = 0
    max_accuracy_class = 0
    early_stopping_counter = 0
    if decay_type == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=patience,
                                         cooldown=0, verbose=True, threshold=threshold)
    elif decay_type == 'exp':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.85)
    if retrain:
        val_accuracy = list(np.load(filename + '-CONVERGENCE-val.npy'))
        train_accuracy = list(np.load(filename + '-CONVERGENCE-train.npy'))
        val_losses = list(np.load(filename + '-CONVERGENCE-val-loss.npy'))
        train_losses = list(np.load(filename + '-CONVERGENCE-train-loss.npy'))
    else:
        train_accuracy = []
        val_accuracy = []
        train_losses = []
        val_losses = []
    print('------------begin training---------------')

    for epoch in tqdm(range(n_epoch), disable=not use_tqdm):
        train_loss = []; predictions = []; ground_truths = []
        if epoch < eval_after:
            model.train()
        else:
            model.eval()
        for i, d in tqdm(enumerate(train_loader), disable=not use_tqdm):
            x, aux, y = d
            if scales_all is not None:
                L = np.random.randint(16, x.shape[2])
                indexes = np.sort(np.random.choice(range(x.shape[2]), L, replace=False))
                x = x[:, :, indexes]
                x, means, scales = preprocess(x)
                x -= mean_x[None, :, None]
                x /= std_x[None, :, None]
                aux[:, 0] = (means - aux_mean[0]) / aux_std[0]
                aux[:, 1] = (scales - aux_mean[1]) / aux_std[1]
            if perm:
                x = permute(x)
            logprob = model(x.type(dtype), aux.type(dtype))
            loss = loss_fn(logprob, y.type(dint))
            optimizer.zero_grad()
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_loss.append(loss.detach().cpu())
            predictions.extend(list(np.argmax(softmax(logprob).detach().cpu(), axis=1)))
            ground_truths.extend(list(y.numpy()))
        train_loss = np.array(train_loss).mean()
        train_losses.append(train_loss)
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        accuracy = (predictions == ground_truths).mean()
        train_accuracy.append(accuracy)
        val_loss = []
        predictions = []
        ground_truths = []
        model.eval()
        np.random.seed(0)
        with torch.no_grad():
            for i, d in enumerate(val_loader):
                x, aux, y = d
                if scales_all is not None:
                    L = np.random.randint(16, x.shape[2])
                    indexes = np.sort(np.random.choice(range(x.shape[2]), L, replace=False))
                    x = x[:, :, indexes]
                    x, means, scales = preprocess(x)
                    x -= mean_x[None, :,None]
                    x /= std_x[None, :,None]
                    aux[:, 0] = (means - aux_mean[0]) / aux_std[0]
                    aux[:, 1] = (scales - aux_mean[1]) / aux_std[1]
                # allow random cyclic permutation on the fly
                # as data augmentation for the non-invariant networks
                if perm:
                    x = permute(x)
                logprob = model(x.type(dtype), aux.type(dtype))
                loss = loss_fn(logprob, y.type(dint))
                val_loss.extend([loss.detach().cpu()]*x.shape[0])

                predictions.extend(list(np.argmax(softmax(logprob).detach().cpu(), axis=1)))
                ground_truths.extend(list(y.numpy()))

        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)

        accuracy = (predictions == ground_truths).mean()
        val_accuracy.append(accuracy)
        accuracy_class_average = np.array(
            [(predictions[ground_truths == label] == ground_truths[ground_truths == label]).mean() for label in
             np.unique(ground_truths)]).mean()

        val_loss = np.array(val_loss).mean()
        val_losses.append(val_loss)

        if decay_type == 'plateau':
            lr_scheduler.step(train_loss)
        else:
            lr_scheduler.step()
        if log:
            wandb.log({"Train Loss": train_loss,
                       "Val Loss": val_loss,
                       "Train Acc": train_accuracy[-1] * 100,
                       "Val Acc": accuracy * 100})
        if print_every != -1 and epoch % print_every == 0:
            print('epoch:%d: train_loss = %.4f, val_loss = %.4f, accuracy = %.2f' % (epoch, train_loss,
                                                                                     val_loss, accuracy * 100))
        if monitor == 'val_loss':
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                early_stopping_counter = 0
                if save:
                    min_val_loss = val_loss
                    torch.save(model.state_dict(), filename+'.pth')
                    print('Saved: epoch:%d: val_loss = %.4f' % (epoch, val_loss))
        elif monitor == 'accuracy':
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_accuracy_class = accuracy_class_average
                early_stopping_counter = 0
                if save:
                    torch.save(model.state_dict(), filename+'.pth')
                    print('Saved: epoch:%d: accuracy = %.2f' % (epoch, accuracy*100))
        else:
            torch.save(model.state_dict(), filename + '.pth')
        if early_stopping_counter > early_stopping_limit > 0:
            print('Metric did not improve for %d rounds' % early_stopping_limit)
            print('Early stopping at epoch %d' % epoch)
            return train_losses, val_losses, max_accuracy, max_accuracy_class
        early_stopping_counter += 1
        np.save(filename + '-CONVERGENCE-val.npy', np.array(val_accuracy))
        np.save(filename + '-CONVERGENCE-train.npy', np.array(train_accuracy))
        np.save(filename + '-CONVERGENCE-val-loss.npy', np.array(val_losses))
        np.save(filename + '-CONVERGENCE-train-loss.npy', np.array(train_losses))
    return train_losses, val_losses, max_accuracy, max_accuracy_class
