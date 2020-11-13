# Author: Keming Zhang
# Date: Nov 2020
# arXiv: 2011.01243

import os
import sys
import joblib
import argparse
import torch.multiprocessing as mp
sys.path.append('./')


parser = argparse.ArgumentParser(description='')
parser.add_argument('--L', type=int, default=128,
                    help='training sequence length')
parser.add_argument('--filename', type=str, default='test.pkl',
                    help='dataset filename. file is expected in ./data/')
parser.add_argument('--frac-train', type=float, default=0.8,
                    help='training sequence length')
parser.add_argument('--frac-valid', type=float, default=0.25,
                    help='training sequence length')
parser.add_argument('--train-batch', type=int, default=32,
                    help='training sequence length')
parser.add_argument('--varlen_train', action='store_true', default=False,
                    help='enable variable length training')
parser.add_argument('--use-error', action='store_true', default=False,
                    help='use error as additional dimension')
parser.add_argument('--use-meta', action='store_true', default=False,
                    help='use meta as auxiliary network input')
parser.add_argument('--input', type=str, default='dtf',
                    help='obsolete. input representation of data. use either dtf or dtfe, which include errors')
parser.add_argument('--n_test', type=int, default=1,
                    help='number of different sequence length to test')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout rate')
parser.add_argument('--dropout-classifier', type=float, default=0,
                    help='dropout rate')
parser.add_argument('--permute', action='store_true', default=False,
                    help='data augmentation')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clipping')
parser.add_argument('--path', type=str, default='temp',
                    help='folder name to save experiement results')
parser.add_argument('--max_epoch', type=int, default=50,
                    help='maximum number of training epochs')
parser.add_argument('--min_maxpool', type=int, default=2,
                    help='minimum length required for maxpool operation.')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of gpu devices to use. neg number refer to particular single device number')
parser.add_argument('--njob', type=int, default=1,
                    help='maximum number of networks to train on each gpu')
parser.add_argument('--K', type=int, default=8,
                    help='number of data partition to use')
parser.add_argument('--pseed', type=int, default=0,
                    help='random seed for data partition (only when K = 1)')
parser.add_argument('--network', type=str, default='iresnet',
                    help='name of the neural network to train')
parser.add_argument('--kernel', type=int, default=2,
                    help='kernel size')
parser.add_argument('--depth', type=int, default=7,
                    help='network depth')
parser.add_argument('--n_layer', type=int, default=2,
                    help='(iresnet/resnet only) number of convolution per residual block')
parser.add_argument('--hidden', type=int, default=128,
                    help='hidden dimension')
parser.add_argument('--hidden-classifier', type=int, default=32,
                    help='hidden dimension for final layer')
parser.add_argument('--max_hidden', type=int, default=128,
                    help='(iresnet/resnet only) maximum hidden dimension')
parser.add_argument('--two_phase', action='store_true', default=False,
                    help='')
parser.add_argument('--print_every', type=int, default=-1,
                    help='')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for network seed and random partition')
parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                    help='')
parser.add_argument('--min_sample', type=int, default=0,
                    help='minimum number of pre-segmented light curve per class')
parser.add_argument('--max_sample', type=int, default=100000,
                    help='maximum number of pre-segmented light curve per class during testing')
parser.add_argument('--test', action='store_true', default=False,
                    help='test pre-trained model')
parser.add_argument('--retrain', action='store_true', default=False,
                    help='continue training from checkpoint')
parser.add_argument('--no-log', action='store_true', default=False,
                    help='continue training from checkpoint')
parser.add_argument('--note', type=str, default='',
                    help='')
parser.add_argument('--project-name', type=str, default='',
                    help='for weights and biases tracking')
parser.add_argument('--decay-type', type=str, default='plateau',
                    help='')
parser.add_argument('--patience', type=int, default=5,
                    help='patience for learning decay')
parser.add_argument('--early_stopping', type=int, default=0,
                    help='terminate training if loss does not improve by 10% after waiting this number of epochs')
args = parser.parse_args()


def get_device(path):
    device = os.listdir(path)[0]
    os.remove(path+'/'+device)
    return device


if args.network == 'resnet' or args.network == 'iresnet':
    save_name = '{}-{}-K{}-D{}-NL{}-H{}-MH{}-L{}-V{}-{}-LR{}-CLIP{}-DROP{}-TP{}'.format(args.filename[:-4],
                                                                              args.network,
                                                                              args.kernel,
                                                                              args.depth,
                                                                              args.n_layer,
                                                                              args.hidden,
                                                                              args.max_hidden,
                                                                              args.L,
                                                                              int(args.varlen_train),
                                                                              args.input,
                                                                              args.lr,
                                                                              args.hidden_classifier,
                                                                              max(args.dropout, args.dropout_classifier),
                                                                              int(args.two_phase))
else:
    save_name = '{}-{}-K{}-D{}-H{}-L{}-V{}-{}-LR{}-CLIP{}-DROP{}-TP{}'.format(args.filename[:-4],
                                                                              args.network,
                                                                              args.kernel,
                                                                              args.depth,
                                                                              args.hidden,
                                                                              args.L,
                                                                              int(args.varlen_train),
                                                                              args.input,
                                                                              args.lr,
                                                                              args.clip,
                                                                              args.dropout,
                                                                              int(args.two_phase))
from torch.multiprocessing import current_process
if current_process().name != 'MainProcess':
    if args.njob > 1 or args.ngpu > 1:
        path = 'device'+save_name+args.note
        device = get_device(path)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device[0])
else:
    print('save filename:')
    print(save_name)
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.iresnet import Classifier as iresnet
from model.itcn import Classifier as itcn
from model.rnn import Classifier as rnn
from data import MyDataset as MyDataset

from light_curve import LightCurve
from util import *
from _train import train

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dint = torch.cuda.LongTensor
    map_loc = 'cuda:0'
else:
    assert args.ngpu == 1
    dtype = torch.FloatTensor
    dint = torch.LongTensor
    map_loc = 'cpu'

if args.cudnn_deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if 'asassn' in args.filename:
    args.max_sample = 20000
if args.n_test == 1:
    lengths = [args.L]
else:
    lengths = np.linspace(16, args.L * 2, args.n_test).astype(np.int)
    if args.L not in lengths:
        lengths = np.sort(np.append(lengths, args.L))
data = joblib.load('data/{}'.format(args.filename))
# sanity check on dataset
for lc in data:
    positive = lc.errors > 0
    positive *= lc.errors < 99
    lc.times = lc.times[positive]
    lc.measurements = lc.measurements[positive]
    lc.errors = lc.errors[positive]

if 'macho' in args.filename:
    for lc in data:
        if 'LPV' in lc.label:
            lc.label = "LPV"

# Generate a list all labels for train/test split
unique_label, count = np.unique([lc.label for lc in data], return_counts=True)
use_label = unique_label[count >= args.min_sample]

n_classes = len(use_label)
new_data = []
for cls in use_label:
    class_data = [lc for lc in data if lc.label == cls]
    new_data.extend(class_data[:min(len(class_data), args.max_sample)])
data = new_data

all_label_string = [lc.label for lc in data]
unique_label, count = np.unique(all_label_string, return_counts=True)
print('------------before segmenting into L={}------------'.format(args.L))
print(unique_label)
print(count)
convert_label = dict(zip(use_label, np.arange(len(use_label))))
all_labels = np.array([convert_label[lc.label] for lc in data])

# if args.input in ['dtdfg', 'dtfg', 'dtfe']:
#     n_inputs = 3
# elif args.input in ['df', 'f', 'g']:
#     n_inputs = 1
# else:
#     n_inputs = 2
n_inputs = 3 if args.use_error else 2


def get_network(n_classes):

    if args.network in ['itcn', 'iresnet']:
        padding = 'cyclic'
    else:
        padding = 'zero'

    if args.network in ['itcn', 'tcn']:
        clf = itcn(
            num_inputs=n_inputs,
            num_class=n_classes,
            depth=args.depth,
            hidden_conv=args.hidden,
            hidden_classifier=args.hidden_classifier,
            dropout=args.dropout,
            kernel_size=args.kernel,
            dropout_classifier=args.dropout_classifier,
            aux=3,
            padding=padding
        ).type(dtype)

    elif args.network in ['iresnet', 'resnet']:
        clf = iresnet(
            n_inputs,
            n_classes,
            depth=args.depth,
            nlayer=args.n_layer,
            kernel_size=args.kernel,
            hidden_conv=args.hidden,
            max_hidden=args.max_hidden,
            padding=padding,
            min_length=args.min_maxpool,
            aux=3,
            dropout_classifier=args.dropout_classifier,
            hidden=args.hidden_classifier
        ).type(dtype)

    elif args.network in ['gru', 'lstm']:
        clf = rnn(
            num_inputs=n_inputs,
            hidden_rnn=args.hidden,
            num_layers=args.depth,
            num_class=n_classes,
            hidden=args.hidden_classifier,
            rnn=args.network.upper(),
            dropout=args.dropout,
            aux=3
        ).type(dtype)

    return clf


def train_helper(param):
    global map_loc
    train_index, test_index, name = param
    split = [chunk for i in train_index for chunk in data[i].split(args.L, args.L) if data[i].label is not None]
    for lc in split:
        lc.period_fold()
    unique_label, count = np.unique([lc.label for lc in split], return_counts=True)
    print('------------after segmenting into L={}------------'.format(args.L))
    print(unique_label)
    print(count)

    # shape: (N, L, 3)
    X_list = [np.c_[chunk.times, chunk.measurements, chunk.errors] for chunk in split]
    periods = np.array([lc.p for lc in split])
    label = np.array([convert_label[chunk.label] for chunk in split])

    # x, means, scales = getattr(PreProcessor, args.input)(np.array(X_list), periods)
    x, means, scales = preprocess(np.array(X_list), periods, use_error=args.use_error)
    print('shape of the training dataset array:', x.shape)
    mean_x = x.reshape(-1, n_inputs).mean(axis=0)
    std_x = x.reshape(-1, n_inputs).std(axis=0)
    x -= mean_x
    x /= std_x
    if args.varlen_train:
        x = np.array(X_list)
    if args.two_phase:
        x = np.concatenate([x, x], axis=1)
    x = np.swapaxes(x, 2, 1)
    # shape: (N, 3, L-1)

    aux = np.c_[means, scales, np.log10(periods)]
    if args.use_meta and split[0].metadata is not None:
        metadata = np.array([lc.metadata for lc in split])  # Metadata must have same dimension!
        aux = np.c_[aux, metadata]                          # Concatenate metadata
        print('metadata will be used as auxiliary inputs.')
    aux_mean = aux.mean(axis=0)
    aux_std = aux.std(axis=0)
    aux -= aux_mean
    aux /= aux_std
    scales_all = np.array([np.append(mean_x, 0), np.append(std_x, 0), aux_mean, aux_std])
    if not args.varlen_train:
        scales_all = None
    else:
        np.save(name + '_scales.npy', scales_all)

    train_idx, val_idx = train_test_split(label, 1 - args.frac_valid, -1)
    if args.ngpu < 0:
        torch.cuda.set_device(int(-1*args.ngpu))
        map_loc = 'cuda:{}'.format(int(-1*args.ngpu))

    print('Using ', torch.cuda.current_device())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sys.stdout = sys.__stdout__
    train_dset = MyDataset(x[train_idx], aux[train_idx], label[train_idx])
    val_dset = MyDataset(x[val_idx], aux[val_idx], label[val_idx])
    train_loader = DataLoader(train_dset, batch_size=args.train_batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dset, batch_size=128, shuffle=False, drop_last=False)

    split = [chunk for i in test_index for chunk in data[i].split(args.L, args.L)]
    for lc in split:
        lc.period_fold()

    # shape: (N, L, 3)
    x_list = [np.c_[chunk.times, chunk.measurements, chunk.errors] for chunk in split]
    periods = np.array([lc.p for lc in split])
    # x, means, scales = getattr(PreProcessor, args.input)(np.array(x_list), periods)
    x, means, scales = preprocess(np.array(x_list), periods, use_error=args.use_error)

    # whiten data
    x -= mean_x
    x /= std_x
    if args.varlen_train:
        x = np.array(X_list)
    if args.two_phase:
        x = np.concatenate([x, x], axis=1)
    x = np.swapaxes(x, 2, 1)
    # shape: (N, 3, L)

    label = np.array([convert_label[chunk.label] for chunk in split])
    aux = np.c_[means, scales, np.log10(periods)]
    if args.use_meta and split[0].metadata is not None:
        metadata = np.array([lc.metadata for lc in split])  # Metadata must have same dimension!
        aux = np.c_[aux, metadata]                          # Concatenate metadata
        print('metadata will be used as auxiliary inputs.')
    aux -= aux_mean
    aux /= aux_std

    test_dset = MyDataset(x, aux, label)
    test_loader = DataLoader(test_dset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True)

    mdl = get_network(n_classes)
    if not args.no_log:
        import wandb
        wandb.init(project=args.project_name, config=args, name=name)
        wandb.watch(mdl)
    if not args.test:
        if args.retrain:
            mdl.load_state_dict(torch.load(name + '.pth', map_location=map_loc))
            args.lr *= 0.01
        optimizer = optim.Adam(mdl.parameters(), lr=args.lr)
        torch.manual_seed(args.seed)
        train(mdl, optimizer, train_loader, val_loader, test_loader, args.max_epoch,
              print_every=args.print_every, save=True, filename=name+args.note, patience=args.patience,
              early_stopping_limit=args.early_stopping, use_tqdm=True, scales_all=scales_all, clip=args.clip,
              retrain=args.retrain, decay_type=args.decay_type, monitor='accuracy', log=not args.no_log,
              perm=args.permute)

    # load the model with the best validation accuracy for testing on the test set
    mdl.load_state_dict(torch.load(name + args.note + '.pth', map_location=map_loc))

    # Evaluate model on sequences of different length
    accuracy_length = np.zeros(len(lengths))
    accuracy_class_length = np.zeros(len(lengths))
    mdl.eval()
    with torch.no_grad():
        for j, length in enumerate(lengths):
            split = [chunk for i in test_index for chunk in data[i].split(length, length)]
            for lc in split:
                lc.period_fold()

            # shape: (N, L, 3)
            x_list = [np.c_[chunk.times, chunk.measurements, chunk.errors] for chunk in split]
            periods = np.array([lc.p for lc in split])
            x, means, scales = preprocess(np.array(x_list), periods, use_error=args.use_error)
            # x, means, scales = getattr(PreProcessor, args.input)(np.array(x_list), periods)

            # whiten data
            x -= mean_x
            x /= std_x
            if args.two_phase:
                x = np.concatenate([x, x], axis=1)
            x = np.swapaxes(x, 2, 1)
            # shape: (N, 3, L)

            label = np.array([convert_label[chunk.label] for chunk in split])
            aux = np.c_[means, scales, np.log10(periods)]
            if args.use_meta and split[0].metadata is not None:
                metadata = np.array([lc.metadata for lc in split])  # Metadata must have same dimension!
                aux = np.c_[aux, metadata]  # Concatenate metadata
                print('metadata will be used as auxiliary inputs.')
            aux -= aux_mean
            aux /= aux_std

            test_dset = MyDataset(x, aux, label)
            test_loader = DataLoader(test_dset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True)
            softmax = torch.nn.Softmax(dim=1)
            predictions = []
            ground_truths = []
            probs = []
            for i, d in enumerate(test_loader):
                x, aux_, y = d
                logprob = mdl(x.type(dtype), aux_.type(dtype))
                predictions.extend(list(np.argmax(softmax(logprob).detach().cpu(), axis=1)))
                probs.extend(list(softmax(logprob).detach().cpu().numpy()))
                ground_truths.extend(list(y.numpy()))

            predictions = np.array(predictions)
            ground_truths = np.array(ground_truths)

            accuracy_length[j] = (predictions == ground_truths).mean()
            accuracy_class_length[j] = np.array(
                [(predictions[ground_truths == l] == ground_truths[ground_truths == l]).mean()
                 for l in np.unique(ground_truths)]).mean()
    if args.ngpu > 1:
        return_device(path, device)
    return accuracy_length, accuracy_class_length


if __name__ == '__main__':

    jobs = []
    np.random.seed(args.seed)
    for i in range(args.K):
        if args.K == 1:
            i = args.pseed
        trains, tests = train_test_split(all_labels, train_size=args.frac_train, random_state=i)
        jobs.append((trains, tests, '{}/{}-{}'.format(args.path, save_name, i)))
    try:
        os.mkdir(args.path)
    except:
        pass
    if args.ngpu <= 1 and args.njob == 1:
        results = []
        for j in jobs:
            results.append(train_helper(j))
    else:
        create_device('device'+save_name+args.note, args.ngpu, args.njob)
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.ngpu * args.njob) as p:
            results = p.map(train_helper, jobs)
        shutil.rmtree('device' + save_name+args.note)
    results = np.array(results)
    results_all = np.c_[lengths, results[:, 0, :].T]
    results_class = np.c_[lengths, results[:, 1, :].T]
    np.save('{}/{}{}-results.npy'.format(args.path, save_name, args.note), results_all)
    np.save('{}/{}{}-results-class.npy'.format(args.path, save_name, args.note), results_class)
