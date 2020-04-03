import sys
import joblib
import argparse
import multiprocessing as mp
sys.path.append('./')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.resnet import Classifier as resnet
from model.itcn import Classifier as itcn
from model.itin import Classifier as itin
from model.rnn import Classifier as rnn
from data import MyDataset as MyDataset

from light_curve import LightCurve
from util import *
from _train import train

parser = argparse.ArgumentParser(description='')
parser.add_argument('--L', type=int, default=128,
                    help='training sequence length')
parser.add_argument('--varlen_train', action='store_true', default=False,
                    help='enable variable length training')
parser.add_argument('--input', type=str, default='dtf',
                    help='input representation of data. combination of t/dt/f/df/g.')
parser.add_argument('--l_min', type=int, default=128,
                    help='test sequence minimum length')
parser.add_argument('--l_max', type=int, default=128,
                    help='test sequence maximum length')
parser.add_argument('--n_test', type=int, default=1,
                    help='number of different sequence length to test')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout rate')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clipping')
parser.add_argument('--survey', type=str, default='macho',
                    help='dataset name; dataset file at ./survey/filename')
parser.add_argument('--filename', type=str, default='raw.pkl',
                    help='filename: raw.pkl / cleaned.pkl; ./survey/filename')
parser.add_argument('--path', type=str, default='temp',
                    help='folder name to save experiement results')
parser.add_argument('--max_epoch', type=int, default=50,
                    help='maximum number of training epochs')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of gpus available')
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
parser.add_argument('--min_sample_test', type=int, default=0,
                    help='minimum number of pre-segmented light curve per class during testing')
parser.add_argument('--max_sample', type=int, default=20000,
                    help='maximum number of pre-segmented light curve per class during testing')
parser.add_argument('--test', action='store_true', default=False,
                    help='test pre-trained model')
parser.add_argument('--ssoff', action='store_true', default=False,
                    help='turn off super smoother')
parser.add_argument('--retrain', action='store_true', default=False,
                    help='continue training from checkpoint')
parser.add_argument('--note', type=str, default='',
                    help='')
parser.add_argument('--patience', type=int, default=5,
                    help='patience for learning decay')
parser.add_argument('--early_stopping', type=int, default=15,
                    help='terminate training if loss does not improve by 10% after waiting this number of epochs')
args = parser.parse_args()
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dint = torch.cuda.LongTensor
else:
    assert args.ngpu == 1
    dtype = torch.FloatTensor
    dint = torch.LongTensor
if args.min_sample_test == -1:
    args.min_sample_test = args.min_sample
if args.cudnn_deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if args.network == 'resnet' or args.network == 'iresnet':
    save_name = '{}-{}-K{}-D{}-NL{}-H{}-MH{}-L{}-V{}-{}-LR{}-CLIP{}-DROP{}-TP{}'.format(args.survey,
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
                                                                              args.clip,
                                                                              args.dropout,
                                                                              int(args.two_phase))
else:
    save_name = '{}-{}-K{}-D{}-H{}-L{}-V{}-{}-LR{}-CLIP{}-DROP{}-TP{}'.format(args.survey,
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
print(save_name)
lengths = np.linspace(args.l_min, args.l_max, args.n_test).astype(np.int)
if args.L not in lengths:
    lengths = np.sort(np.append(lengths, args.L))
survey = args.survey
data = joblib.load('data/{}/{}'.format(survey, args.filename))
data = [lc for lc in data if lc.label is not None]
if not args.ssoff:
    data = [lc for lc in data if lc.ss_resid <= 0.9]
    print('super smooth threshold = 0.9')

if args.survey == 'macho':
    for lc in data:
        if 'LPV' in lc.label:
            lc.label = "LPV"

# Generate a list all labels for train/test split
all_label_string = [lc.label for lc in data]
unique_label, count = np.unique(all_label_string, return_counts=True)
use_label = unique_label[count >= args.min_sample]
test_label = unique_label[count >= args.min_sample_test]

n_classes = len(use_label)
data = [lc for lc in data if lc.label in use_label]
new_data = []
for cls in use_label:
    class_data = [lc for lc in data if lc.label == cls]
    new_data.extend(class_data[:min(len(class_data), args.max_sample)])
data = new_data
all_label_string = [lc.label for lc in data]
unique_label, count = np.unique(all_label_string, return_counts=True)
print(unique_label)
print(count)
convert_label = dict(zip(use_label, np.arange(len(use_label))))
all_labels = np.array([convert_label[lc.label] for lc in data])
test_label = np.array([convert_label[label] for label in test_label])

# sanity check on dataset
for lc in data:
    positive = lc.errors > 0
    positive *= lc.errors < 99
    lc.times = lc.times[positive]
    lc.measurements = lc.measurements[positive]
    lc.errors = lc.errors[positive]


if args.input in ['dtdfg', 'dtfg']:
    n_inputs = 3
elif args.input in ['df', 'f', 'g']:
    n_inputs = 1
else:
    n_inputs = 2


def get_network(n_classes):
    if args.network == 'itcn':
        clf = itcn(num_inputs=n_inputs, num_channels=[args.hidden] * args.depth, num_class=n_classes, hidden=32,
                   dropout=args.dropout, kernel_size=args.kernel, dropout_classifier=0, aux=3,
                   padding='cyclic').type(dtype)
    elif args.network == 'tcn':
        clf = itcn(num_inputs=n_inputs, num_channels=[args.hidden] * args.depth, num_class=n_classes, hidden=32,
                   dropout=args.dropout, kernel_size=args.kernel, dropout_classifier=0, aux=3,
                   padding='zero').type(dtype)
    elif args.network == 'itin':
        clf = itin(num_inputs=n_inputs, kernel_sizes=[args.kernel,args.kernel+2,args.kernel+4],
                   num_channels=[args.hidden] * args.depth, num_class=n_classes, hidden=32, dropout=args.dropout,
                   dropout_classifier=0, aux=3, padding='cyclic').type(dtype)
    elif args.network == 'tin':
        clf = itin(num_inputs=n_inputs, kernel_sizes=[args.kernel,args.kernel+2,args.kernel+4],
                   num_channels=[args.hidden] * args.depth, num_class=n_classes, hidden=32,
                   dropout=args.dropout, dropout_classifier=0, aux=3, padding='zero').type(dtype)
    elif args.network == 'iresnet':
        clf = resnet(n_inputs, n_classes, depth=args.depth, nlayer=args.n_layer, kernel_size=args.kernel,
                     hidden_conv=args.hidden, max_hidden=args.max_hidden, padding='cyclic',
                     aux=3, dropout_classifier=0, hidden=32).type(dtype)
    elif args.network == 'resnet':
        clf = resnet(n_inputs, n_classes, depth=args.depth, nlayer=args.n_layer, kernel_size=args.kernel,
                     hidden_conv=args.hidden, max_hidden=args.max_hidden, padding='zero',
                     aux=3, dropout_classifier=0, hidden=32).type(dtype)
    elif args.network == 'gru':
        clf = rnn(num_inputs=n_inputs, hidden_rnn=args.hidden, num_layers=args.depth, num_class=n_classes, hidden=32,
                  rnn='GRU', dropout=args.dropout, aux=3).type(dtype)
    elif args.network == 'lstm':
        clf = rnn(num_inputs=n_inputs, hidden_rnn=args.hidden, num_layers=args.depth, num_class=n_classes, hidden=32,
                  rnn='LSTM', dropout=args.dropout, aux=3).type(dtype)
    return clf


def train_helper(param):
    train_index, test_index, name = param
    split = [chunk for i in train_index for chunk in data[i].split(args.L, args.L) if data[i].label is not None]
    for lc in split:
        lc.period_fold()
    all_label_string = [lc.label for lc in split]
    unique_label, count = np.unique(all_label_string, return_counts=True)
    print('------------after segmenting into L={}------------'.format(args.L))
    print(unique_label)
    print(count)

    # shape: (N, L, 3)
    X_list = [np.c_[chunk.times, chunk.measurements, chunk.errors] for chunk in split]
    periods = np.array([lc.p for lc in split])
    label = np.array([convert_label[chunk.label] for chunk in split])

    x, means, scales = getattr(PreProcessor, args.input)(np.array(X_list), periods)
    print(x.shape)
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
    aux_mean = aux.mean(axis=0)
    aux_std = aux.std(axis=0)
    aux -= aux_mean
    aux /= aux_std
    scales_all = np.array([np.append(mean_x, 0), np.append(std_x, 0), aux_mean, aux_std])
    if not args.varlen_train:
        scales_all = None
    else:
        np.save(name + '_scales.npy', scales_all)

    train_idx, val_idx = train_test_split(label, 0.75, -1)
    if args.ngpu > 1:
        path = 'device'+save_name
        device = get_device(path)
        torch.cuda.set_device(int(device[0]))
    else:
        if args.ngpu < 0:
            torch.cuda.set_device(int(-1*args.ngpu))
    #print('Using ', torch.cuda.current_device())
    torch.manual_seed(args.seed)
    train_dset = MyDataset(x[train_idx], aux[train_idx], label[train_idx])
    val_dset = MyDataset(x[val_idx], aux[val_idx], label[val_idx])
    train_loader = DataLoader(train_dset, batch_size=32, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dset, batch_size=512, shuffle=False, drop_last=False, pin_memory=True)

    mdl = get_network(n_classes)

    if not args.test:
        if args.retrain:
            mdl.load_state_dict(torch.load(name + '.pth'))
            args.lr *= 0.01
            print(args.lr)
        optimizer = optim.Adam(mdl.parameters(), lr=args.lr)
        train(mdl, optimizer, train_loader, val_loader, args.max_epoch,
              print_every=args.print_every, save=True, filename=name+args.note, patience=args.patience,
              early_stopping_limit=args.early_stopping, use_tqdm=False, scales_all=scales_all, clip=args.clip,
              retrain=args.retrain)

    # load the model with the best validation accuracy for testing on the test set
    mdl.load_state_dict(torch.load(name + '.pth'))

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
            x, means, scales = getattr(PreProcessor, args.input)(np.array(x_list), periods)

            # whiten data
            x -= mean_x
            x /= std_x
            if args.two_phase:
                x = np.concatenate([x, x], axis=1)
            x = np.swapaxes(x, 2, 1)
            # shape: (N, 3, L)

            label = np.array([convert_label[chunk.label] for chunk in split])
            aux = np.c_[means, scales, np.log10(periods)]
            aux -= aux_mean
            aux /= aux_std

            test_dset = MyDataset(x, aux, label)
            if length > 200:
                batch_size = 32
            else:
                batch_size = 256
            test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
            softmax = torch.nn.Softmax(dim=1)
            predictions = []
            ground_truths = []
            for i, d in enumerate(test_loader):
                x, aux_, y = d
                logprob = mdl(x.type(dtype), aux_.type(dtype))
                predictions.extend(list(np.argmax(softmax(logprob).detach().cpu(), axis=1)))
                ground_truths.extend(list(y.numpy()))

            predictions = np.array(predictions)
            ground_truths = np.array(ground_truths)
            accuracy_length[j] = (predictions == ground_truths).mean()
            accuracy_class_length[j] = np.array(
                [(predictions[ground_truths == l] == ground_truths[ground_truths == l]).mean() for l in
                 np.unique(ground_truths) if l in test_label]).mean()
    if args.ngpu > 1:
        return_device(path, device)
    return accuracy_length, accuracy_class_length


if __name__ == '__main__':
    jobs = []
    LightCurve = LightCurve
    np.random.seed(args.seed)
    for i in range(args.K):
        if args.K == 1:
            i = args.pseed
        trains, tests = train_test_split(all_labels, train_size=0.8, random_state=i)
        jobs.append((trains, tests, '{}/{}-{}'.format(args.path, save_name, i)))
    try:
        os.mkdir(args.path)
    except:
        pass
    if args.ngpu <= 1:
        results = []
        for j in jobs:
            results.append(train_helper(j))
    else:
        create_device('device'+save_name, args.ngpu, args.njob)
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.ngpu * args.njob) as p:
            results = p.map(train_helper, jobs)
    results = np.array(results)
    print(results.shape)
    results_all = np.c_[lengths, results[:, 0, :].T]
    results_class = np.c_[lengths, results[:, 1, :].T]
    np.save('{}/{}{}-results.npy'.format(args.path, save_name, args.note), results_all)
    np.save('{}/{}{}-results-class.npy'.format(args.path, save_name, args.note), results_class)
