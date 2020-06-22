import os
import argparse

parser = argparse.ArgumentParser(description='Periodic Permuted Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',default=True,
                    help='use CUDA (default: True)')
parser.add_argument('--periodic', action='store_true',default=False,
                    help='PP-MNIST if true; P-MNIST if false.')
parser.add_argument('--augment', action='store_true',default=False,
                    help='use data augmentation')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='depth of network')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--nhid_max', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--network', type=str, default='tcn',
                    help='network name itcn/tcn/itin/tin/iresnet/resnet/gru/lstm')
parser.add_argument('--path', type=str, default='results',
                    help='network')
parser.add_argument('--project', type=str, default='none',
                    help='for weights and biases tracking')
parser.add_argument('--ngpu', type=str, default='0',
                    help='gpu device number to use when gpu_count > 1')
parser.add_argument('--name', type=str, default='none',
                    help='save filename')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.ngpu
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from data_ppmnist import data_generator
from model.resnet import Classifier as resnet
from model.itcn import Classifier as itcn
from model.itin import Classifier as itin
from model.rnn import Classifier as rnn
import numpy as np
import wandb

if 'tcn' not in args.network:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

root = './data/mnist'
name = args.path + '/{}_{}_{}_{}_{}'.format(args.network, args.nhid, args.nhid_max, args.dropout, args.levels)
print(name)
wandb.init(project=args.project, config=args, name=name)

batch_size = args.batch_size
n_classes = 10
input_channels = 1
seq_length = int(784 / input_channels)
epochs = 25 if args.network in ['iresnet','resnet'] else 40
steps = 0
test_accuracy = []
train_accuracy = []

print(args)
train_loader, test_loader = data_generator(root, batch_size)

permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize

if args.network == 'itcn':
    model = itcn(input_channels, channel_sizes, n_classes, hidden=1, kernel_size=kernel_size, dropout=args.dropout,
                 padding='cyclic')
elif args.network == 'tcn':
    model = itcn(input_channels, channel_sizes, n_classes, hidden=1, kernel_size=kernel_size, dropout=args.dropout,
                 padding='zero')
elif args.network == 'iresnet':
    model = resnet(input_channels, n_classes, depth=args.levels, nlayer=2, kernel_size=kernel_size,
                   hidden_conv=args.nhid, max_hidden=args.nhid_max, padding='cyclic', aux=0,
                   dropout_classifier=args.dropout, hidden=32)
elif args.network == 'resnet':
    model = resnet(input_channels, n_classes, depth=args.levels, nlayer=2, kernel_size=kernel_size,
                   hidden_conv=args.nhid, max_hidden=args.nhid_max, padding='zero', aux=0,
                   dropout_classifier=args.dropout, hidden=32)

if args.cuda:
    model.cuda()
    permute = permute.cuda()
wandb.watch(model)

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train(ep):
    global steps
    train_accuracies = []
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        data = data[:, :, permute]
        if args.periodic:
            for i in range(data.shape[0]):
                start = np.random.randint(0, seq_length - 1)
                data[i] = torch.cat((data[i, :, start:], data[i, :, :start]), dim=1)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss.cpu().detach().numpy()
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            train_accuracy.append(train_loss.item()/args.log_interval)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss/args.log_interval, steps))
            train_accuracies.append(train_loss/args.log_interval)
            train_loss = 0

    return np.array(train_accuracies).mean()


def test(save=False):
    model.eval()
    test_loss = 0
    correct = 0
    global test_accuracy
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            data = data[:, :, permute]
            if args.permute:
                for i in range(data.shape[0]):
                    start = np.random.randint(0, seq_length - 1)
                    data[i] = torch.cat((data[i, :, start:], data[i, :, :start]), dim=1)
            data, target = Variable(data, volatile=True), Variable(target)
            output = F.log_softmax(model(data), dim=1)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        test_accuracy.append(100. * correct.numpy() / len(test_loader.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * float(correct) / len(test_loader.dataset)))
        return test_loss


if __name__ == "__main__":

    if args.network not in ['itcn', 'tcn']:
        every = 10
    else:
        every = 15
    epochs = int(epochs)

    for epoch in range(1, epochs+1):
        if not args.augment:
            np.random.seed(args.seed)
        train_loss = train(epoch)
        test_loss = test()
        wandb.log({"Train Loss": train_loss,
                   "Test Loss": test_loss,
                   "Test Acc": test_accuracy[-1]})
        if epoch % every == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    np.save(name + '_test.npy', np.array(test_accuracy))
    np.save(name + '_train.npy', np.array(train_accuracy))
