from __future__ import print_function
import argparse
import torch
from utils import *
from taskcv_loader import CVDataLoader
from models.basenet import *
from torchvision import transforms, datasets
import torch.nn.functional as F
import os
import time
import numpy as np
import warnings
from data_loader.folder import ImageFolder_ind

from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser(description='ImageClef Classification')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='learning rate (default: 0.0003)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP', help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=100, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=4, metavar='K', help='how many steps to repeat the generator update')
parser.add_argument('--num_layer', type=int, default=2, metavar='K', help='how many layers for classifier')
parser.add_argument('--train_path', type=str, default='dataset/clef/i', metavar='B',
                    help='directory of source datasets')
parser.add_argument('--val_path', type=str, default='dataset/clef/p', metavar='B',
                    help='directory of target datasets')
parser.add_argument('--class_num', type=int, default='12', metavar='B', help='The number of classes')
parser.add_argument('--gmn_N', type=int, default='12', metavar='B', help='The number of classes to calulate gradient similarity')
parser.add_argument('--resnet', type=str, default='50', metavar='B', help='which resnet 18,50,101,152,200')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

source = 'c'
traget = 'i'

print(source, " to ", traget)

args.train_path = "dataset/clef/%s/" % source
args.val_path = "dataset/clef/%s/" % traget

train_path = args.train_path
val_path = args.val_path
num_k = args.num_k
num_layer = args.num_layer
batch_size = args.batch_size
lr = args.lr


data_transforms = {
    train_path: transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    val_path: transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dsets = {x: ImageFolder_ind(os.path.join(x), data_transforms[x]) for x in [train_path, val_path]}

dset_sizes = {x: len(dsets[x]) for x in [train_path, val_path]}
dset_classes = dsets[train_path].classes
print(dset_classes)
classes_acc = {}
for i in dset_classes:
    classes_acc[i] = []
    classes_acc[i].append(0)
    classes_acc[i].append(0)

train_loader = CVDataLoader()
train_loader.initialize(dsets[train_path], dsets[val_path], batch_size, shuffle=True, drop_last=False)
dataset = train_loader.load_data()
test_loader = CVDataLoader()
test_loader.initialize(dsets[train_path], dsets[val_path], batch_size, shuffle=False, drop_last=False)
dataset_test = test_loader.load_data()

option = 'resnet' + args.resnet
G = torch.load(os.path.join("models_trained","DANN","extractor.pth"))
G.eval()
if args.cuda:
    G.cuda()
feature = []

print('-' * 100, '\nTesting')
start_test = True
#source
for batch_idx, data in enumerate(dataset):
    if dataset.stop_S:
        break
    if args.cuda:
        img = data['S']
        img = img.cuda()
    img = Variable(img, volatile=True)
    with torch.no_grad():
        feas = G(img)
        print(feas.size())
    if start_test:
        all_fea = feas.float().cpu()
        start_test = False
    else:
        all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
#target
for batch_idx, data in enumerate(dataset_test):
    if dataset_test.stop_T:
        break
    if args.cuda:
        img = data['T']
        img = img.cuda()
    img = Variable(img, volatile=True)
    with torch.no_grad():
        feas = G(img)
    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)

all_fea = all_fea.numpy()
tsne = TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(all_fea)
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
plt.axis('off')
plt.plot(X_norm[:600,0], X_norm[:600,1], 'r.')
plt.plot(X_norm[600:,0], X_norm[600:,1], 'g.')

plt.savefig("Visualization_DANN.pdf")


