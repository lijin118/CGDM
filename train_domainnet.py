from __future__ import print_function
import argparse
import torch.optim as optim
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

warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser(description='DomainNet Classification')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.0003)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP', help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=4, metavar='K', help='how many steps to repeat the generator update')
parser.add_argument('--num_layer', type=int, default=2, metavar='K', help='how many layers for classifier')
parser.add_argument('--train_path', type=str, default='/home/dzk/DomainNet/', metavar='B',
                    help='directory of source datasets')
parser.add_argument('--val_path', type=str, default='/home/dzk/DomainNet/', metavar='B',
                    help='directory of target datasets')
parser.add_argument('--class_num', type=int, default='345', metavar='B', help='The number of classes')
parser.add_argument('--gmn_N', type=int, default='345', metavar='B', help='The number of classes to calulate gradient similarity')
parser.add_argument('--resnet', type=str, default='50', metavar='B', help='which resnet 18,50,101,152,200')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

source = 'real'
traget = 'sketch'

print(source," to ", traget)

args.train_path = "/home/dzk/DomainNet/%s/" % source
args.val_path = "/home/dzk/DomainNet/%s/" % traget

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
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    val_path: transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dsets = {x: ImageFolder_ind(os.path.join(x), data_transforms[x]) for x in [train_path, val_path]}
dsets_tgt_no_shuffle = ImageFolder_ind(os.path.join(val_path), data_transforms[val_path])
data_loader_T_no_shuffle = torch.utils.data.DataLoader(
            dsets_tgt_no_shuffle,
            batch_size=32,
            shuffle=False,
            drop_last=False,
            num_workers=4)

dset_sizes = {x: len(dsets[x]) for x in [train_path, val_path]}
dset_classes = dsets[train_path].classes
print(dset_classes)
classes_acc = {}
for i in dset_classes:
    classes_acc[i] = []
    classes_acc[i].append(0)
    classes_acc[i].append(0)

train_loader = CVDataLoader()
train_loader.initialize(dsets[train_path], dsets[val_path], batch_size, shuffle=True, drop_last=True)
dataset = train_loader.load_data()
test_loader = CVDataLoader()
test_loader.initialize(dsets[train_path], dsets[val_path], batch_size, shuffle=False, drop_last=False)
dataset_test = test_loader.load_data()

option = 'resnet' + args.resnet
G = ResBottle(option)
F1 = ResClassifier(num_classes=345, num_layer=num_layer, num_unit=G.output_num(), middle=1000)
F2 = ResClassifier(num_classes=345, num_layer=num_layer, num_unit=G.output_num(), middle=1000)
F1.apply(weights_init)
F2.apply(weights_init)

if args.cuda:
    G.cuda()
    F1.cuda()
    F2.cuda()
if args.optimizer == 'momentum':
    optimizer_g = optim.SGD(list(G.features.parameters()), lr=args.lr, weight_decay=0.0005)
    optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9, lr=args.lr,
                            weight_decay=0.0005)
elif args.optimizer == 'adam':
    optimizer_g = optim.Adam(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)
else:
    optimizer_g = optim.Adadelta(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_f = optim.Adadelta(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)

start = 0

def train(num_epoch):
    
    criterion = nn.CrossEntropyLoss()
    criterion_w = Weighted_CrossEntropy 

    for ep in range(num_epoch):
        
        since = time.time()

        print("Obtaining target label...")
        mem_label = obtain_label(data_loader_T_no_shuffle, G, F1, F2, args)
        mem_label = torch.from_numpy(mem_label).cuda()
        for batch_idx, data in enumerate(dataset):
            if dataset.stop_S:
                break
            G.train()
            F1.train()
            F2.train()

            data_s = data['S']
            label_s = data['S_label']
            data_t = data['T']
            label_t = data['T_label']
            index_t = data['T_index']
            if ep > start:
                pseudo_label_t = mem_label[index_t]


            if args.cuda:
                data_s, label_s = data_s.cuda(), label_s.cuda()
                data_t, label_t = data_t.cuda(), label_t.cuda()
                if ep > start:
                    pseudo_label_t = pseudo_label_t.cuda()
            data_all = Variable(torch.cat((data_s, data_t), 0))
            label_s = Variable(label_s)
            bs = len(label_s)

            """source domain discriminative"""
            # Step A train all networks to minimize loss on source
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output = G(data_all)
            output1 = F1(output)
            output2 = F2(output)
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1_s = F.softmax(output_t1)
            output_t2_s = F.softmax(output_t2)

            entropy_loss = Entropy(output_t1_s)
            entropy_loss += Entropy(output_t2_s)

            if ep > start:
                supervision_loss = criterion_w(output_t1, pseudo_label_t) + criterion_w(output_t2, pseudo_label_t)
            else:
                supervision_loss = 0
            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)
            all_loss = loss1 + loss2 + 0.1 * entropy_loss + 0.01 * supervision_loss
            all_loss.backward()
            optimizer_g.step()
            optimizer_f.step()

            """target domain diversity"""
            # Step B train classifier to maximize CDD loss
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output = G(data_all)
            output1 = F1(output)
            output2 = F2(output)
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1_s = F.softmax(output_t1)
            output_t2_s = F.softmax(output_t2)

            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)

            entropy_loss = Entropy(output_t1_s)
            entropy_loss += Entropy(output_t2_s)

            loss_dis = discrepancy(output_t1,output_t2)

            all_loss = loss1 + loss2 - 1.0 * loss_dis + 0.1 * entropy_loss
            all_loss.backward()
            optimizer_f.step()

            """target domain discriminability"""
            # Step C train genrator to minimize CDD loss
            for i in range(num_k):
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                output = G(data_all)
                output1 = F1(output)
                output2 = F2(output)
                output_s1 = output1[:bs, :]
                output_s2 = output2[:bs, :]
                output_t1 = output1[bs:, :]
                output_t2 = output2[bs:, :]
                output_t1_s = F.softmax(output_t1)
                output_t2_s = F.softmax(output_t2)

                entropy_loss = Entropy(output_t1_s)
                entropy_loss += Entropy(output_t2_s)

                loss_dis = discrepancy(output_t1,output_t2)
                #print(pseudo_label_t)
                if ep > start:
                    gmn_loss = gradient_discrepancy_loss_margin(args, output_s1,output_s2, label_s, output_t1, output_t2, pseudo_label_t, G, F1, F2)
                else: 
                    gmn_loss = 0

                all_loss = 1.0 * loss_dis + 0.1 * entropy_loss + 0.01 * gmn_loss

                all_loss.backward()
                optimizer_g.step()

            if batch_idx % args.log_interval == 0:
                print(
                    'Train Ep: {} [{}/{} ({:.6f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t CDD: {:.6f} Entropy: {:.6f} '.format(
                        ep, batch_idx, len(dataset.data_loader_S), 100. * batch_idx / len(dataset.data_loader_S),
                        loss1.item(), loss2.item(), loss_dis.item(), entropy_loss.item()))


        # test
        test(ep)
        print('time:', time.time() - since)
        print('-' * 100)


def test(epoch):
    G.eval()
    F1.eval()
    F2.eval()
    test_loss = 0
    correct_add = 0
    size = 0
    print('-' * 100, '\nTesting')
    for batch_idx, data in enumerate(dataset_test):
        if dataset_test.stop_T:
            break
        if args.cuda:
            img = data['T']
            label = data['T_label']
            img, label = img.cuda(), label.cuda()
        img, label = Variable(img, volatile=True), Variable(label)
        with torch.no_grad():
            output = G(img)
            output1 = F1(output)
            output2 = F2(output)
        test_loss += F.nll_loss(output1, label).item()
        output_add = output1 + output2
        pred = output_add.data.max(1)[1]
        correct_add += pred.eq(label.data).cpu().sum()
        size += label.data.size()[0]
        for i in range(len(label)):
            key_label = dset_classes[label.long()[i].item()]
            key_pred = dset_classes[pred.long()[i].item()]
            classes_acc[key_label][1] += 1
            if key_pred == key_label:
                classes_acc[key_pred][0] += 1

    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('Epoch: {:d} Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
        epoch, test_loss, correct_add, size, 100. * float(correct_add) / size))
    avg = []
    for i in dset_classes:
        # print('\t{}: [{}/{}] ({:.6f}%)'.format(i, classes_acc[i][0], classes_acc[i][1],
        #                                        100. * classes_acc[i][0] / classes_acc[i][1]))
        avg.append(100. * float(classes_acc[i][0]) / classes_acc[i][1])
    print('\taverage:', np.average(avg))
    for i in dset_classes:
        classes_acc[i][0] = 0
        classes_acc[i][1] = 0


train(args.epochs + 1)

