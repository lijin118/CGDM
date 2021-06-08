from __future__ import print_function
import argparse
import torch.optim as optim
from utils import *
from models.basenet import *
import torch.nn.functional as F
import os
import time
import numpy as np
import warnings
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio
import platform
from math import cos, sin, pi
from sklearn.datasets import make_moons
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('TkAgg')

warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser(description='Toy Classification')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.0003)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP', help='the name of optimizer')
parser.add_argument('-mode', type=str, default="sup_gmn",
                        choices=["source_only", "adapt_mcd", "adapt_swd","sup_gmn"])
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=4, metavar='K', help='how many steps to repeat the generator update')
parser.add_argument('--num_layer', type=int, default=2, metavar='K', help='how many layers for classifier')
parser.add_argument('--class_num', type=int, default='31', metavar='B', help='The number of classes')
parser.add_argument('--gmn_N', type=int, default='31', metavar='B', help='The number of classes to calulate gradient similarity')
parser.add_argument('--resnet', type=str, default='50', metavar='B', help='which resnet 18,50,101,152,200')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)

torch.cuda.manual_seed(args.seed)


num_k = args.num_k
num_layer = args.num_layer
lr = args.lr

option = 'resnet' + args.resnet
G = ResBottle(option)
F1 = ResClassifier(num_classes=2, num_layer=num_layer, num_unit=G.output_num(), middle=1000)
F2 = ResClassifier(num_classes=2, num_layer=num_layer, num_unit=G.output_num(), middle=1000)
F1.apply(weights_init)
F2.apply(weights_init)


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

def generate_grid_point():
    x_min, x_max = x_s[:, 0].min() - .5, x_s[:, 0].max() + 0.5
    y_min, y_max = x_s[:, 1].min() - .5, x_s[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    return xx, yy
def load_data_sk(theta=20, nb=250, noise=.05):
    X, y = make_moons(nb, noise=noise, random_state=1) 
    Xt, yt = make_moons(nb, noise=noise, random_state=2)
    
    trans = -np.mean(X, axis=0) 
    X  = 2*(X+trans)
    Xt = 2*(Xt+trans)
    
    theta = -theta*pi/180
    rotation = np.array( [  [cos(theta), sin(theta)], [-sin(theta), cos(theta)] ] )
    Xt = np.dot(Xt, rotation.T)
    # print X
    # print Xt
    return torch.from_numpy(X).float().cuda(), torch.from_numpy(y).float().cuda(), torch.from_numpy(Xt).float().cuda(), torch.from_numpy(yt).float().cuda()
                                       
#聚类获得伪标签                                       
def obtain_label_cluster(data_x_t, data_y_t, netE, netC1, netC2, c=None):
    netE.eval()
    netC1.eval()
    netC2.eval()
    with torch.no_grad():
        all_fea = netE(data_x_t)
        outputs1 = netC1(all_fea)
        #outputs2 = netC2(all_fea)
        all_output = outputs1.float().cpu() 
        all_label = data_y_t

    predict = (all_output > 0.5).float()
    #print("all_label:",all_label.size()[0],"right:",torch.squeeze(predict).float().eq(all_label.data).sum().item())
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Only source accuracy = {:.2f}% -> After the clustering = {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str+'\n')
    return pred_label.astype('int')

x_s, y_s, x_t, y_t = load_data_sk()

def train():
    
    criterion = nn.CrossEntropyLoss()
    criterion_w = Weighted_CrossEntropy 

    bs = 250
    since = time.time()
    for step in range(10001):
        if step%1000==0:
            print("Obtaining target label...")
            mem_label = obtain_label_cluster(x_t, y_t, G, F1, F2)
            mem_label = torch.from_numpy(mem_label).cuda()
            G.train()
            F1.train()
            F2.train()  

            print("Iteration: %d / %d" % (step, 10000))
            z = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
            with torch.no_grad():
                fea = G(z)
                Z = nn.Softmax(dim=1)(F1(fea))
            Z = Z.reshape(xx.shape)
            f = plt.figure()
            plt.contourf(xx, yy, Z, cmap=plt.cm.copper_r, alpha=0.9)
            #plt.contour(xx, yy, Z, cmap=plt.cm.copper_r, alpha=0.9)
            plt.scatter(x_s[:, 0], x_s[:, 1], c=y_s.reshape((len(x_s))),
                        cmap=plt.cm.coolwarm, alpha=0.8)
            plt.scatter(x_t[:, 0], x_t[:, 1], color='green', alpha=0.7)
            #plt.text(1.6, -0.9, 'Iter: ' + str(step), fontsize=14, color='#FFD700',
                     #bbox=dict(facecolor='dimgray', alpha=0.7))
            plt.axis('off')
            f.savefig('toy/'+args.mode + '_pytorch_iter' + str(step) + ".png", bbox_inches='tight',
                      pad_inches=0, dpi=100, transparent=True)
            gif_images.append(imageio.imread(
                              'toy/'+args.mode + '_pytorch_iter' + str(step) + ".png"))
            plt.close()

        """source domain discriminative"""
        # Step A train all networks to minimize loss on source
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        output = G(torch.cat((x_s,x_t),dim=0))
        output1 = F1(output)
        output2 = F2(output)
        output_s1 = output1[:bs, :]
        output_s2 = output2[:bs, :]
        output_t1 = output1[bs:, :]
        output_t2 = output2[bs:, :]

        supervision_loss = criterion_w(output_t1, mem_label) + criterion_w(output_t2, mem_label)

        loss1 = criterion(output_s1, y_s)
        loss2 = criterion(output_s2, y_s)
        all_loss = loss1 + loss2 + 0.01 * supervision_loss
        all_loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        """target domain diversity"""
        # Step B train classifier to maximize CDD loss
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        output = G(torch.cat((x_s,x_t),dim=0))
        output1 = F1(output)
        output2 = F2(output)
        output_s1 = output1[:bs, :]
        output_s2 = output2[:bs, :]
        output_t1 = output1[bs:, :]
        output_t2 = output2[bs:, :]

        loss1 = criterion(output_s1, y_s)
        loss2 = criterion(output_s2, y_s)

        loss_dis = discrepancy(output_t1,output_t2)

        all_loss = loss1 + loss2 - 1.0 * loss_dis 
        all_loss.backward()
        optimizer_f.step()

        """target domain discriminability"""
        # Step C train genrator to minimize CDD loss
        for i in range(num_k):
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            output = G(torch.cat((x_s,x_t),dim=0))
            output1 = F1(output)
            output2 = F2(output)
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1_s = F.softmax(output_t1)
            output_t2_s = F.softmax(output_t2)

            loss_dis = discrepancy(output_t1,output_t2)

            gmn_loss = gradient_mathing_loss_margin(args, output_s1,output_s2, y_s, output_t1, output_t2, mem_label, G, F1, F2)

            all_loss = 1.0 * loss_dis + 0.01 * gmn_loss

            all_loss.backward()
            optimizer_g.step()

        print('time:', time.time() - since)
        print('-' * 100)

train()

