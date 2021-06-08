from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import *
import imageio
import platform
from math import cos, sin, pi
from sklearn.datasets import make_moons
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('TkAgg')

def toyNet():
    # Define network architecture
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.l1 = nn.Linear(2, 15)
            self.l2 = nn.Linear(15, 15)
            self.l3 = nn.Linear(15, 15)
            self.relu = nn.ReLU(inplace=True)

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

        def forward(self, x):
            x = self.relu(self.l1(x))
            x = self.relu(self.l2(x))
            x = self.relu(self.l3(x))
            return x
    class Classifier1(nn.Module):
        def __init__(self):
            super(Classifier1, self).__init__()
            self.l1 = nn.Linear(15, 15)
            self.l2 = nn.Linear(15, 15)
            self.l3 = nn.Linear(15, 1)
            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

        def forward(self, x):
            x = self.relu(self.l1(x))
            x = self.relu(self.l2(x))
            x = self.sigmoid(self.l3(x))
            return x
    class Classifier2(nn.Module):
        def __init__(self):
            super(Classifier2, self).__init__()
            self.l1 = nn.Linear(15, 15)
            self.l2 = nn.Linear(15, 15)
            self.l3 = nn.Linear(15, 1)
            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

        def forward(self, x):
            x = self.relu(self.l1(x))
            x = self.relu(self.l2(x))
            x = self.sigmoid(self.l3(x))
            return x
    return Generator(), Classifier1(), Classifier2()

def discrepancy_slice_wasserstein(p1, p2):
    s = p1.shape
    if s[1]>1:
        proj = torch.randn(s[1], 128)
        proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
        p1 = torch.matmul(p1, proj)
        p2 = torch.matmul(p2, proj)
    p1 = torch.topk(p1, s[0], dim=0)[0]
    p2 = torch.topk(p2, s[0], dim=0)[0]
    dist = p1-p2
    wdist = torch.mean(torch.mul(dist, dist))
    
    return wdist

def discrepancy_mcd(out1, out2):
    return torch.mean(torch.abs(out1 - out2))


def load_data():
    # Load inter twinning moons 2D dataset by F. Pedregosa et al. in JMLR 2011
    moon_data = np.load('moon_data.npz')
    x_s = moon_data['x_s']
    y_s = moon_data['y_s']
    x_t = moon_data['x_t']

    return torch.from_numpy(x_s).float(), torch.from_numpy(y_s).float(), torch.from_numpy(x_t).float()

def load_data_sk(theta=20, nb=200, noise=.05):
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
    return torch.from_numpy(X).float(), torch.from_numpy(y).float(), torch.from_numpy(Xt).float(), torch.from_numpy(yt).float()

def generate_grid_point():
    x_min, x_max = x_s[:, 0].min() - .5, x_s[:, 0].max() + 0.5
    y_min, y_max = x_s[:, 1].min() - .5, x_s[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    return xx, yy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default="sup_gmn",
                        choices=["source_only", "adapt_mcd", "adapt_swd","sup_gmn"])
    parser.add_argument('-seed', type=int, default=1234)
    opts = parser.parse_args()

    # Load data
    x_s, y_s, x_t, y_t = load_data_sk()
    #print size
    #print("x_s:",x_s.size(),"y_s:",y_s,"x_t:",x_t.size())

    # set random seed
    torch.manual_seed(opts.seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

    # Network definition
    generator, cls1, cls2 = toyNet()
    generator.train()
    cls1.train()
    cls2.train()

    # Cost functions
    bce_loss = nn.BCELoss()
    bce_loss_w = Weighted_BCE

    # Setup optimizers
    optim_g = torch.optim.SGD(generator.parameters(), lr=0.005)
    optim_f = torch.optim.SGD(list(cls1.parameters())+list(cls2.parameters()), lr=0.005)
    optim_g.zero_grad()
    optim_f.zero_grad()

    # # Generate grid points for visualization
    xx, yy = generate_grid_point()

    # For creating GIF purpose
    gif_images = []

    for step in range(10001):
        if step%1000==0:
            if opts.mode == 'sup_gmn':
                p_label = obtain_label(x_t, y_t, generator, cls1, cls2)
                generator.train()
                cls1.train()
                cls2.train()     

            print("Iteration: %d / %d" % (step, 10000))
            z = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
            with torch.no_grad():
                fea = generator(z)
                Z = (cls2(fea).cpu().numpy()>0.5).astype(np.float32)
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
            f.savefig(opts.mode + '_pytorch_iter' + str(step) + ".png", bbox_inches='tight',
                      pad_inches=0, dpi=100, transparent=True)
            gif_images.append(imageio.imread(
                              opts.mode + '_pytorch_iter' + str(step) + ".png"))
            plt.close()

        optim_g.zero_grad()
        optim_f.zero_grad()
        fea = generator(x_s)
        pred1 = cls1(fea)
        pred2 = cls2(fea)
        loss_s = bce_loss(pred1, y_s) + bce_loss(pred2, y_s)
        loss_s.backward()
        # if opts.mode == 'sup_gmn':
        #     fea_t = generator(x_t)
        #     predt1,predt2 = cls1(fea_t), cls2(fea_t)
        #     loss_t = (bce_loss_w(predt1, p_label) + bce_loss_w(predt2, p_label)) * 0.0005
        #     loss_t.backward()
        optim_g.step()
        optim_f.step()

        if opts.mode == 'source_only':
            continue
        
        optim_g.zero_grad()
        optim_f.zero_grad()
        loss = 0
        src_fea = generator(x_s)
        src_fea = src_fea.detach()
        src_pred1 = cls1(src_fea)
        src_pred2 = cls2(src_fea)
        loss += bce_loss(src_pred1, y_s) + bce_loss(src_pred2, y_s)
        # loss_s.backward()

        tgt_fea = generator(x_t)
        tgt_fea = tgt_fea.detach()
        tgt_pred1 = cls1(tgt_fea)
        tgt_pred2 = cls2(tgt_fea)
        if opts.mode == 'adapt_swd':
            loss_dis = 2*discrepancy_slice_wasserstein(tgt_pred1, tgt_pred2)
        else:
            loss_dis = discrepancy_mcd(tgt_pred1, tgt_pred2)
        loss -= loss_dis
        loss.backward()
        optim_f.step()

        optim_g.zero_grad()
        tgt_fea = generator(x_t)
        tgt_pred1 = cls1(tgt_fea)
        tgt_pred2 = cls2(tgt_fea)
        if opts.mode == 'adapt_swd':
            loss_dis = discrepancy_slice_wasserstein(tgt_pred1, tgt_pred2)
        else:
            loss_dis = discrepancy_mcd(tgt_pred1, tgt_pred2)
        if opts.mode == 'sup_gmn':
            fea_s = generator(x_s)
            src_pred1,src_pred2 = cls1(fea_s), cls2(fea_s)
            gmn_loss = gradient_mathing_loss_margin(src_pred1,src_pred2, y_s, tgt_pred1, tgt_pred2, p_label, generator, cls1, cls2)
            print(loss_dis,gmn_loss)
            loss_dis += 0.00001* gmn_loss
        loss_dis.backward()
        optim_g.step()
    
    # Save GIF
    imageio.mimsave(opts.mode + '_pytorch.gif', gif_images, duration=0.8)
    print("[Finished]\n-> Please see the current folder for outputs.")
