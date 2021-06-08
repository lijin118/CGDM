import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable, Function
import math
import pdb
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
from .basenet import *


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class LeNetClassifier(nn.Module):
    def __init__(self, prob=0.5):
          super(LeNetClassifier, self).__init__()
          self.fc1 = nn.Linear(48*5*5, 100)
          self.bn1_fc = nn.BatchNorm1d(100)
          self.fc2 = nn.Linear(100, 100)
          self.bn2_fc = nn.BatchNorm1d(100)
          self.fc3 = nn.Linear(100, 10)
          #self.bn_fc3 = nn.BatchNorm1d(10)
          self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd
    def forward(self, x):
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        return x


class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self):
      super(LeNetEncoder, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
      self.bn1 = nn.BatchNorm2d(32)
      self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
      self.bn2 = nn.BatchNorm2d(48)

    def forward(self, x):
        x = torch.mean(x,1).view(x.size()[0],1,x.size()[2],x.size()[3])
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, dilation=(1, 1))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, dilation=(1, 1))
        #print(x.size())
        x = x.view(x.size(0), 48*5*5)
        return x

def load_LeNet(num_class=10):

    return LeNetEncoder(),LeNetClassifier(),LeNetClassifier()

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

def load_pretrain_resnet(args, num_class=31):
    option = 'resnet' + args.resnet
    G = ResBottle(option)
    F1 = ResClassifier(num_classes=num_class, num_layer=args.res_cls_num_layer, num_unit=G.output_num(), middle=1000)
    F2 = ResClassifier(num_classes=num_class, num_layer=args.res_cls_num_layer, num_unit=G.output_num(), middle=1000)
    return G,F1,F2


class Resnet50_Feature(nn.Module):
    def __init__(self):
        super(Resnet50_Feature, self).__init__()
        self.sharedNet = resnet50(False)

    def forward(self, input_data):
        x = self.sharedNet(input_data)
        return x


class Resnet50_Predictor(nn.Module):
    def __init__(self, num_classes=31):
        super(Resnet50_Predictor, self).__init__()
        layers = []
        # currently 10000 units
        layers.append(nn.Dropout(p=0.5))
        layers.append(nn.Linear(2048, 1000))
        layers.append(nn.BatchNorm1d(1000, affine=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=0.5))
        layers.append(nn.Linear(1000, 1000))
        layers.append(nn.BatchNorm1d(1000, affine=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(1000, num_classes))
        self.class_classifier = nn.Sequential(*layers)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        x = self.class_classifier(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        print(x.size())

        return x


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model