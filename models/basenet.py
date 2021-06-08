from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output * -self.lambd


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert (padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResBottle(nn.Module):
    def __init__(self, option='resnet18', pret=True):
        super(ResBottle, self).__init__()
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        mod = list(model_ft.children())
        mod.pop()
        # self.model_ft =model_ft
        self.features = nn.Sequential(*mod)

        self.bottleneck = nn.Linear(model_ft.fc.in_features, 256)
        nn.init.normal_(self.bottleneck.weight.data, 0, 0.005)
        nn.init.constant_(self.bottleneck.bias.data, 0.1)

        self.dim = 256

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.bottleneck(x)

        x = x.view(x.size(0), self.dim)
        return x

    def output_num(self):
        return self.dim


class ResNet_all(nn.Module):
    def __init__(self, option='resnet18', pret=True):
        super(ResNet_all, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        # mod = list(model_ft.children())
        # mod.pop()
        # self.model_ft =model_ft
        self.conv1 = model_ft.conv1
        self.bn0 = model_ft.bn1
        self.relu = model_ft.relu
        self.maxpool = model_ft.maxpool
        self.layer1 = model_ft.layer1
        self.layer2 = model_ft.layer2
        self.layer3 = model_ft.layer3
        self.layer4 = model_ft.layer4
        self.pool = model_ft.avgpool
        self.fc = nn.Linear(2048, 12)

    def forward(self, x, layer_return=False, input_mask=False, mask=None, mask2=None):
        if input_mask:
            x = self.conv1(x)
            x = self.bn0(x)
            x = self.relu(x)
            conv_x = x
            x = self.maxpool(x)
            fm1 = mask * self.layer1(x)
            fm2 = mask2 * self.layer2(fm1)
            fm3 = self.layer3(fm2)
            fm4 = self.pool(self.layer4(fm3))
            x = fm4.view(fm4.size(0), self.dim)
            x = self.fc(x)
            return x  # ,fm1
        else:
            x = self.conv1(x)
            x = self.bn0(x)
            x = self.relu(x)
            conv_x = x
            x = self.maxpool(x)
            fm1 = self.layer1(x)
            fm2 = self.layer2(fm1)
            fm3 = self.layer3(fm2)
            fm4 = self.pool(self.layer4(fm3))
            x = fm4.view(fm4.size(0), self.dim)
            x = self.fc(x)
            if layer_return:
                return x, fm1, fm2
            else:
                return x


class ResClassifier(nn.Module):
    def __init__(self, num_classes=12, num_layer=2, num_unit=2048, prob=0.5, middle=1000):
        super(ResClassifier, self).__init__()
        layers = []
        # currently 10000 units
        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(num_unit, middle))
        layers.append(nn.BatchNorm1d(middle, affine=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(num_layer - 1):
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle, middle))
            layers.append(nn.BatchNorm1d(middle, affine=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(middle, num_classes))
        self.classifier = nn.Sequential(*layers)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x
