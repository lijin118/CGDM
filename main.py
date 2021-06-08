from multiprocessing.spawn import freeze_support
import torch
from models.network import *
from train_test import train, test, test_per_class
from utils import *
import torchvision
import os
import params as args
import torch.nn as nn

if args.category == 'digit':
    extractor, classifier1, classifier2 = load_LeNet()
    data  = digit_load(args)
elif args.category == 'office31':
    extractor, classifier1, classifier2 = load_pretrain_resnet(args, num_class = args.class_num)
    data = office31_load(args)
else:
    raise RuntimeError('Dataset not found.' +
                        ' You should choose either "digit" or "office31"')
classifier1.apply(init_weights_xavier_normal)
classifier2.apply(init_weights_xavier_normal)

extractor.to(args.device)
classifier1.to(args.device)
classifier2.to(args.device)

src_train_data = data["source_train"]

tgt_train_data = data["target_train"]

tgt_train_data_no_shuff = data["target_train_no_shuff"]

src_test_data = data["source_test"]

tgt_test_data = data["target_test"]

if args.optimizer == 'adam':
    l2_decay = 5e-4
    momentum = 0.9
    opt_e = torch.optim.Adam(extractor.features.parameters(), lr=args.lr,weight_decay=l2_decay)
    opt_c1 = torch.optim.Adam(classifier1.parameters(), lr=args.lr,weight_decay=l2_decay)
    opt_c2 = torch.optim.Adam(classifier2.parameters(), lr=args.lr,weight_decay=l2_decay)
elif args.optimizer == 'sgd':
    l2_decay = 5e-4
    momentum = 0.9
    opt_e = torch.optim.SGD(extractor.parameters(), lr=args.lr, momentum=momentum, weight_decay=l2_decay)
    opt_c1 = torch.optim.SGD(classifier1.parameters(), lr=args.lr, momentum=momentum, weight_decay=l2_decay)
    opt_c2 = torch.optim.SGD(classifier2.parameters(), lr=args.lr, momentum=momentum, weight_decay=l2_decay)
else:
    raise RuntimeError('Optimizer not found.' +
                            ' You should choose either "adam" or "sgd"')


if __name__ == '__main__':
    print(args)
    #freeze_support()

    for i in range(args.num_epoch):
        print('-' * 100, '\nTraining  Epoch',i)

        extractor, classifier1, classifier2 = train(args,src_train_data, tgt_train_data,tgt_train_data_no_shuff, extractor, classifier1, classifier2,opt_e,opt_c1,opt_c2,i)

        torch.save(extractor, os.path.join(args.models_save, "extractor.pth"))
        torch.save(classifier1, os.path.join(args.models_save, "classifier1.pth"))
        torch.save(classifier2, os.path.join(args.models_save, "classifier2.pth"))

        # 源域测试
        print('Testing source')
        test_per_class(args, src_test_data, extractor, classifier1, classifier2, i)
        # 目标域测试
        print('Testing target')
        test_per_class(args, tgt_test_data, extractor, classifier1, classifier2, i)




