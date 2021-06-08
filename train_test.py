import torch
import torch.nn as nn
from utils import discrepancy, CrossEntropyLabelSmooth, obtain_label, gradient_mathing_loss, discrepancy_matrix
import torch.nn.functional as F
import numpy as np


def train(args, src_data, tgt_data, tgt_data_no_shuff, extractor, classifier1, classifier2,opt_e,opt_c1,opt_c2, epoch):

    criterion = nn.CrossEntropyLoss()

    data = enumerate(iter(zip(src_data, tgt_data)))

    len_dataloader = min(len(src_data), len(tgt_data))
    # extractor.eval()
    # classifier1.eval()
    # classifier2.eval()
    # mem_label = obtain_label(tgt_data_no_shuff, extractor, classifier1, classifier2, args)
    # mem_label = torch.from_numpy(mem_label).cuda()

    extractor.train()
    classifier1.train()
    classifier2.train()
    for idx, ((src_img, src_labels,src_ind), (tgt_img, _, tgt_ind)) in data:
        if idx > len_dataloader:
            break

        src_img = src_img.to(args.device)
        #print(src_img.shape)
        tgt_img = tgt_img.to(args.device)
        src_labels = src_labels.to(args.device)
        #tgt_pred = mem_label[tgt_ind].to(args.device)

        #print(tgt_pred)

        data_all = torch.cat((src_img, tgt_img), 0)
        bs = args.batch_size

        '''
        STEP A
        '''
        reset_grad(opt_e,opt_c1,opt_c2)

        # src_feat = extractor(src_img)
        # tgt_feat = extractor(tgt_img)

        # preds_s1 = classifier1(src_feat)
        # preds_s2 = classifier2(src_feat)
        # preds_t1 = classifier1(tgt_feat)
        # preds_t2 = classifier2(tgt_feat)

        # p_mean_t1 = torch.mean(F.softmax(preds_t1), 0) + 1e-6
        # p_mean_t2 = torch.mean(F.softmax(preds_t2), 0) + 1e-6
        feat_all = extractor(data_all)
        output1 = classifier1(feat_all)
        output2 = classifier2(feat_all)
        preds_s1 = output1[:bs, :]
        preds_s2 = output2[:bs, :]
        preds_t1 = output1[bs:, :]
        preds_t2 = output2[bs:, :]

        p_soft_t1 = F.softmax(preds_t1,dim=1)
        p_soft_t2 = F.softmax(preds_t2,dim=1)

        
        # entropy_loss = - torch.sum( p_mean_t1 * torch.log(p_mean_t1))
        # entropy_loss -= torch.sum( p_mean_t2 * torch.log(p_mean_t2))
        entropy_loss = - torch.mean(torch.log(torch.mean(p_soft_t1, 0) + 1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(p_soft_t2, 0) + 1e-6))

        if epoch < 500:
            loss_A = criterion(preds_s1, src_labels) + criterion(preds_s2, src_labels) \
                + args.weight_ent * entropy_loss
        else:
            #print("tgt_pred:",preds_t1,"label:",tgt_pred)
            loss_A = criterion(preds_s1, src_labels) + criterion(preds_s2, src_labels) \
                + args.weight_cls * (criterion(preds_t1, tgt_pred) + criterion(preds_t2, tgt_pred)) \
                + args.weight_ent * entropy_loss

        loss_A.backward()


        opt_e.step()
        opt_c1.step()
        opt_c2.step()

        '''
        STEP B
        '''
        #print("step B")
        reset_grad(opt_e,opt_c1,opt_c2)

        # src_feat = extractor(src_img)
        # tgt_feat = extractor(tgt_img)

        # preds_s1 = classifier1(src_feat)
        # preds_s2 = classifier2(src_feat)
        # preds_t1 = classifier1(tgt_feat)
        # preds_t2 = classifier2(tgt_feat)

        # p_mean_t1 = torch.mean(F.softmax(preds_t1), 0) + 1e-6
        # p_mean_t2 = torch.mean(F.softmax(preds_t2), 0) + 1e-6
        feat_all = extractor(data_all)
        output1 = classifier1(feat_all)
        output2 = classifier2(feat_all)
        preds_s1 = output1[:bs, :]
        preds_s2 = output2[:bs, :]
        preds_t1 = output1[bs:, :]
        preds_t2 = output2[bs:, :]

        p_soft_t1 = F.softmax(preds_t1,dim=1)
        p_soft_t2 = F.softmax(preds_t2,dim=1)
        
        # entropy_loss = - torch.sum( p_mean_t1 * torch.log(p_mean_t1))
        # entropy_loss -= torch.sum( p_mean_t2 * torch.log(p_mean_t2))

        entropy_loss = - torch.mean(torch.log(torch.mean(p_soft_t1, 0) + 1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(p_soft_t2, 0) + 1e-6))


        loss_B = criterion(preds_s1, src_labels) + criterion(preds_s2, src_labels) \
            - args.weight_dis * discrepancy(preds_t1, preds_t2) \
            + args.weight_ent * entropy_loss

        loss_B.backward()

        opt_c1.step()
        opt_c2.step()

        reset_grad(opt_e,opt_c1,opt_c2)

        '''
        STEP C
        '''
        for i in range(args.N):
            # feat_src = extractor(src_img)
            # feat_tgt = extractor(tgt_img)
            # preds_s1 = classifier1(feat_src)
            # preds_s2 = classifier2(feat_src)
            # preds_t1 = classifier1(feat_tgt)
            # preds_t2 = classifier2(feat_tgt)

            # p_mean_t1 = torch.mean(F.softmax(preds_t1), 0) + 1e-6
            # p_mean_t2 = torch.mean(F.softmax(preds_t2), 0) + 1e-6
            feat_all = extractor(data_all)
            output1 = classifier1(feat_all)
            output2 = classifier2(feat_all)
            preds_s1 = output1[:bs, :]
            preds_s2 = output2[:bs, :]
            preds_t1 = output1[bs:, :]
            preds_t2 = output2[bs:, :]
            
            p_soft_t1 = F.softmax(preds_t1,dim=1)
            p_soft_t2 = F.softmax(preds_t2,dim=1)
        
            # entropy_loss = - torch.sum( p_mean_t1 * torch.log(p_mean_t1))
            # entropy_loss -= torch.sum( p_mean_t2 * torch.log(p_mean_t2))

            entropy_loss = - torch.mean(torch.log(torch.mean(p_soft_t1, 0) + 1e-6))
            entropy_loss -= torch.mean(torch.log(torch.mean(p_soft_t2, 0) + 1e-6))

            if epoch < 500 :
                loss_C = args.weight_dis * discrepancy(preds_t1, preds_t2) \
                    + args.weight_ent * entropy_loss
            else:
                loss_C = args.weight_dis * discrepancy(preds_t1, preds_t2) \
                + args.weight_gmn * gradient_mathing_loss(args, preds_s1,preds_s2, src_labels, preds_t1, preds_t2, tgt_pred, extractor, classifier1, classifier2)\
                + args.weight_ent * entropy_loss
            loss_C.backward()
            opt_e.step()

            reset_grad(opt_e,opt_c1,opt_c2)

        if (idx+1) % 10 == 0:
            print("loss_A = {:.2f}, loss_B = {:.2f}, loss_C = {:.2f}".format(loss_A.item(), loss_B.item(), loss_C.item()))
    return extractor, classifier1, classifier2


def test(args, data, extractor, classifier1, classifier2,epoch):
    acc1 = 0
    acc2 = 0
    extractor.eval()
    classifier1.eval()
    classifier2.eval()
    for img, labels, ind in data:
        img = img.to(args.device)
        labels = labels.to(args.device)
        preds1 = classifier1(extractor(img))
        preds2 = classifier2(extractor(img))

        acc1 += (preds1.argmax(dim=1) == labels).sum().item()
        acc2 += (preds2.argmax(dim=1) == labels).sum().item()

    print("acc1={:.2%}, acc2={:.2%}".format(acc1/len(data.dataset), acc2/len(data.dataset)))

def test_per_class(args, dataloader, extractor, classifier1, classifier2, epoch):
    extractor.eval()
    classifier1.eval()
    classifier2.eval()
    test_loss = 0
    correct_add = 0
    size = 0
    dset_classes = dataloader.dataset.classes
    classes_acc = {}
    for i in dset_classes:
        classes_acc[i] = []
        classes_acc[i].append(0)
        classes_acc[i].append(0)

    for img, labels, index in dataloader:
        img = img.to(args.device)
        labels = labels.to(args.device)
        output = extractor(img)
        output1 = classifier1(output)
        output2 = classifier2(output)
        test_loss += F.nll_loss(output1, labels).item()
        output_add = output1 + output2
        pred = output_add.data.max(1)[1]
        correct_add += pred.eq(labels.data).cpu().sum()
        size += labels.data.size()[0]
        for i in range(len(labels)):
            key_label = dset_classes[labels.long()[i].item()]
            key_pred = dset_classes[pred.long()[i].item()]
            classes_acc[key_label][1] += 1
            if key_pred == key_label:
                classes_acc[key_pred][0] += 1

    test_loss /= len(dataloader)  # loss function already averages over batch size
    print('Test Epoch: {:d}  Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
        epoch, test_loss, correct_add, size, 100. * float(correct_add) / size))
    avg = []
    for i in dset_classes:
        print('\t{}: [{}/{}] ({:.6f}%)'.format(i, classes_acc[i][0], classes_acc[i][1],
                                               100. * classes_acc[i][0] / classes_acc[i][1]))
        avg.append(100. * float(classes_acc[i][0]) / classes_acc[i][1])
    print('\taverage:', np.average(avg))
    for i in dset_classes:
        classes_acc[i][0] = 0
        classes_acc[i][1] = 0


def reset_grad(opt_e,opt_c1,opt_c2):
    opt_e.zero_grad()
    opt_c1.zero_grad()
    opt_c2.zero_grad()
    return







