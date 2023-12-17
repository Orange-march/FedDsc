from copy import deepcopy
import torch
from torch.backends import cudnn
from torchvision import models as models
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

import Gen_noniid_public
import user_model
import os
import time
import socket
import torch.nn.functional as F
import Gen_noniid
import my_resnet
import random

from losses import SupConLoss
from resnet_big import SupConResNet, LinearClassifier
from util import set_optimizer, adjust_learning_rate
import util


class Arguments():
    def __init__(self):
        self.model_name = 'resnet18'
        self.batch_size = 128
        self.test_batch_size = 1000
        self.epochs = 10
        self.round = 300
        self.lr = 0.01
        self.momentum = 0.9
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 20
        self.save_model = False
        self.path = r'./server'
        self.best_acc = 0
        self.global_model = {}
        self.agg_param = {}  #?
        self.user_list = [i for i in range(10)]
        self.sample_cnt = 3
        self.mu = 5
        self.t = 0.5

def init_local_model(args, model, device):
    for k,v in model.named_parameters():
        v.data = args.global_model[k]

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr ,weight_decay=5e-4 )

    return optimizer

def get_dis_loss(args, features, images, device, gt_model,bsz):

    gt_model = gt_model.to(device)
    g_features = gt_model(images)
    f1, f2 = torch.split(g_features, [bsz, bsz], dim=0)
    g_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    dis_loss = torch.sub(g_features,features,alpha = 1)

    dis_loss = torch.norm(dis_loss)

    return dis_loss



def train(args,device,model,criterion,train_loader,optimizer,epoch,gt_model,round):
    model.train()

    for batch_idx, (images,labels) in enumerate(train_loader):
        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        cl_loss = criterion(features, labels)
        dis_loss = get_dis_loss(args, features, images, device, gt_model,bsz)


        loss = cl_loss + 0.01 * dis_loss

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def save_param(args, model):
    for k, v in model.named_parameters():
        args.agg_param[k] = args.agg_param[k] + v.data

def agg(args):
    for k in args.global_model:
        args.global_model[k] = args.agg_param[k] / args.sample_cnt
        args.agg_param[k] = 0

def global_model_update(args, model):
    for k, v in model.named_parameters():
        v.data = args.global_model[k]

def linear_train(train_loader, model, classifier, criterion,optimizer, epoch):
    model.eval()
    classifier.train()

    top1 = util.AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric

        acc1, acc5 = util.accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return top1.avg

def validate(val_loader, model, classifier, criterion):
    """validation"""
    model.eval()
    classifier.eval()

    top1 = util.AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))

            # update metric
            acc1, acc5 = util.accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def linear_test(args, device, model, test_loader, round, userID, epoch,pb_train_loaders):
    best_acc = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=args.model_name, num_classes=10)
    model = model.cuda()
    classifier = classifier.cuda()
    criterion = criterion.cuda()
    optimizer = set_optimizer(classifier)

    for epoch in range(1, 11):

        # train for one epoch
        time1 = time.time()
        acc = linear_train(pb_train_loaders, model, classifier, criterion,optimizer, epoch)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        val_acc = validate(test_loader, model, classifier, criterion)
        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy: {:.2f}'.format(best_acc))
   # with open('./FedCL.txt', 'a') as f:
     #   f.write('{}%\n'.format(best_acc))



def train_main(train_loaders,test_loader,args,model_list,device,criterion,pb_train_loaders,gt_model):
    for round in range(args.round):
        ls = random.sample(args.user_list,args.sample_cnt)
        for i in ls:
            optimizer = init_local_model(args,model_list[i],device)
            for epoch in range(1,args.epochs+1):
                gt_model.load_state_dict(model_list[i].state_dict())
                train(args, device, model_list[i], criterion, train_loaders[i], optimizer, epoch,gt_model,round)
                print("Round:{} i:{} epoch: {}".format(round, i, epoch))
            save_param(args,model_list[i])
        agg(args)
        global_model_update(args,model_list[0])
        linear_test(args,device,model_list[0],test_loader,round,i,epoch,pb_train_loaders)



def main():
    t_start = time.time()

    args = Arguments()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    device = torch.device("cuda" if use_cuda else "cpu")

    model_list = [SupConResNet(name=args.model_name) for _ in range(10)]

    gt_model = SupConResNet(name=args.model_name)

    for k,v in model_list[0].named_parameters():
        args.global_model[k] = deepcopy(v.data)
        args.agg_param[k] = 0

    criterion = SupConLoss()

    pb_train_loaders , train_loaders , test_loaders = Gen_noniid_public.get_user_dataset()

    train_main(train_loaders,test_loaders,args,model_list,device,criterion , pb_train_loaders,gt_model)


    t_end = time.time()

    print('Train cost:{}'.format(t_end-t_start))

    print("best accuracy:{:.1f}".format(args.best_acc))





if __name__ == '__main__':
    main()
