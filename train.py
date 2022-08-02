# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

def train(epoch, net, optimizer, loss_function, cifar100_training_loader,  train_scheduler):

    start = time.time()
    net.train()
        
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        
        if args.tpu_core_num>0:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        
        train_scheduler.step()
        
        #n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        #last_layer = list(net.children())[-1]
        
        

        #print('Training Epoch: {epoch} [{trained_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        #    loss.item(),
        #    optimizer.param_groups[0]['lr'],
        #    epoch=epoch,
        #    trained_samples=batch_index * args.b + len(images)
        #))

        #update training loss for each iteration
        #if epoch <= args.warm:
        #    warmup_scheduler.step()

    #for name, param in net.named_parameters():
    #    layer, attr = os.path.splitext(name)
    #    attr = attr[1:]

    finish = time.time()

    return  loss, finish - start

@torch.no_grad()
def eval_training(epoch, net, cifar100_test_loader):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    l = 0.0
    for (images, labels) in cifar100_test_loader:
                
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        #loss = loss_function(outputs, labels)

        #test_loss += loss.item()
        
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()      
        l += labels.shape[0]
    
    correct_rate = correct/ l
    
    return correct_rate
    #if args.gpu:
    #    print('GPU INFO.....')
    #    print(torch.cuda.memory_summary(), end='')
    #print('Evaluating Network.....')
    #print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
    #    epoch,
    #    test_loss / len(cifar100_test_loader.dataset),
    #    correct.float() / len(cifar100_test_loader.dataset),
    #    finish - start
    #))
    #print()

    #add informations to tensorboard
    #if tb:
    #    writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    #    writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)

def run_wrapper(_):
    torch.manual_seed(1)
    np.random.seed(1)
    
    net = get_network(args)#!
    if args.tpu_core_num>0:
        device = xm.xla_device()
        net = xmp.MpModelWrapper(net)
        net=net.to(device)
    
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        tpu_core_num=args.tpu_core_num,
    )
    
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=False,
        tpu_core_num=args.tpu_core_num,
    )
    
    if args.tpu_core_num>0:
        pass    
    else:
        data_loader=cifar100_training_loader
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    #train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, settings.EPOCH*len(cifar100_training_loader))#!
    iter_per_epoch = len(cifar100_training_loader)
    #warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    train_scheduler.step(0)
    for epoch in range(1, settings.EPOCH + 1):
        if args.tpu_core_num>0:
            data_loader = pl.ParallelLoader(cifar100_training_loader, [device])
            data_loader = data_loader.per_device_loader(device)
        loss, time_use = train(epoch, net, optimizer, loss_function, data_loader, train_scheduler)
        if epoch%args.eval_every==0:
            if args.tpu_core_num>0:
                data_loader = pl.ParallelLoader(cifar100_test_loader, [device])
                data_loader = data_loader.per_device_loader(device)

            correct_rate = eval_training(epoch, net, data_loader)
            if args.tpu_core_num>0:
                def reduce_fn(vals):
                    return sum(vals) / len(vals)
                correct_rate_reduced = xm.mesh_reduce('correct_rate', correct_rate, reduce_fn)
                correct_rate=correct_rate_reduced
                xm.master_print(f"epoch={epoch}, test_acc={correct_rate}, time={time_use}, lr={        optimizer.param_groups[0]['lr']}")
            else:
                print(f"test_acc={correct_rate}, time={time_use}")
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-tpu_core_num', type=int, default=1, help='tpu_core_num')
    parser.add_argument('-eval_every', type=int, default=1, help='')

    args = parser.parse_args()
    
    if args.tpu_core_num>0:
        args.b=int(args.b/args.tpu_core_num)
    
    if args.tpu_core_num>0:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.utils.utils as xu
        torch.set_default_tensor_type('torch.FloatTensor')
        xmp.spawn(run_wrapper, args=(), nprocs=args.tpu_core_num, start_method='fork')
    else:
        run_wrapper(0)
        
        
def old():        
    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    #writer.close()
