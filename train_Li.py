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

import sys
sys.path.insert(0, '../InstaAug/')
from InstaAug_module import learnable_invariance
import yaml

import torch_xla.debug.metrics as met

def train(epoch, net, optimizer, loss_function, cifar100_training_loader,  train_scheduler, Li=None, Li_configs={}):

    start = time.time()
    net.train()
   
    loss_sum=0
    sample_sum=0
    entropy_sum=0
    
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            
        if Li:
            optimizer_Li=Li.optimizer
            train_scheduler_Li=Li.scheduler
            images_Li, logprob, entropy_every=Li(images)
            
            outputs = net(images_Li)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss_predictor = loss_function(outputs, labels)
            loss_Li_pre=(loss_predictor.detach()*logprob).mean()+loss_predictor.mean()
            entropy=entropy_every.mean(dim=0)
            
            r=min(1, epoch/Li_configs['entropy_increase_period'])
            mid_target_entropy=Li.target_entropy*r+Li_configs['start_entropy']*(1-r)
            loss=loss_Li_pre+(entropy.mean()-mid_target_entropy)**2*0.3#!#!#!#!
            
            loss.backward()
            
            
            if args.tpu_core_num>0:
                xm.optimizer_step(optimizer)
                xm.optimizer_step(optimizer_Li)
            else:
                optimizer.step()
                optimizer_Li.step()
            
            train_scheduler.step()
            train_scheduler_Li.step()
            
            with torch.no_grad():
                entropy_sum=entropy_sum+entropy_every.sum()
                loss_sum=loss_sum+loss_predictor.sum()
                sample_sum=sample_sum+labels.shape[0]
            
        else:
            outputs = net(images)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)            
            loss.backward()
        
            if args.tpu_core_num>0:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
        
            train_scheduler.step()
            
            with torch.no_grad():
                loss_sum=loss_sum+loss.sum()
                sample_sum=sample_sum+labels.shape[0]
    
    loss_ave=loss_sum/sample_sum
    entropy_ave=entropy_sum/sample_sum
    finish = time.time()

    return  loss_ave, entropy_ave, finish - start

@torch.no_grad()
def eval_training(epoch, net, cifar100_test_loader, Li=None, Li_configs={}):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    l = 0.0
    
    
    for (images, labels) in cifar100_test_loader:
                
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        if Li and Li_configs['test_time_aug']:
            n_copies=Li_configs['test_copies']
            images_Li, logprob, entropy_every=Li(images, n_copies=n_copies)
            outputs = net(images_Li) 
            
            bs=outputs.shape[0]
            logit=F.log_softmax(outputs[:n_copies*bs])
            logit=logit.reshape([n_copies, bs, -1]).transpose(0,1)
            logprob=logprob.reshape([n_copies, -1]).transpose(0,1).unsqueeze(-1)
            outputs=torch.sum(torch.exp(logit)*torch.exp(logprob_new*0.0), dim=1)#? this is prob
            
            
        else:
            outputs = net(images)
        
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()      
        l += labels.shape[0]
        

    
    correct_rate = correct/ l
    
    return correct_rate

def reduce_fn(vals):
    return sum(vals) / len(vals)

def run_wrapper(_, Li_config, args):
    torch.manual_seed(124)
    acc_memory=0.0
    
    if args.tpu_core_num>0:
        device = xm.xla_device()
        save_op=xm
    elif args.gpu:
        device = 'cuda'
        save_op=torch
    else:
        device='cpu'
        save_op=torch
    
    net = get_network(args)#!
    
    if Li_configs['li_flag']:
        Li=learnable_invariance(Li_configs, device=device)  
    else:
        Li=None
    
    if args.tpu_core_num>0:
        
        net = xmp.MpModelWrapper(net)
        net=net.to(device)
        if Li_configs['li_flag']:
            Li = xmp.MpModelWrapper(Li)
            Li=Li.to(device)
    
    
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=1,
        batch_size=args.b,
        shuffle=True,
        tpu_core_num=args.tpu_core_num,
        args=args,
    )
    
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=1,
        batch_size=args.b,
        shuffle=True,
        tpu_core_num=args.tpu_core_num,
    )
            
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    iter_per_epoch = len(cifar100_training_loader)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, settings.EPOCH*iter_per_epoch)
    
    if Li_configs['li_flag']:
        optimizer_Li=optim.SGD(Li.parameters(), lr=Li_configs['lr'])
        train_scheduler_Li = optim.lr_scheduler.CosineAnnealingLR(optimizer_Li, settings.EPOCH*iter_per_epoch)
        Li.optimizer=optimizer_Li
        Li.scheduler=train_scheduler_Li
        Li.target_entropy=args.target_entropy
    
    train_scheduler.step(0)#?
    
    for epoch in range(1, settings.EPOCH + 1):
        if args.tpu_core_num>0:
            data_loader = pl.ParallelLoader(cifar100_training_loader, [device])
            data_loader = data_loader.per_device_loader(device)
        else:
            data_loader=cifar100_training_loader
        train_loss, train_entropy, time_use = train(epoch, net, optimizer, loss_function, data_loader, train_scheduler, Li=Li, Li_configs=Li_configs)
        
        
        if epoch%args.eval_every==0:
            if args.tpu_core_num>0:
                data_loader = pl.ParallelLoader(cifar100_test_loader, [device])
                data_loader = data_loader.per_device_loader(device)
            else:
                data_loader=cifar100_test_loader
            correct_rate = eval_training(epoch, net, data_loader, Li=Li, Li_configs=Li_configs)
            if args.tpu_core_num>0:
                train_loss_reduced = xm.mesh_reduce('train_loss', train_loss, reduce_fn)
                train_entropy_reduced = xm.mesh_reduce('train_entropy', train_entropy, reduce_fn)
                correct_rate_reduced = xm.mesh_reduce('correct_rate', correct_rate, reduce_fn)
                correct_rate=correct_rate_reduced
                xm.master_print(f"epoch={epoch}, train_loss={train_loss_reduced}, train_entropy={train_entropy_reduced}, test_acc={correct_rate}, time={time_use}, lr={optimizer.param_groups[0]['lr']}")
            else:
                print(f"test_acc={correct_rate}, time={time_use}")
            
            
            if correct_rate>acc_memory:
                acc_memory=correct_rate
                if args.net_save_path:
                    save_op.save(net.state_dict(), args.net_save_path+str(epoch))
                if Li and args.Li_save_path:
                    save_op.save(Li.augmentation.get_param.conv.state_dict(), args.Li_save_path+str(epoch))
            

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
    parser.add_argument('-Li_config_path', type=str, default='', help='')
    parser.add_argument('-entropy_weights', type=float, default=0.0, help='')
    parser.add_argument('-target_entropy', type=float, default=3.0, help='log(17)=2.83')
    parser.add_argument('-net_save_path', type=str, default='', help='')
    parser.add_argument('-Li_save_path', type=str, default='', help='')
    parser.add_argument('-random_crop_method', type=str, default='mirror_pad_then_crop', help='no_crop/mirror_pad_then_crop/black_pad_then_crop/random_resized_crop')
    parser.add_argument('-randomresizecrop_min_ratio', type=float, default=0.08, help='')
    parser.add_argument('-randomcrop_padding', type=int, default=4, help='')
    
    args = parser.parse_args()
    
    if args.Li_config_path:
        import yaml
        Li_configs=yaml.safe_load(open(args.Li_config_path,'r'))
        if args.entropy_weights:
            Li_configs['entropy_weights']=args.entropy_weights
        args.random_crop_method='no_crop'
  
    else:
        Li_configs={'li_flag': False}
    
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
        xmp.spawn(run_wrapper, args=(Li_configs, args), nprocs=args.tpu_core_num, start_method='fork')
    else:
        run_wrapper('', Li_configs, args)
        
        
