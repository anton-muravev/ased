# -*- coding: utf-8 -*-
## VALIDATION ON CIFAR100

import utorch
import ased
import numpy as np
from astore import Astore
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

parser = argparse.ArgumentParser(description="Validate ASED results from the file")
parse_group = parser.add_mutually_exclusive_group()
parser.add_argument('--cifarpath', required=True, help="path to CIFAR100 dataset")
parser.add_argument('--input', required=True, help="path to the generation file")
parser.add_argument('--epochs', type=int, default=200, 
                    help="how many epochs to train for")
parser.add_argument('--channels', type=int, default=32, 
                    help="how many channels to use")
parser.add_argument('--nextgen', action='store_true', 
                    help="compute the next prototype and use that for validation")
parse_group.add_argument('--dense', type=int, 
                         help="enable dense shortcut pattern with given value")
parse_group.add_argument('--residual', type=int, 
                         help="enable residual shortcut pattern with given value")
parser.add_argument('--workers', type=int, default=8,
                    help="number of data loading CPU workers per GPU")
args = parser.parse_args()

cifarpath = args.cifarpath
in_path = args.input
shortcut = 'none'
shortcut_value = 2
if args.dense:
    shortcut = 'dense'
    shortcut_value = args.dense
if args.residual:
    shortcut = 'residual'
    shortcut_value = args.residual
lr = 0.01
momentum = 0.9
workers = args.workers
epochs = args.epochs
channels = args.channels
batch_size = 128
class_count = 100
cudnn.benchmark = False

normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                 std=[0.247, 0.243, 0.262])

train_dataset = datasets.CIFAR100(cifarpath, train=True,
                                  transform=transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize]))

test_dataset = datasets.CIFAR100(cifarpath, train=False,
                  transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize]))
    
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=workers,
                                         pin_memory=True)

def adjust_learning_rate(optimizer, epoch, lr):
    lr = {       epoch < 60 : 0.01,
          60  <= epoch < 120: 0.002,
          120 <= epoch < 160: 0.0004,
          160 <= epoch      : 0.0001
         }[True]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
        
opLibrary = ased.get_default_library()

# load the search output and construct the network
store = Astore()
store.load(in_path)
if args.nextgen:
    topnets = np.argsort(store['matthews'])[-100:][::-1]
    topbinaries = [ased.phenotype_to_binary(store['phenotypes'][i]) 
                    for i in topnets]
    prototype = np.stack(topbinaries, axis=-1).mean(axis=-1)
else:
    prototype = store['phenotypes'][0]['prototype']
net = ased.generate_shortcut_feature_network(3, channels, 32, prototype, 
                                             opLibrary, shortcuts=shortcut, 
                                             skip_value=shortcut_value,
                                             batch_norms=True, nonrandom=True)
evnet = ased.EvalNet2(net, class_count).cuda()
print("Parameters: "+str(utorch.count_parameters(evnet)))
# train the validation network
decayed_weights = []
prelu_weights = []
for m in evnet.modules():
    if isinstance(m, nn.PReLU):
        prelu_weights.append(m.weight)
    else:
        if hasattr(m, 'weight') and m.weight is not None:
            decayed_weights.append(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            decayed_weights.append(m.bias)
        
par_dict = [{'params': decayed_weights},
            {'params': prelu_weights, 'weight_decay': 0}]
optim = torch.optim.SGD(par_dict, lr, momentum=momentum, 
                        weight_decay=1e-4)
crit = nn.CrossEntropyLoss().cuda()
for epoch in range(0, epochs):
    lr = adjust_learning_rate(optim, epoch, lr)
    utorch.train1epoch(train_loader, evnet, crit, optim, epoch)
    utorch.validate_cfmat(val_loader, evnet, crit, class_count)
    with torch.no_grad():
        for name, layer in evnet.named_modules():
            if isinstance(layer, nn.Conv2d):
                norms = torch.sqrt(torch.sum(layer.weight.data**2, 
                                             dim=[1,2,3], keepdim=True))
                desired = torch.clamp(norms, max=0.5)
                layer.weight *= (desired / (norms + 1e-7))