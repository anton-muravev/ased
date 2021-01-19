# -*- coding: utf-8 -*-
## BOUNDED PROBABILITY EXPERIMENT ON CIFAR10

import utorch
import ased
import ased_util
import time
import numpy as np
from astore import Astore
from sklearn.model_selection import StratifiedShuffleSplit
import argparse

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

parser = argparse.ArgumentParser(description="Run the ASED search with the given settings and probability bounding")
parse_group = parser.add_mutually_exclusive_group()
parser.add_argument('--cifarpath', required=True, help="path to CIFAR100 dataset")
parser.add_argument('--init', required=True, help="path to init file, including filename")
parser.add_argument('--out', required=True, help="prefix for output files")
parse_group.add_argument('--dense', type=int, 
                         help="enable dense shortcut pattern with given value")
parse_group.add_argument('--residual', type=int, 
                         help="enable residual shortcut pattern with given value")
parser.add_argument('--iter', type=int, default=9,
                    help="number of search iterations to run")
parser.add_argument('--bound', type=float, default=0.9,
                    help="the upper bound for probability")
parser.add_argument('--gpus', type=int, default=4, 
                    help="number of GPU devices to use")
parser.add_argument('--netcount', type=int, default=250,
                    help="networks to sample per GPU")
parser.add_argument('--workers', type=int, default=8,
                    help="number of data loading CPU workers per GPU")
parser.add_argument('--resume', type=int, default=-1,
                    help="from which iteration to continue, omit to start from scratch")
args = parser.parse_args()

cifarpath = args.cifarpath
init_path = args.init
out_prefix = args.out
gpu_count = args.gpus
netcount = args.netcount
big_iterations = args.iter
shortcut = 'none'
shortcut_value = 2
if args.dense:
    shortcut = 'dense'
    shortcut_value = args.dense
if args.residual:
    shortcut = 'residual'
    shortcut_value = args.residual
prob_bound = args.bound
base_lr = 0.01
momentum = 0.9
workers = args.workers
epochs = 20
batch_size = 128
class_count = 10
top_slice = 1
cudnn.benchmark = False
resume = args.resume

normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])

train_dataset = datasets.CIFAR10(cifarpath, train=True,
                                  transform=transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize]))

test_dataset = datasets.CIFAR10(cifarpath, train=False,
                  transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize]))

def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_layer_schedule(iteration):
    if iteration<2:
        return 2
    else:
        return 1
        
opLibrary = ased.get_default_library()

def gpu_proc(gpu_id, it, prototype, netcount, train_idx, val_idx, seed):
    pars = ['perf', 'runtime', 'cfmat', 'matthews', 'loss', 'params', 
            'phenotypes']
    fname = "./data/"+out_prefix+"_iter"+str(it)+"_gpu"+str(gpu_id)+".pickle"
    store = Astore()
    for p in pars:
        store[p] = []
    np.random.seed(seed+gpu_id)
    torch.manual_seed(seed+gpu_id)
    torch.cuda.set_device(gpu_id)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idx),
        num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(val_idx),
        num_workers=workers, pin_memory=True)
    
    for n in range(netcount):
        if n % 50 == 0:
            print("GPU "+str(gpu_id)+" processed "+str(n)+" networks")
        net = ased.generate_shortcut_feature_network(3, 32, 32, prototype,
                                                     opLibrary, 
                                                     shortcuts=shortcut,
                                                     skip_value=shortcut_value)
        store['phenotypes'].append(ased.network_to_phenotype(net))
        evnet = ased.EvalNet2(net, class_count).cuda()
        crit = nn.CrossEntropyLoss().cuda()
        lr = base_lr
        optim = torch.optim.SGD(evnet.parameters(), lr,
                                momentum=momentum)
        start = time.time()
        for epoch in range(0, epochs):
            lr = adjust_learning_rate(optim, epoch, lr)
            utorch.train1epoch(train_loader, evnet, crit, optim, epoch,
                               verbose=False)
        store['runtime'].append(time.time()-start)
        acc, loss, cfmat = utorch.validate_cfmat(val_loader, evnet, crit, 
                                                  class_count, verbose=False)
        store['loss'].append(loss)
        store['cfmat'].append(cfmat)
        store['matthews'].append(ased_util.multiclass_matthews(cfmat))
        store['perf'].append(acc)
        store['params'].append(utorch.count_parameters(evnet))
        
        store.dump(fname)
                
        del evnet
        del crit
        del optim
        
    store.dump(fname)
    print("GPU "+str(gpu_id)+" finished generation "+str(it))
        
if __name__ == '__main__':

    base_seed = 3051991
    np.random.seed(base_seed)    
    torch.manual_seed(base_seed)
    pars = ['perf', 'runtime', 'cfmat', 'matthews', 'loss', 'params', 
            'phenotypes']
    mainstore = Astore()
    
    splitter = StratifiedShuffleSplit(n_splits=big_iterations, test_size=0.2)
    tr = splitter.split(np.zeros((50000,1)), train_dataset.targets)

    if resume == -1:
        init_store = Astore()
        init_store.load(init_path)
        topnets = np.argsort(init_store['matthews'])[-top_slice:][::-1]
        topbinaries = [ased.phenotype_to_binary(init_store['phenotypes'][i])
                        for i in topnets]
        prototype = np.stack(topbinaries, axis=-1).mean(axis=-1)
        prototype = ased.normalize_prototype(prototype, len(opLibrary),
                                             prob_bound)
        del init_store
    else:
        print("Resuming the process from iteration "+str(resume))
        sname = "./data/"+out_prefix+"_iter"+str(resume)+"_cumul.pickle"
        mainstore.load(sname)
        topnets = np.argsort(mainstore['matthews'])[-top_slice:][::-1]
        topbinaries = [ased.phenotype_to_binary(mainstore['phenotypes'][i])
                        for i in topnets]
        prototype = np.stack(topbinaries, axis=-1).mean(axis=-1)
        prototype = ased.normalize_prototype(prototype, len(opLibrary), 
                                             prob_bound)

    for it in np.arange(resume+1,big_iterations):
        
        print("Starting evo generation "+str(it))
        fname = "./data/"+out_prefix+"_iter"+str(it)+"_cumul.pickle"
        for p in pars:
            mainstore[p] = []
        add_layer = get_layer_schedule(it)
        prototype = np.vstack([prototype, 
                               ased.get_uniform_prototype(add_layer, opLibrary)])
        train_idx, val_idx = next(tr)
        
        processes = []
        for r in range(gpu_count):
            p = mp.Process(target=gpu_proc, args=(r,it, prototype, netcount, 
                                                  train_idx, val_idx, base_seed))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        smallstore = Astore()
        for r in range(gpu_count):
            sname = "./data/"+out_prefix+"_iter"+str(it)+"_gpu"+str(r)+".pickle"
            smallstore.load(sname)
            for v in pars:
                mainstore[v].extend(smallstore[v])
        mainstore.dump(fname)

        topnets = np.argsort(mainstore['matthews'])[-top_slice:][::-1]
        topbinaries = [ased.phenotype_to_binary(mainstore['phenotypes'][i]) 
                        for i in topnets]
        prototype = np.stack(topbinaries, axis=-1).mean(axis=-1)
        prototype = ased.normalize_prototype(prototype, len(opLibrary),
                                             prob_bound)