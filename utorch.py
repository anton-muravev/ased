#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:10:48 2018

@author: muravev
"""

import time
import pickle

from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

class AverageValueMeter:
    def __init__(self):
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean, self.std = self.sum, np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
        
class ConfusionMeter:
    def __init__(self, k, normalized=False):
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))
        self.conf += conf

    def value(self):
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf

def train1epoch(train_loader, model, criterion, optimizer, epoch, 
                print_freq=None, verbose=True):
    """Train a given neural network for one epoch.
    
    Arguments
    ----------
    train_loader : pyTorch DataLoader instance
        An instance of DataLoader class that supplies the training data.
    model: subclass instance of pyTorch nn.Module
        The class containing the network model to train, as per convention.
    criterion: pyTorch Loss instance
        The instance of pyTorch Loss class, defining the value to be optimized.
    optimizer: pyTorch Optimizer instance
        The instance of pyTorch Optimizer, defining the algorithm used.
    epoch: int
        The current epoch number, only for printing purposes.
    print_freq: None (default) or int
        Defines how many batches are processed between printing the summary of
        the training process to the console.
    verbose: bool (default True)
        Defines whether the training process prints progress to the console.
    
    Returns
    ----------
    Tuple of (accuracy, loss).
    
    """
    batch_time = AverageValueMeter()
    data_time = AverageValueMeter()
    losses = AverageValueMeter()
    top1 = AverageValueMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.add(time.time() - end)

        inputs = inputs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.add(loss.item())
        top1.add(prec1[0][0].cpu().numpy())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.add(time.time() - end)
        end = time.time()

        if not verbose:
            continue
        if (print_freq is not None) and (i % print_freq == 0):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.mean:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.mean:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.mean:.4f})\t'
                  'Acc {top1.val:.3f} ({top1.mean:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time,
                   loss=losses,
                   top1=top1))
            
    if verbose:
        print('Epoch [{0}] training: * Loss {losses.mean:.3f} Accuracy {top1.mean:.3f}'
          .format(epoch, top1=top1, losses=losses))
    
    return top1.mean, losses.mean

def train1epoch_cfmat(train_loader, model, criterion, optimizer, numclasses,
                      epoch, print_freq=None, verbose=True):
    """Train a given neural network for one epoch, while maintaining a
    confusion matrix.
    
    Arguments
    ----------
    train_loader : pyTorch DataLoader instance
        An instance of DataLoader class that supplies the training data.
    model: subclass instance of pyTorch nn.Module
        The class containing the network model to train, as per convention.
    criterion: pyTorch Loss instance
        The instance of pyTorch Loss class, defining the value to be optimized.
    optimizer: pyTorch Optimizer instance
        The instance of pyTorch Optimizer, defining the algorithm used.
    epoch: int
        The current epoch number, only for printing purposes.
    print_freq: None (default) or int
        Defines how many batches are processed between printing the summary of
        the training process to the console.
    verbose: bool (default True)
        Defines whether the training process prints progress to the console.
    
    Returns
    ----------
    Tuple of (accuracy, loss, confusion matrix).
    
    """
    batch_time = AverageValueMeter()
    data_time = AverageValueMeter()
    losses = AverageValueMeter()
    top1 = AverageValueMeter()
    cfmat = ConfusionMeter(numclasses)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.add(time.time() - end)

        inputs = inputs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.add(loss.item())
        top1.add(prec1[0][0].cpu().numpy())
        cfmat.add(output.data.cpu().numpy(), target.cpu().numpy())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.add(time.time() - end)
        end = time.time()

        if not verbose:
            continue
        if (print_freq is not None) and (i % print_freq == 0):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.mean:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.mean:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.mean:.4f})\t'
                  'Acc {top1.val:.3f} ({top1.mean:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time,
                   loss=losses,
                   top1=top1))
            
    if verbose:
        print('Epoch [{0}] training: * Loss {losses.mean:.3f} Accuracy {top1.mean:.3f}'
          .format(epoch, top1=top1, losses=losses))
        
    return top1.mean, losses.mean, cfmat.value()
            
def validate(val_loader, model, criterion, verbose=True):
    """Validate the neural network's performance on the given data.
    
    Arguments
    ----------
    val_loader : pyTorch DataLoader instance
        An instance of DataLoader class that supplies the validation data.
    model: subclass instance of pyTorch nn.Module
        The class containing the network model to evaluate, as per convention.
    criterion: pyTorch Loss instance
        The instance of pyTorch Loss class, defining the value to be measured.
    verbose: bool (default True)
        Defines whether the validation process prints progress to the console.
    
    Returns
    ----------
    Tuple of (accuracy, loss).
    
    """
    batch_time = AverageValueMeter()
    losses = AverageValueMeter()
    top1 = AverageValueMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.add(loss.item())
        top1.add(prec1[0].item())

        # measure elapsed time
        batch_time.add(time.time() - end)
        end = time.time()

    if verbose:
        print('Validation: * Loss {losses.mean:.3f} Accuracy {top1.mean:.3f}'
          .format(top1=top1, losses=losses))

    return top1.mean, losses.mean

def validate_cfmat(val_loader, model, criterion, numclasses, verbose=True):
    """Validate the neural network's performance on the given data and compute 
    the confusion matrix.
    
    Arguments
    ----------
    val_loader : pyTorch DataLoader instance
        An instance of DataLoader class that supplies the validation data.
    model: subclass instance of pyTorch nn.Module
        The class containing the network model to evaluate, as per convention.
    criterion: pyTorch Loss instance
        The instance of pyTorch Loss class, defining the value to be measured.
    numclasses: int
        The number of classes in the data (for the confusion matrix creation).
    verbose: bool (default True)
        Defines whether the validation process prints progress to the console.
    
    Returns
    ----------
    Tuple of (accuracy, loss, confusion matrix).
    
    """
    batch_time = AverageValueMeter()
    losses = AverageValueMeter()
    top1 = AverageValueMeter()
    cfmat = ConfusionMeter(numclasses)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.add(loss.item())
        top1.add(prec1[0].item())
        cfmat.add(output.data.cpu().numpy(), target.cpu().numpy())

        # measure elapsed time
        batch_time.add(time.time() - end)
        end = time.time()

    if verbose:
        print('Validation: * Loss {losses.mean:.3f} Accuracy {top1.mean:.3f}'
          .format(top1=top1, losses=losses))

    return top1.mean, losses.mean, cfmat.value()

def predict_raw(loader, model):
    """Compute the raw output of the neural network model for the given data.
    
    Arguments
    ----------
    loader : pyTorch DataLoader instance
        An instance of DataLoader class that supplies the data.
    model: subclass instance of pyTorch nn.Module
        The class containing the network model to evaluate, as per convention.
    
    Returns
    ----------
    The network output tensor.
    
    """
    model.eval()
    out = []
    for i, (input, target) in enumerate(loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)

        # compute output
        output = model(input_var)
        out.append(output.data)
    return out

def predict_probs(loader, model):
    """Compute the softmax-processed predictions of the neural network model 
    on the given data.
    
    Arguments
    ----------
    loader : pyTorch DataLoader instance
        An instance of DataLoader class that supplies the data.
    model: subclass instance of pyTorch nn.Module
        The class containing the network model to evaluate, as per convention.
    
    Returns
    ----------
    The tensor of probabilities after softmax.
    
    """
    model.eval()
    out = []
    for i, (input, target) in enumerate(loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
        output = nn.functional.softmax(model(input_var), dim=1)
        out.append(output.data.cpu().numpy())
    return np.concatenate(out)

def validate_ensemble_average(loader, models, numclasses, verbose=True):
    """Evaluate the accuracy and compute the confusion matrix for the averaged
    ensemble of the given set of neural networks on the given data.
    
    Arguments
    ----------
    loader : pyTorch DataLoader instance
        An instance of DataLoader class that supplies the data.
    models: list of subclass instances of pyTorch nn.Module
        A Python list of pyTorch model instances.
    numclasses: int
        The number of classes in the data (for the confusion matrix creation).
    verbose: bool (default True)
        Defines whether the validation process prints progress to the console.
    
    Returns
    ----------
    Tuple of (accuracy, confusion matrix).
    
    """
    top1 = AverageValueMeter()
    cfmat = ConfusionMeter(numclasses)
    
    for m in models:
        m.eval()
        
    for i, (input, target) in enumerate(loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
        tensorlist = []
        for m in models:
            tensorlist.append(nn.functional.softmax(m(input_var),dim=1).data)
        ts = torch.mean(torch.stack(tensorlist, dim=2), 2)
        prec1 = accuracy(ts, target)
        top1.add(prec1[0].item())
        cfmat.add(ts, target)
        
    if verbose:
        print('Validation: * Accuracy {top1.mean:.3f}'
          .format(top1=top1))

    return top1.mean, cfmat.value()

def validate_weighted_ensemble_average(loader, models, numclasses, weight, 
                                       verbose=True):
    """Evaluate the accuracy and compute the confusion matrix for the weighted
    averaged ensemble of the given set of neural networks on the given data.
    
    Arguments
    ----------
    loader : pyTorch DataLoader instance
        An instance of DataLoader class that supplies the data.
    models: list of subclass instances of pyTorch nn.Module
        A Python list of pyTorch model instances.
    numclasses: int
        The number of classes in the data (for the confusion matrix creation).
    weight: double
        
    verbose: bool (default True)
        Defines whether the validation process prints progress to the console.
    
    Returns
    ----------
    Tuple of (accuracy, confusion matrix).
    
    """
    top1 = AverageValueMeter()
    cfmat = ConfusionMeter(numclasses)
    
    for m in models:
        m.eval()
        
    for i, (input, target) in enumerate(loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
        tensorlist = []
        for j, m in enumerate(models):
            out = nn.functional.softmax(m(input_var),dim=1).data
            weights = np.ones((1,numclasses))*(100/(weight+99))
            weights[:,j] *= weight
            tensorlist.append(torch.mul(out, torch.Tensor(weights).cuda()))
        ts = torch.mean(torch.stack(tensorlist, dim=2), 2)
        prec1 = accuracy(ts, target)
        top1.add(prec1[0].item())
        cfmat.add(ts, target)
        
    if verbose:
        print('Validation: * Accuracy {top1.mean:.3f}'
          .format(top1=top1))

    return top1.mean, cfmat.value()

def count_parameters(model):
    """Count the number of differentiable parameters in the model.
    
    Arguments
    ----------
    model: subclass instance of pyTorch nn.Module
        The neural network model in question.
    
    Returns
    ----------
    The number of model parameters as an integer.
    
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(output, target, topk=(1,)):
    """A stolen function to compute accuracy, not meant for external use."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def draw_convfilters(model, numlayer, channel):
    """Draw the convolutional filters of the given model in a Matplotlib 
    figure.
    
    Arguments
    ----------
    model: subclass instance of pyTorch nn.Module
        The neural network model.
    numlayer: int
        The zero-based number of the layer from which the filters are 
        extracted. Only convolutional layers are counted, so numlayer=0 will 
        plot the filters of the first (closest to input) convlayer.
    channel: int
        The channel number from which to plot filters.
        
    Returns
    ----------
    The Matplotlib figure with the filter images in a grid. If the return value
    is not assigned, the plot will open in a separate window by default.
    
    """
    k = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if k!=numlayer:
                k+=1
            else:
                fig = plt.Figure(figsize=(10,10))
                plotside = int( np.ceil( np.sqrt( m.weight.size()[0])))
                for idx, filt in enumerate(m.weight[:,channel,:,:]):
                    plt.subplot(plotside,plotside,idx+1)
                    plt.imshow(filt.data, cmap="gray")
                    plt.axis('off')
                return fig

def save_some_vars(filename, varnames):
    with open(filename, 'wb') as f:
        for v in varnames:
            pickle.dump(v, f)

def data_show(data, nrow):
    """Draw some image data in a grid on a Matplotlib figure.
    
    Arguments
    ----------
    data: np.array or Tensor 
        The data to draw.
    nrow: int
        The number of rows for the image grid.
    Returns
    ----------
    Nothing, the figure window is opened as a side effect.
    
    """
    data_T = [ToTensor()(data[i,:]) for i in range(data.shape[0])]
    def show(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    show(make_grid(data_T, nrow=nrow))
            
def data_show_class(data, data_labels, class_index, nrow):
    """Draw some image data from a specific class in a grid on a 
    Matplotlib figure.
    
    Arguments
    ----------
    data: np.array or Tensor 
        The data to draw.
    data_labels: np.array or Tensor
        The data labels.
    class_index: int
        The index of the class which should be drawn.
    nrow: int
        The number of rows for the image grid.
    Returns
    ----------
    Nothing, the figure window is opened as a side effect.
    
    """
    subset = [data[i,:] for i in np.where(np.asarray(data_labels)==class_index)][0]
    subset_T = [ToTensor()(subset[i,:]) for i in range(subset.shape[0])]
    def show(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    show(make_grid(subset_T, nrow=nrow))
    