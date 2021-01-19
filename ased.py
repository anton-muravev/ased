#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:46:18 2018

@author: muravev
"""
from collections import OrderedDict, abc
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

conv_filter_counts = [8, 16, 32, 64, 96, 128, 256]

# define library functions here
class Ident():
    # identity mapping
    pass

class Conv1x(nn.Conv2d):
    # 1x1 convolution
    def __init__(self, in_channels, out_channels):
        super(Conv1x, self).__init__(in_channels, out_channels, 1)
        
class Conv3x(nn.Conv2d):
    # 3x3 convolution
    def __init__(self, in_channels, out_channels):
        super(Conv3x, self).__init__(in_channels, out_channels, 3, padding=1)
        
class Conv5x(nn.Conv2d):
    # 5x5 convolution
    def __init__(self, in_channels, out_channels):
        super(Conv5x, self).__init__(in_channels, out_channels, 5, padding=2)
    
class Conv7x(nn.Conv2d):
    # 7x7 convolution
    def __init__(self, in_channels, out_channels):
        super(Conv7x, self).__init__(in_channels, out_channels, 7, padding=3)

class DilConv3x(nn.Conv2d):
    # 3x3 dilated convolution
    def __init__(self, in_channels, out_channels):
        super(DilConv3x, self).__init__(in_channels, out_channels, 3,
             dilation=2, padding=2)

class DilConv5x(nn.Conv2d):
    # 5x5 dilated convolution
    def __init__(self, in_channels, out_channels):
        super(DilConv5x, self).__init__(in_channels, out_channels, 5, 
             dilation=2, padding=4)
    
class Max2x(nn.MaxPool2d):
    # 2x2 max pooling
    def __init__(self):
        super(Max2x, self).__init__(2, stride=2, ceil_mode=True)

def fMax2x(x):
    return F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

class Max3x(nn.MaxPool2d):
    # 3x3 max pooling
    def __init__(self):
        super(Max3x, self).__init__(3, stride=2, ceil_mode=True)
        
def fMax3x(x):
    return F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
    
class Avg3x(nn.AvgPool2d):
    # 3x3 avg pooling
    def __init__(self):
        super(Avg3x, self).__init__(3, stride=2, ceil_mode=True)

def fAvg3x(x):
    return F.avg_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
        
DEFAULT_LIBRARY = [Ident, Conv1x, Conv3x, Conv5x, Conv7x, DilConv3x, DilConv5x, 
             Max2x, Max3x, Avg3x]
BLOCK_LIBRARY = [Ident, Conv1x, Conv3x, Conv5x, Conv7x, DilConv3x, DilConv5x]
CLASS_DICT = {"Ident": Ident, "Conv1x": Conv1x, "Conv3x": Conv3x, 
              "Conv5x": Conv5x, "Conv7x": Conv7x, "DilConv3x": DilConv3x,
              "DilConv5x": DilConv5x, "Max2x": Max2x, "Max3x": Max3x,
              "Avg3x": Avg3x }

def get_default_library():
    """Returns a default layer library, defined as a constant inside the class.
    
    Arguments
    ----------
    None.
    
    Returns
    ----------
    A list of layer classes.
    """
    return DEFAULT_LIBRARY

def get_nopooling_library():
    """Returns a layer library without pooling layers.
    
    Arguments
    ----------
    None.
    
    Returns
    ----------
    A list of layer classes.
    """
    return BLOCK_LIBRARY
    
def get_uniform_prototype(nlayer, opLibrary):
    """Creates a prototype over the uniform layer distribution (all 
    probabilities are equal).
    
    Arguments
    ----------
    nlayer: int
        A number of layers in the prototype
    opLibrary: list of layer classes
        The layer library.
    
    Returns
    ----------
    A 2-dimensional np.array of layer probabilities.
    """
    protogene = np.repeat(1/len(opLibrary), len(opLibrary))
    return np.tile(protogene, (nlayer,1))

def normalize_prototype(prototype, libsize, upbound):
    """Rescales prototype values to not exceed the given upper bound, while
    maintaining the sum. The lower bound is computed internally from the upper
    bound.
    
    Arguments
    ----------
    prototype: 2-d np.array
        The original prototype to be normalized.
    libsize: int
        The size of the layer library (number of possibilities).
    upbound: double
        The upper bound of allowed probability values.
    
    Returns
    ----------
    2-d np.array containing a new normalized prototype.
    """
    lowbound = (1.-upbound)/(libsize-1)
    newtype = prototype.copy()
    row, col = np.where(newtype < lowbound)
    row, count = np.unique(row, return_counts=True)
    for i, k in enumerate(row):
        small_indices = np.flatnonzero(newtype[k,:]<lowbound)
        big_indices = np.flatnonzero(newtype[k,:]>lowbound)
        smallsum = newtype[k, small_indices].sum()
        bigsum = newtype[k, big_indices].sum()
        cumulate = count[i]*lowbound - smallsum
        multiplier = (bigsum-cumulate)/bigsum
        newrow = np.repeat(lowbound, libsize)
        newrow[big_indices] = newtype[k, big_indices]*multiplier
        newtype[k,:] = newrow
    return newtype

def invert_prototype(prototype, libsize, upbound):
    return normalize_prototype((1-prototype)/(libsize-1), libsize, upbound)

def sample_network(prototype, opLibrary):
    """Sample a network structure from the library, using a given prototype.
    
    Arguments
    ----------
    prototype: 2-d np.array
        The prototype with sampling probabilities.
    opLibrary: list of layer classes
        The layer library.
    
    Returns
    ----------
    An ordered dictionary, where keys are layer names and values are layer 
    instances from the library.
    """
    layerdict = OrderedDict()
    for i in range(len(prototype)):
        l = np.random.choice(opLibrary, p=prototype[i,])
        layerdict['layer'+str(i)+'_'+str(l.__name__)] = l
    return layerdict

def generate_feature_network(in_channels, def_channels, input_size, prototype,
                           opLibrary, batch_norms=False, nonrandom=False):
    """Generates a network (without the classification layer) from the given
    prototype and library.
    
    Arguments
    ----------
    in_channels: int
        The number of input channels.
    def_channels: int
        The default width of intermediate layers.
    input_size: int
        The input dimensionality.
    prototype: 2-d np.array
        The prototype with layer probabilities.
    opLibrary: list of layer classes
        The layer library.
    batch_norms: bool, default False
        Whether to include batch normalization after convolutional layers.
    nonrandom: boo, default False
        If true, takes most likely layers instead of sampling (used during
        validation).
    
    Returns
    ----------
    A sampled network as an instance of EvoNet class.
    """
    
    layerdict = OrderedDict()
    cur_channels = in_channels
    data_size = input_size
    for i in range(len(prototype)):
        if nonrandom:
            l = opLibrary[np.argmax(prototype[i,])]
        else:
            l = np.random.choice(opLibrary, p=prototype[i,])
        lname = 'layer'+str(i)+'_'+str(l.__name__)
        if issubclass(l, Ident):
            continue
        elif issubclass(l, nn.Conv2d):
            layerdict[lname] = l(cur_channels, def_channels)
            cur_channels = def_channels
            layerdict[lname+'_act'] = nn.PReLU()
            if batch_norms:
                layerdict[lname+'_bn'] = nn.BatchNorm2d(cur_channels)
        else:
            if data_size==1:
                continue
            layerdict[lname] = l()
            data_size = int(np.ceil(data_size/2))
            
    class EvoNet(nn.Module):
        
        def __init__(self):
            super(EvoNet, self).__init__()
            self.features = nn.Sequential(layerdict)
            self.structure = layerdict
            self.prototype = prototype
            self.in_channels = in_channels
            self.out_channels = cur_channels
            self.out_size = data_size
            
        def forward(self, x):
            return self.features(x)
            
    return EvoNet()

def generate_shortcut_feature_network(in_channels, def_channels, input_size, 
                                      prototype, opLibrary, 
                                      shortcuts='residual', skip_value=1,
                                      batch_norms=False, nonrandom=False):
    """Generates a sequence of feature extracting blocks with shortcut 
    connections of the chosen pattern from the given prototype and library.
    
    Arguments
    ----------
    in_channels: int
        The number of input channels.
    def_channels: int
        The default width of intermediate layers.
    input_size: int
        The input dimensionality.
    prototype: 2-d np.array
        The prototype with layer probabilities.
    L: int
        The number of blocks in the sequence.
    skip_value: int
        The scope parameter for shortcut connections.
    opLibrary: list of layer classes
        The layer library.
    batch_norms: bool, default False
        Whether to include batch normalization after convolutional layers.
    nonrandom: boo, default False
        If true, takes most likely layers instead of sampling (used during
        validation).
    
    Returns
    ----------
    A sampled network as an instance of EvoNet class.
    """
    # check arguments
    if shortcuts not in ['residual', 'dense', 'prototype', 'none']:
        raise ValueError("Not an allowed argument for skip_connections")
    if shortcuts == 'none':
        return generate_feature_network(in_channels, def_channels, input_size,
                                      prototype, opLibrary, batch_norms, 
                                      nonrandom)
    if skip_value<0:
        raise ValueError("Skip value must be 0 or higher")
    if not skip_value:
        skip_value = -1
    layerdict = OrderedDict()
    if batch_norms:
        bn_dict = OrderedDict()
    cur_channels = in_channels
    data_size = input_size
    dict_out = dict()
    dict_in = dict()
    skip_dict = dict()
    op_dict = dict()
    skip_counter = 0
    index_counter = 0
    ignore_skips = True
    prevname = None
    for i in range(len(prototype)):
        if nonrandom:
            l = opLibrary[np.argmax(prototype[i,])]
        else:
            l = np.random.choice(opLibrary, p=prototype[i,])
        lname = 'layer'+str(i)+'_'+str(l.__name__)
        if issubclass(l, Ident):
            continue
        elif issubclass(l, nn.Conv2d):
            layerdict[lname] = l(cur_channels, def_channels)
            cur_channels = def_channels
            layerdict[lname+'_act'] = nn.PReLU()
            if batch_norms:
                bn_dict[lname+'_act'] = nn.BatchNorm2d(cur_channels)
            ignore_skips = False
        else:
            if data_size==1:
                continue
            layerdict[lname] = l()
            data_size = int(np.ceil(data_size/2))
            if skip_value<0:
                skip_counter = -1
            
        if ignore_skips:
            skip_counter=0
            continue
        for index in op_dict:
            op_dict[index].append(lname)
        if shortcuts == 'residual':
            if skip_counter==0:
                dict_out[lname] = [index_counter]
                skip_dict[index_counter] = [lname, None, None, []]
                op_dict[index_counter] = []
            elif skip_counter==skip_value:
                if skip_value<0:
                    lname = prevname
                dict_in[lname] = [index_counter]
                skip_dict[index_counter][1] = lname
                skip_dict[index_counter][3] = op_dict[index_counter]
                op_dict.clear()
                index_counter += 1
                skip_counter = -1
        elif shortcuts == 'dense':
            if skip_counter!=skip_value:
                dict_out[lname] = [index_counter]
                skip_dict[index_counter] = [lname, None, None, []]
                op_dict[index_counter] = []
                index_counter += 1
            else:
                if skip_value<0:
                    dict_in[lname] = list(op_dict.keys())
                    for index in op_dict.keys():
                        skip_dict[index][1] = lname
                        skip_dict[index][3] = op_dict[index]
                    op_dict.clear()
                    skip_counter = -1 
                else:
                    dict_in[lname] = list(op_dict.keys())
                    for index in op_dict.keys():
                        skip_dict[index][1] = lname
                        skip_dict[index][3] = op_dict[index]
                    op_dict.clear()
                    skip_counter = -1            
        skip_counter += 1
        prevname = lname
    # check skip connections
    for k, v in list(skip_dict.items()):
        # delete connections without endpoints
        if v[1] == None:
            dict_out[v[0]].remove(k)
            del skip_dict[k]
            continue
        # get rid of endpoint identities
        pure_ident = False
        while v[1].find('Ident')>0:
            dict_in[v[1]].remove(k)
            v[3].pop()
            if not v[3]:
                pure_ident = True
                break
            v[1] = v[3][-1]
            if v[1] not in dict_in:
                dict_in[v[1]] = []
            dict_in[v[1]].append(k)
        if pure_ident:
            dict_out[v[0]].remove(k)
            del skip_dict[k]
            continue
        # process intermediate operations
        op_list = []
        last_conv = None
        for op in v[3]:
            # check if there are convolutions between skips
            if op.find('Conv')>0:
                last_conv = op
            elif op.find('Ident')>0:
                pass
            # check if the entry point needs to be moved past pooling
            else:
                if not last_conv:
                    # check for identical connections
                    if op in dict_out:
                        same_skip = False
                        for m in dict_out[op]:
                            if skip_dict[m][1] == v[1]:
                                same_skip = True
                                break
                        if same_skip:
                            break
                    else:
                        dict_out[op] = []
                    dict_out[v[0]].remove(k)
                    dict_out[op].append(k)
                    v[0] = op
                else:    
                    op_list.append(op)
        if not last_conv:
            dict_out[v[0]].remove(k)
            dict_in[v[1]].remove(k)
            del skip_dict[k]
            continue
        op_list_f = []
        for op in op_list:
            if op.find('Max2x')>0:
                op_list_f.append(fMax2x)
                continue
            if op.find('Max3x')>0:
                op_list_f.append(fMax3x)
                continue
            if op.find('Avg3x')>0:
                op_list_f.append(fAvg3x)
        v[3] = op_list_f
    # clean up redundant elements in dictionaries
    for k, v in list(dict_out.items()):
        if not v:
            del dict_out[k]
    for k, v in list(dict_in.items()):
        if not v:
            del dict_in[k]
    
    class EvoNet(nn.Module):
        
        def __init__(self):
            super(EvoNet, self).__init__()
            self.features = nn.ModuleDict(layerdict)
            self.batch_norms = nn.ModuleDict(bn_dict)
            self.structure = layerdict
            self.prototype = prototype
            self.in_channels = in_channels
            self.out_channels = cur_channels
            self.out_size = data_size
            self.dict_out = dict_out
            self.dict_in = dict_in
            self.skip_dict = skip_dict           
            
        def forward(self, x):
            for layername in self.features:
                x = self.features[layername](x)
                if layername in self.dict_out:
                    for k in self.dict_out[layername]:
                        self.skip_dict[k][2] = x
                if layername in self.dict_in:
                    for k in self.dict_in[layername]:
                        x1 = self.skip_dict[k][2]
                        for op in self.skip_dict[k][3]:
                            x1 = op(x1)
                        x+=x1
                if layername in self.batch_norms:
                    x = self.batch_norms[layername](x)
            return x
            
    return EvoNet()

def init_feature_network(network):
    """Initializes a network, using Kaiming-He method.
    
    Arguments
    ----------
    network: EvoNet instance
        A network to initialize.
    
    Returns
    ----------
    Nothing.
    """
    for m in network.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight, a=0.25)
            m.bias.data.zero_()
            
def compute_output_channels(network, input_size):
    """Computes the number of output channels of the network. It is assumed 
    that only pooling operations affect the size change.
    
    Arguments
    ----------
    network: EvoNet instance
        A network.
    input_size: int
        The input size of the network.
    
    Returns
    ----------
    Integer: the number of output channels.
    """
    output_size = input_size
    for m in network.structure:
        if m.find('Max')>0 or m.find('Avg')>0:
            output_size = int(np.ceil(output_size/2))
    return int(output_size)*int(output_size)*network.out_channels

class EvalNet(nn.Module):
    """The network class which incorporates the sampled feature extractor and 
    a fully-connected classifier layer.
    """
    def __init__(self, featblock, nclasses, fc_neurons=0):
        """Creates an EvalNet instance and initializes with Kaiming-He method.
        
        Arguments
        ----------
        featboi: EvoNet instance
            The feature extractor obtained via sampling.
        nclasses: int
            The number of output classes for the classifier layer.
        fc_neurons: int
            The number of neurons in a fully connected layer (can be zero).
        
        """
        super(EvalNet, self).__init__()
        self.features = featblock
        init_feature_network(self.features)
        channels = compute_output_channels(featblock, self.out_size)
        if fc_neurons!=0:
            self.classifier = nn.Sequential(nn.Linear(channels, fc_neurons), 
                                            nn.PReLU(), 
                                            nn.Linear(fc_neurons, nclasses))
            self.classifier[0].bias.data.zero_()
            init.kaiming_uniform_(self.classifier[0].weight, a=0.25)
        else:
            self.classifier = nn.Sequential(nn.Linear(channels, nclasses))
            
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class EvalNet2(nn.Module):
    """The network class which incorporates the sampled feature extractor and 
    a fully-connected classifier layer. Uses global average pooling to reduce
    the complexity of the classifier layer.
    """
    def __init__(self, featblock, nclasses, fc_neurons=0):
        """Creates an EvalNet2 instance and initializes with Kaiming-He method.
        
        Arguments
        ----------
        featboi: EvoNet instance
            The feature extractor obtained via sampling.
        nclasses: int
            The number of output classes for the classifier layer.
        fc_neurons: int
            The number of neurons in a fully connected layer (can be zero).
        
        """
        super(EvalNet2, self).__init__()
        self.features = featblock
        init_feature_network(self.features)
        channels = featblock.out_channels
        if fc_neurons!=0:
            self.classifier = nn.Sequential(nn.Linear(channels, fc_neurons), 
                                            nn.PReLU(), nn.Dropout(),
                                            nn.Linear(fc_neurons, nclasses))
            self.classifier[0].bias.data.zero_()
            init.kaiming_uniform_(self.classifier[0].weight, a=0.25)
        else:
            self.classifier = nn.Sequential(nn.Linear(channels, nclasses))
            
    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x,1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def network_to_phenotype(net):
    """Converts a network instance to a phenotype representation, 
    i.e. a structure dictionary.
    
    Arguments
    ----------
    net: EvoNet instance
        The network to convert.
    
    Returns
    ----------
    An ordered dictionary. Phenotype contains prototype, channel counts and
    layer information.
    """
    phenotype = OrderedDict()
    phenotype["prototype"] = net.prototype
    phenotype["in_channels"] = net.in_channels
    phenotype["out_channels"] = net.out_channels
    for layer in net.structure:
        if layer.endswith("_act"):
            continue
        if hasattr(net.structure[layer], "in_channels"):
            in_channels = net.structure[layer].in_channels
            out_channels = net.structure[layer].out_channels
        else:
            in_channels = None
            out_channels = None
        dct = dict(classname=layer.split('_')[1], in_channels=in_channels,
                   out_channels=out_channels)
        phenotype[layer.split('_')[0]] = dct
    return phenotype
    
def phenotype_to_network(phenotype):
    """Creates a network instance from the phenotype representation.
    
    Arguments
    ----------
    phenotype: OrderedDict
        The phenotype in the specific format.
    
    Returns
    ----------
    A network instance via EvoNet class.
    """
    layerdict = OrderedDict()
    prototype = phenotype.popitem(False)[1]
    in_channels = phenotype.popitem(False)[1]
    out_channels = phenotype.popitem(False)[1]
    if out_channels is None:
        out_channels = in_channels
    for layer in phenotype:
        classname = phenotype[layer]["classname"]
        class_ = CLASS_DICT[classname]
        if issubclass(class_, Ident):
            continue
        elif issubclass(class_, nn.Conv2d):
            layerdict[layer+'_'+classname] = class_(phenotype[layer]["in_channels"],
                                      phenotype[layer]["out_channels"])
            layerdict[layer+'_'+classname+'_act'] = nn.PReLU()
        else:
            layerdict[layer+'_'+classname] = class_()
    phenotype["prototype"] = prototype
    phenotype["in_channels"] = in_channels
    phenotype["out_channels"] = out_channels
    phenotype.move_to_end("out_channels", last=False)
    phenotype.move_to_end("in_channels", last=False)
    phenotype.move_to_end("prototype", last=False)
                    
    class EvoNet(nn.Module):
        
        def __init__(self):
            super(EvoNet, self).__init__()
            self.features = nn.Sequential(layerdict)
            self.structure = layerdict
            self.prototype = prototype
            self.in_channels = in_channels
            self.out_channels = out_channels
            
        def forward(self, x):
            return self.features(x)
            
    return EvoNet()
        
def phenotype_to_binary(phenotype):
    """Converts a phenotype to the binary representation, i.e. one-hot encoding
    of layer choices.
    
    Arguments
    ----------
    phenotype: OrderedDict
        The phenotype in the specific format.
    
    Returns
    ----------
    2-d np.array with one-hot encoded layers.
    """
    length = phenotype["prototype"].shape[0]
    binary_code = np.zeros((length, len(DEFAULT_LIBRARY)))
    for i in range(length):
        try:
            layer = phenotype["layer"+str(i)]["classname"]
            ind = DEFAULT_LIBRARY.index(CLASS_DICT[layer])
            binary_code[i,ind] = 1
        except KeyError:
            binary_code[i,0] = 1
    return binary_code

def phenotype_to_readable_string(phenotype):
    """Converts a phenotype to a human-readable string for printing.
    
    Arguments
    ----------
    phenotype: OrderedDict
        The phenotype in the specific format. 
    
    Returns
    ----------
    String with a human-readable description of the network structure.
    """
    out_string = ""
    layer_counter = 0
    for k in phenotype:
        if k[0:5] != "layer":
            continue
        expected = "layer"+str(layer_counter)
        while k != expected:
            out_string += "->X"
            layer_counter+=1
            expected = "layer"+str(layer_counter)
        out_string += "->"+phenotype[k]["classname"]
        layer_counter+=1
    return out_string

def change_phenotype_channels(phenotype, channels):
    """Changes the width of all the convolutional layers in the given 
    phenotype to the given value.
    
    Arguments
    ----------
    phenotype: OrderedDict
        The phenotype in the specific format.
    channels: int
        The new width value for convolutional layers.
    
    Returns
    ----------
    OrderedDict -- a modified phenotype.
    """
    prototype = phenotype.popitem(False)[1]
    in_channels = phenotype.popitem(False)[1]
    out_channels = phenotype.popitem(False)[1]
    cur_channels = in_channels
    
    new_phenotype = OrderedDict()
    new_phenotype["prototype"] = prototype
    new_phenotype["in_channels"] = in_channels
    new_phenotype["out_channels"] = out_channels
    
    if isinstance(channels, abc.Sequence):
        for index, key in enumerate(phenotype):
            new_phenotype[key] = phenotype[key].copy()
            if new_phenotype[key]["in_channels"] is not None:
                new_phenotype[key]["in_channels"] = cur_channels
                new_phenotype[key]["out_channels"] = channels[index]
                cur_channels = channels[index]
    else:
        for key in phenotype:
            new_phenotype[key] = phenotype[key].copy()
            if new_phenotype[key]["in_channels"] is not None:
                new_phenotype[key]["in_channels"] = cur_channels
                new_phenotype[key]["out_channels"] = channels
                cur_channels = channels
                
    new_phenotype["out_channels"] = cur_channels
    
    phenotype["prototype"] = prototype
    phenotype["in_channels"] = in_channels
    phenotype["out_channels"] = out_channels
    phenotype.move_to_end("out_channels", last=False)
    phenotype.move_to_end("in_channels", last=False)
    phenotype.move_to_end("prototype", last=False)
    
    return new_phenotype