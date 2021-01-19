#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from collections.abc import MutableMapping

class Astore(MutableMapping):
    """Class to store the data arrays. Extends the standard dictionary,
    but offers additional file operations.
    
    Needs to be initialized before use."""
    
    def __init__(self):
        """Initialize with an empty dictionary."""
        self.dicty = dict()
    
    def __getitem__(self, key):
        return self.dicty.__getitem__(key)
    
    def __setitem__(self, key, value):
        return self.dicty.__setitem__(key, value)
        
    def __delitem__(self, key):
        return self.dicty.__delitem__(key)
        
    def keys(self):
        return self.dicty.keys()
            
    def __iter__(self):
        yield from self.dicty
        
    def __len__(self):
        return self.dicty.__len__()
        
    def extract(self, key):
        """Return the safe copy of the given contained variable.
        
        Arguments
        ----------
        key: string
            The key of the target variable.
            
        Returns
        ----------
        A memory copy of the given variable, safe for modification."""
        return self.dicty[key].copy()
        
    def load(self, filename, names=None):
        """Fill the store with the contents of a given file.
        
        This function can operate in two modes. If the names argument is not 
        provided, the new file format is implied (pickled file containing only
        a single dictionary). Otherwise the contents of the names argument are
        used to deserialize the file contents in the given order.
        
        Arguments
        ----------
        filename: string
            Full name of the file to load from.
        names: list of strings, optional
            Indicates the variable names to be loaded from the file (in the
            given order).
        
        Returns
        ----------
        Nothing.
        
        """
        if names is None:
            with open(filename, 'rb') as f:
                self.dicty = pickle.load(f)
        else:
            with open(filename, 'rb') as f:
                for k in names:
                    self.dicty[k] = pickle.load(f)
                
    def get_names(self):
        """Return the list of value names."""
        return list(self.dicty.keys())
        
    def dump(self, filename):
        """Save the store contents to the given file.
        
        This operation just pickles the internal dictionary.
        
        Arguments
        ----------
        filename: string
            Full name of the file to save to.
        
        Returns
        ----------
        Nothing."""
        with open(filename, 'wb') as f:
            pickle.dump(self.dicty, f)
        
