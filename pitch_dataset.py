from __future__ import division
import torch
import random
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv  
import h5py

import os
import os.path

import get_subsets

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))



def make_dataset(pitcher_name='Aaron Nola', k=20):
    #dataset = get_subsets.last_k_as_injured(get_subsets.get_by_pitcher(pitcher_name), k)
    dataset = get_subsets.last_k_as_injured(get_subsets.get_by_arm(pitcher_name), k)
    tot = healthy = inj = 0.
    for p in dataset:
        tot += 1
        if p['inj'] == 0:
            healthy += 1
        else:
            inj += 1
    print(pitcher_name, tot, 'healthy:', healthy, 'inj:', inj)
    return dataset

class Pitcher(data_utl.Dataset):

    def __init__(self, name, k, sub=0.8):
        
        self.data = make_dataset(name, k)
        random.shuffle(self.data)
        self.train = self.data[:int(len(self.data)*sub)]
        self.val = self.data[int(len(self.data)*sub):]
        self.root = '/Users/juxiang/Documents/InjuryVideo/RBG/'
        self.in_mem = {}
        self.mode = 'train'
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.mode == 'train':
            entry = self.train[index]
        else:
            entry = self.val[index]
        if entry['clip_name'] in self.in_mem:
            feat = self.in_mem[entry['clip_name']]
        else:
            feat = np.load(os.path.join(self.root, entry['clip_name']+'.npy'))
            feat = feat.astype(np.float32)
            self.in_mem[entry['clip_name']] = feat
            
        label = entry['inj']
        return video_to_tensor(feat), label, entry['clip_name']

    def __len__(self):
        if self.mode == 'train':
            return len(self.train)
        return len(self.val)
