import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)

args = parser.parse_args()
#os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable


import numpy as np
import lintel

from pytorch_i3d import InceptionI3d
import scipy.misc


save_dir = '/Users/juxiang/Documents/InjuryVideo/RBG/'

i3d = InceptionI3d(400, in_channels=3)
i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
i3d.cuda()
i3d.train(False)

with open('injury_data.json', 'r') as f:
    data = json.load(f)

import random
p = data['pitch_data']
random.shuffle(p)
for v in p:
    vid = v['clip_name']

    with open(os.path.join('/Users/juxiang/Documents/InjuryVideo/Data/', vid+'.mp4'), 'rb') as f:
        enc_vid = f.read()

    if os.path.exists(os.path.join(save_dir, vid+'.npy')):
        continue

    num_frames = 60*10
    df, w, h, u = lintel.loadvid(enc_vid, num_frames=num_frames)
    df = np.frombuffer(df, dtype=np.uint8)

                          
    df = np.reshape(df, newshape=(num_frames, h, w, 3))
    print(df.shape)
    # exit()

    dfi = 2*(df.astype(np.float32)/255)-1

    with torch.no_grad():
        df = Variable(torch.from_numpy(dfi[:368].transpose([3,0,1,2])).cuda())
        df = df.unsqueeze(0)
        fts = i3d.extract_features(df).squeeze(0).permute(1,2,3,0).data.cpu().numpy()
        fts = fts[:-8]

        df = Variable(torch.from_numpy(dfi[(300-68):].transpose([3,0,1,2])).cuda())
        df = df.unsqueeze(0)
        fts2 = i3d.extract_features(df).squeeze(0).permute(1,2,3,0).data.cpu().numpy()
        fts = np.concatenate([fts,fts2[9:]],axis=0)
        #print(fts2.shape)
        #print(fts.shape)
        #exit()
        np.save(os.path.join(save_dir, vid), fts)
        print(fts.shape, vid)
