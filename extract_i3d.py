import os
from tracemalloc import start
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)

args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable


import numpy as np
import cv2

# import lintel

from pytorch_i3d import InceptionI3d
import scipy.misc


import nvidia_smi
import gc
# gc.collect()
# torch.cuda.empty_cache()
def memoryCheck():
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
    nvidia_smi.nvmlShutdown()


save_dir = '/home/ec2-user/injuryproject/InjuryVideo/RGB/'

i3d = InceptionI3d(400, in_channels=3)
i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
i3d.cuda()
i3d.train(False)

with open('injury_data.json', 'r') as f:
    data = json.load(f)

import random
p = data['pitch_data']
random.shuffle(p)
numVids = 0
for v in p:
    vid = v['clip_name']
    
    # numVids += 1
    # if(numVids%100==0):
    #     print(numVids, "---------------------")

    if not os.path.exists(os.path.join('/home/ec2-user/injuryproject/InjuryVideo/DataCropped/', vid+'.mp4')):
        continue

    # with open(os.path.join('/home/ec2-user/injuryproject/InjuryVideo/DataCropped/', vid+'.mp4'), 'rb') as f:
    #     enc_vid = f.read()

    if os.path.exists(os.path.join(save_dir, vid+'.npy')):
        continue

    num_frames = 60*10
    # df, w, h, u = lintel.loadvid(enc_vid, num_frames=num_frames)

    df = []
    vidcap = cv2.VideoCapture(os.path.join('/home/ec2-user/injuryproject/InjuryVideo/DataCropped/', vid+'.mp4'))
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            df.append(image)    # save frame as JPG file
            # print(image)
        return hasFrames
    sec = 0
    frameRate = 1/60
    count=1
    success = getFrame(sec)
    while success and count<600:
        count = count + 1
        print(count)
        sec = sec + frameRate
        # sec = round(sec, 2)
        success = getFrame(sec)
    h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if(h == 0 or w==0):
        continue

    del vidcap
    del sec, frameRate, count, success

    df = np.array(df)

    df = np.frombuffer(df, dtype=np.uint8)
    df = np.reshape(df, newshape=(num_frames, h, w, 3))
    print(df.shape)

    memoryCheck()
    
    gc.collect()
    torch.cuda.empty_cache()

    memoryCheck()

    dfi = 2*(df.astype(np.float32)/255)-1

    del df

    batch_size = 10

    with torch.no_grad():
        start = 0
        end = batch_size
        print(start, end)
        
        df = Variable(torch.from_numpy(dfi[start:end].transpose([3,0,1,2])).cuda())
        df = df.unsqueeze(0)
        combined = i3d.extract_features(df).squeeze(0).permute(1,2,3,0).data.cpu().numpy()
    
        start += batch_size
        end += batch_size

        del df

        print(combined.shape)

        while(end<=600):
            gc.collect()
            torch.cuda.empty_cache()

            memoryCheck()

            print(start, end)
            df = Variable(torch.from_numpy(dfi[start:end].transpose([3,0,1,2])).cuda())
            df = df.unsqueeze(0)
            fts = i3d.extract_features(df).squeeze(0).permute(1,2,3,0).data.cpu().numpy()
            combined = np.concatenate([combined,fts],axis=0)
            start += batch_size
            end += batch_size

            del df
            del fts

            # df = Variable(torch.from_numpy(dfi[(300-68):].transpose([3,0,1,2])).cuda())
            # df = df.unsqueeze(0)
            # fts2 = i3d.extract_features(df).squeeze(0).permute(1,2,3,0).data.cpu().numpy()
            # fts = np.concatenate([fts,fts2[9:]],axis=0)
            #print(fts2.shape)
            #print(fts.shape)
            #exit()
        np.save(os.path.join(save_dir, vid), combined)
        print(combined.shape, vid)

        gc.collect()
        torch.cuda.empty_cache()

        del combined

        

