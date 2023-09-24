import os
import sys
import argparse
import json

import numpy as np


save_dir = '/Users/juxiang/Documents/InjuryVideo/RBG/'

with open('injury_data.json', 'r') as f:
    data = json.load(f)

import random
import cv2

p = data['pitch_data']
# random.shuffle(p)
for v in p:
    vid = v['clip_name']

    with open(os.path.join('/Users/juxiang/Documents/InjuryVideo/Data/', vid+'.mp4'), 'rb') as f:
        enc_vid = f.read()

    if os.path.exists(os.path.join(save_dir, vid+'.npy')):
        continue

    num_frames = 60*10

    df = []
    vidcap = cv2.VideoCapture(os.path.join('/Users/juxiang/Documents/InjuryVideo/Data/', vid+'.mp4'))
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            df.append(image)    # save frame as JPG file
            # print(image)
        return hasFrames
    sec = 0
    frameRate = 1/60
    print(frameRate) #//it will capture image in each 0.5 second
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
    df = np.array(df)
    df = np.frombuffer(df, dtype=np.uint8)
    df = np.reshape(df, newshape=(num_frames, h, w, 3))
    # print(h, w)
    
    # df, w, h, u = lintel.loadvid(enc_vid, num_frames=num_frames)
    print(df.shape)
    print(df)
    break