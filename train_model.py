from __future__ import division
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-model_file', type=str)
parser.add_argument('-name', type=str)
parser.add_argument('-name2', type=str)
parser.add_argument('-k', type=int)
parser.add_argument('-gpu', type=str, default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms


import numpy as np
import json

import models


batch_size = 32
from pitch_dataset import Pitcher
classes = 1

def sigmoid(x):
    return 1/(1+np.exp(-x))

def load_data(name, name2, k):
    dataset = Pitcher(name, k, sub=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    
    val_dataset = Pitcher(name,k, sub=1)
    val_dataset.mode = 'val'
    val_dataset.val = dataset.val
    val_dataset.train = dataset.train
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


# train the model
def run(models, num_epochs=50):
    since = time.time()

    best_loss = 10000
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        probs = []
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            train_step(model, gpu, optimizer, dataloader['train'], epoch)
            #if epoch % 5 == 0:
            prob_val, val_loss = val_step(model, gpu, dataloader['val'], epoch)
            probs.append(prob_val)
            sched.step(val_loss)

            if False and val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'models/'+model_file)

def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[2]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1]/other[1][0]

        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    return results

def run_network(model, data, gpu, epoch, val=False, baseline=True):
    # get the inputs
    inputs, labels, other = data
    
    # wrap them in Variable
    inputs = Variable(inputs.cuda(gpu))
    labels = Variable(labels.cuda(gpu)).float()

    # forward
    outputs = model(inputs)
    
    # binary action-prediction loss
    loss = F.binary_cross_entropy_with_logits(outputs, labels)

    corr = torch.sum(((outputs>0.5).float() == labels).float())
    tot = outputs.size(0)

    thresh = 0.5
    preds = outputs > thresh

    labels = labels.byte()
    pos = (labels == 1)
    neg = (labels == 0)
    tp = torch.sum((pos * (preds == labels)).float())
    fp = torch.sum((pos * (preds != labels)).float())
    tn = torch.sum((neg * (preds == labels)).float())
    fn = torch.sum((neg * (preds != labels)).float())

    pos = torch.sum(pos.float())
    neg = torch.sum(neg.float())
    ppos = torch.sum((preds == 1).float())
    pneg = torch.sum((preds == 0).float())
    stats = (pos, neg, tp, fp, tn, fn, ppos, pneg)
    
    return outputs, loss, torch.sigmoid(outputs), corr/tot, stats
                

def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    pos = 0.
    neg = 0.
    ppos = 0.
    pneg = 0.
    tp = tn = fp = fn = 0.
    
    # Iterate over data.
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1
        
        outputs, loss, probs, err, stats = run_network(model, data, gpu, epoch)
        
        error += err.data[0]
        pos += stats[0].data[0]
        neg += stats[1].data[0]
        tp += stats[2].data[0]
        fp += stats[3].data[0]
        tn += stats[4].data[0]
        fn += stats[5].data[0]
        ppos += stats[6].data[0]
        pneg += stats[7].data[0]
        
        tot_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    epoch_loss = tot_loss / num_iter
    error = error / num_iter
    print(('train Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, error)))
    print('FNR:', fn/(pos+1e-6), 'FPR:', fp/(neg+1e-6), 'TNR/specificity:', tn/(neg+1e-6))
    rec = (tp/(pos+1e-6))+1e-6
    prec = tp/(ppos+1e-6)+1e-6
    print('Recall:',rec,'Precision:',prec)
    print('F1:', 1/((0.5*((1/rec)+(1/prec)))+1e-6))
  

def val_step(model, gpu, dataloader, epoch):
    model.train(False)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    num_preds = 0
    pos = 0.
    neg = 0.
    ppos = 0.
    pneg = 0.
    tp = tn = fp = fn =0.
    

    full_probs = {}


    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        other = data[2]
        
        outputs, loss, probs, err, stats = run_network(model, data, gpu, epoch, val=True)
                
        error += err.data[0]
        tot_loss += loss.data[0]
        pos += stats[0].data[0]
        neg += stats[1].data[0]
        tp += stats[2].data[0]
        fp += stats[3].data[0]
        tn += stats[4].data[0]
        fn += stats[5].data[0]
        ppos += stats[6].data[0]
        pneg += stats[7].data[0]
                                                                        
        
        # post-process preds
        outputs = outputs.squeeze()
        probs = probs.squeeze()
        full_probs[other[0][0]] = (probs.data.cpu().numpy().T, 0)
        
        
    epoch_loss = tot_loss / num_iter
    error = error / num_iter

    print('val Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, error))
    print('FNR:', fn/(pos+1e-6), 'FPR:', fp/(neg+1e-6), 'TNR/specificity:', tn/(neg+1e-6))
    rec = (tp/(pos+1e-6))+1e-6
    prec = tp/(ppos+1e-6)+1e-6
    print('Recall:',rec,'Precision:',prec)
    print('F1:', 1/((0.5*((1/rec)+(1/prec)))+1e-6))
    
                            
    return full_probs, epoch_loss


if __name__ == '__main__':

    print args.name
    print args.k

    dataloaders, datasets = load_data(args.name, args.name2, args.k)

    model = nn.DataParallel(models.AvgPool(inp=1024,classes=classes).cuda())

    lr = 0.1
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        
    run([(model,0,dataloaders,optimizer, lr_sched, args.model_file)], num_epochs=100)
