'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import pickle as pkl

import math
import sys
import time

import argparse

from eeg_data import EEG_RNNIterator, EEG_AEIterator
from eeg_model import EEG_Chung, EEG_AE_MODEL

def train(model, input, target, optimizer, criterion, m2m, model_type):
    model.train()
    
    optimizer.zero_grad()

    output = model(input)

    if m2m==1 or model_type=='ae':
        loss = criterion(output, target) # reduce to scalar?
    else:        
        loss = criterion(output[-1,:,:], target) # reduce to scalar?

    loss.backward()
    
    optimizer.step()

    return output, loss.item()

def valid_model(model, validiter, device, criterion, m2m, model_type):
    current_loss = 0
    cnt = 0
    model.eval()
    with torch.no_grad(): 
        for iloop, (tr_x, tr_y, end_of_data) in enumerate(validiter): # no need for mask
            tr_x, tr_y = Variable(tr_x).to(device), Variable(tr_y).to(device)
            
            output = model(tr_x)
            
            if m2m==1 or model_type=='ae':
                loss = criterion(output, tr_y) # reduce to scalar? 
            else:     
                loss = criterion(output[-1,:,:], tr_y) # reduce to scalar?
 
            current_loss += loss
            cnt += 1
            
            if end_of_data == 1:
                break

    return current_loss / cnt

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
 
def train_main(args, neptune):
    device = torch.device("cuda") 
    
    if args.model == 'lstm':
        trainiter = EEG_RNNIterator(args.tr_path, batch_size = args.batch_size, train=True, args=args, m2m=args.m2m, shuffle=args.shuffle, rnn_len=args.rnn_len)
        validiter = EEG_RNNIterator(args.val_path, batch_size = 1, train=False, args=args, m2m=args.m2m)
        model = EEG_Chung(dim_input=args.dim_input, dim_hidden=args.dim_hidden, dim_out=args.dim_input).to(device) # dim_input = dim_output 
         
    if args.model == 'ae':
        trainiter = EEG_AEIterator(args.tr_path, batch_size=args.batch_size, train=True, args=args)
        validiter = EEG_AEIterator(args.val_path, batch_size=1, train=False, args=args)
        
        model = EEG_AE_MODEL(dim_input=args.dim_input, dim_layer=args.dim_layer, dim_z=args.dim_z).to(device)

    # define loss
    mystring = "optim." + args.optimizer
    optimizer = eval(mystring)(model.parameters(), args.lr)
    criterion = nn.MSELoss()

    start = time.time()
 
    loss_total =0
    valid_loss = 0.0
    bad_counter = 0
    best_loss = -1
    best_epoch = 0
    epoch=0
    
    for iloop, (tr_x, tr_y, end_of_data) in enumerate(trainiter):
        tr_x, tr_y = Variable(tr_x).to(device), Variable(tr_y).to(device)
        output, loss = train(model, tr_x, tr_y, optimizer, criterion, args.m2m, args.model)
        loss_total += loss
        
        if end_of_data == 1:
            epoch += 1
            neptune.log_metric('epoch/train_loss', epoch, loss_total/iloop)
            print("%d (%s) %.4f" % (epoch+1, timeSince(start), loss_total/iloop))
            loss_total=0
    
            if (epoch+1) % args.valid_every == 0:
                valid_loss = valid_model(model, validiter, device, criterion, args.m2m, args.model)
                neptune.log_metric('epoch/valid_loss', epoch, valid_loss)
                print("val : %d (%s) %.4f" % (epoch+1, timeSince(start), (valid_loss)))
        
                if valid_loss < best_loss or best_loss < 0:
                    bad_counter = 0
                    torch.save(model, args.out_dir + '/' + args.exp_id)
                    best_loss = valid_loss
                    best_epoch = epoch+1

                else:
                    bad_counter += 1

                if bad_counter > args.patience:
                    print('Early Stopping')
                    break

    neptune.set_property(key='best_epoch', value=best_epoch)
    neptune.set_property(key='valid_loss', value=best_loss.item())
    print('best epoch {:d} {:.4f}'.format(best_epoch, best_loss.item()))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tr_path', type=str, help='', default='./data/eeg_tr.csv')
    parser.add_argument('--val_path', type=str, help='', default='./data/eeg_val.csv')
    parser.add_argument('--batch_size', type=int, help='', default=32)
    parser.add_argument('--lr', type=float, help='', default=0.001)
    parser.add_argument('--optimizer', type=str, help='', default='RMSprop')
    parser.add_argument('--max_epoch', type=int, help='', default=1000)
    #parser.add_argument('--print_every', type=int, help='', default=100)
    parser.add_argument('--valid_every', type=int, help='', default=5)
    parser.add_argument('--model_out_file', type=str, help='', default='./AE_model.pth')
    parser.add_argument('--patience', type=int, help='', default=5)
    
    args = parser.parse_args()

    # temporary code for testing
    train_main(args)
