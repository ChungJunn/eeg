import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neptune
from neptunecontrib.api import log_chart
import argparse
import sys
from eeg_utils import MultivariateGaussianLikelihood
from eeg_utils import smoothBySlidingWindow as smooth

def forward(x, model, x_avg, x_std, args):
    '''
    args: x: 2D array of data (Tx, D), 
          model: LSTM or AE model
          x_avg: avgs of each features of data (D,)
          x_std: sigmas of each featurese (D,)
    out: data: 2D array of reconstructed data
    '''
    x_ = (x - x_avg) / x_std # _ denotes normalized

    if args.model == 'lstm':
        x_ = x_.view(-1,1, args.dim_input)
        xp_ = model(x_).view(-1,args.dim_input) # x prime
    elif args.model == 'ae':
        xp_ = model(x_)
    
    xp = (xp_ * x_std) + x_avg

    return xp

def test_main(args, neptune):
    # some constants
    error_scaler = 1E8
    ar = [1240, 1460] # anomaly range @200608-03:10
    in_n = args.dim_input

    # load model and obtain some stats
    model = torch.load(args.out_dir + '/' + args.exp_id).to('cpu')

    fp = open(args.stat_file, 'r')
    lines = fp.readlines()
    x_avg= torch.tensor([float(s) for s in lines[0].split(',')])
    x_std= torch.tensor([float(s) for s in lines[1].split(',')])
    fp.close()

    # 1. load test data
    val_data = np.loadtxt(args.val_path, delimiter=',')
    val_data = torch.tensor(val_data).type(torch.float32)
    val_recon = forward(val_data, model, x_avg, x_std, args)    
    val_err = torch.sum((val_recon - val_data) ** 2, dim=1, keepdim=True) # squared error
    ve = val_err * error_scaler    

    test_data = np.loadtxt(args.test_path, delimiter=',')
    test_lbl = test_data[:,-1]; data_len = test_data.shape[0] # retreive labels and length
    test_data = torch.tensor(test_data[:,:-1]).type(torch.float32) 
    test_recon = forward(test_data, model, x_avg, x_std, args) 
    test_err = torch.sum((test_recon - test_data) ** 2, dim=1, keepdim=True) # squared error
    te = test_err * error_scaler

    # 2. measure validation error and test error
    neptune.set_property('validation error', torch.sum(ve).item())
    neptune.set_property('test error', torch.sum(te).item())

    # 2. plot reconstruction results
    cols = ['sensor1', 'sensor2'] # features 
    ids_col = range(test_data.shape[0]) # for index
    
    for j, (data, recon, str) in enumerate([(val_data, val_recon, 'Validation'), (test_data, test_recon, 'Test')]):
        fig, axs = plt.subplots(len(cols),1, figsize=(12,3))
        for i, col in enumerate(cols):
            axs[i].plot(ids_col, data.numpy()[:,i], '-c', linewidth=2, label='Raw Data')
            axs[i].plot(ids_col, recon.detach().numpy()[:,i], '-b', linewidth=1, label='Reconstructed Data')
        axs[1].legend() # only add legend for second row
        fig.suptitle('Time Series of ' + str)
        log_chart('Data-Reconstruction', fig) 
    
    #3. find threshold
    T = (torch.mean(ve) + 2 * torch.std(ve)).item()
    T_ = np.empty((data_len,1)); T_[:] = T # for plotting threshold

    if args.use_smoothing==1: 
        ve = smooth(ve.detach().numpy(), args.window_size)
        te = smooth(te.detach().numpy(), args.window_size) # smoothing removes first window_size samples. (Tx, 1) -> (Tx-args.window_size+1, 1)
   
    pred = (te > T) # pred is  classification result
    pad = np.empty((args.window_size-1,1))
    pad[:] = 0; pred = np.vstack([pad, pred]) # add 0 padding

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(test_lbl, pred); neptune.set_property('acc',acc)
    prec = precision_score(test_lbl, pred); neptune.set_property('prec',prec)
    rec = recall_score(test_lbl, pred); neptune.set_property('rec',rec)
    f1 = f1_score(test_lbl, pred);neptune.set_property('f1',f1)

    # 4. draw plot
    fig = plt.figure(figsize=(24,4))
    ids = list(range(data_len))
    pad[:]=np.nan; te = np.vstack([pad, te]) # add nan padding
    plt.plot(ids[:ar[0]], te[:ar[0]], '-c', label='Test Reconstruction Error (Normal)')
    plt.plot(ids[ar[0]:ar[1]], te[ar[0]:ar[1]], '-r', label='Test Reconstruction Error (Anomaly)')
    plt.plot(ids[ar[1]:], te[ar[1]:], '-c')
    plt.plot(ids, T_, '--b', label='Threshold')
    plt.xlabel('Time'); plt.ylabel('Error');plt.legend()
    #plt.ylim((0,2E5))
    plt.title('Reconstruction Error')
    log_chart('Reconstruction Error', fig)

if __name__ == '__main__':
    #neptune.init('cjlee/sandbox')
    #neptune.create_experiment(name='Autoencoder')
    #neptune.append_tag('run2')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, help='', default='/home/chl/eeg/data/eeg_test.csv')
    parser.add_argument('--val_path', type=str, help='', default='/home/chl/eeg/data/eeg_val.csv')
    parser.add_argument('--stat_file', type=str, help='', default='/home/chl/eeg/data/eeg.stat') 
    parser.add_argument('--exp_id', type=str, help='', default='SAN-305')
    parser.add_argument('--out_dir', type=str, help='', default='/home/chl/eeg/lstm')
    parser.add_argument('--model', type=str, help='', default='lstm')
    parser.add_argument('--use_smoothing', type=int, help='', default=1)
    parser.add_argument('--sigma_scale', type=float, help='', default=8.0)
    parser.add_argument('--window_size', type=int, help='', default=20)
    parser.add_argument('--dim_input', type=int, help='', default=2)
    args = parser.parse_args()

    test_main(args, neptune)
