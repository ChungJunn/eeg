import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neptune
import argparse
import sys
from eeg_utils import MultivariateGaussianLikelihood
from eeg_utils import smoothBySlidingWindow as smooth

def test_main(args, neptune):
    model = torch.load(args.model_out_file).to('cpu')
    in_n = args.dim_input 

    fp = open(args.stat_file, 'r')
    lines = fp.readlines()
    x_avg= torch.tensor([float(s) for s in lines[0].split(',')])
    x_std= torch.tensor([float(s) for s in lines[1].split(',')])
    fp.close()

    # load test data
    test_data = np.loadtxt(args.test_path, delimiter=',')
    test_data = torch.tensor(test_data).type(torch.float32).detach()
    test_inp = (test_data - x_avg) / x_std

    if args.model == 'lstm':
        test_inp = test_inp.view(-1,1,in_n)
        test_recon = model(test_inp).view(-1,in_n)
    elif args.model == 'ae':
        test_recon = model(test_inp)
    
    test_recon = (test_recon * x_std) + x_avg
    test_err = test_recon - test_data

    val_data = np.loadtxt(args.val_path, delimiter=',')
    val_data = torch.tensor(val_data).type(torch.float32)
    val_inp = (val_data - x_avg) / x_std

    if args.model == 'lstm':
        val_inp = val_inp.view(-1,1,in_n)
        val_recon = model(val_inp).view(-1,in_n)
    elif args.model == 'ae':
        val_recon = model(val_inp)
    
    val_recon = (val_recon * x_std) + x_avg
    val_err = val_recon - val_data

    # convert to numpy and apply smoothing
    wsz = 40 # window size
    test_err = smooth(test_err.detach().numpy(), wsz)
    val_err = smooth(val_err.detach().numpy(), wsz)

    # 1. draw normal/reconstruct plot
    cols = ['sensor1', 'sensor2'] # features 
    ids_col = range(test_data.shape[0]) # for index
    
    for j, (data, recon, str) in enumerate([(val_data, val_recon, 'Validation'), (test_data, test_recon, 'Test')]):
        fig, axs = plt.subplots(len(cols),1, figsize=(32,16))
        for i, col in enumerate(cols):
            axs[i].plot(ids_col, data.numpy()[:,i], '-k', linewidth=2)
            axs[i].plot(ids_col, recon.detach().numpy()[:,i], '.b', markersize=1)
            axs[i].set_title(col)
        fig.suptitle('Time Series of ' + str)
        neptune.log_image('plot', fig)
    
    # 2. draw error plot
    pad = np.empty((wsz-1, in_n)); pad[:] = np.nan # padding
    te = np.vstack([pad, test_err]) # test error
    ve = np.vstack([pad, val_err]) # valid error

    for j, (err, str) in enumerate([(ve, 'Validation'), (te, 'Test')]):
        fig, axs = plt.subplots(len(cols),1, figsize=(32,16))
        for i, col in enumerate(cols):
            axs[i].plot(ids_col, err[:,i], '-k', linewidth=1)
            axs[i].set_title(col)
        fig.suptitle('Error of ' + str)
        neptune.log_image('plot', fig)

    # 3. draw likelihood plot
    glf = MultivariateGaussianLikelihood()
    glf.fit(val_err)
    
    ts = glf.gaussian(test_err) # test score
    vs = glf.gaussian(val_err) # val score
    
    pad = np.empty((wsz-1, in_n)); pad[:] = np.nan # padding
    ts = np.vstack([pad, ts]) # test error
    vs = np.vstack([pad, vs]) # valid error

    fig, axs = plt.subplots(len(cols), 1, figsize=(32,16))
    for i, (s, str) in enumerate([(ts, 'test'), (vs, 'validation')]):
        axs[i].plot(ids_col, s, '-k', linewidth=1)
        axs[i].set_title(str)
    fig.suptitle('Likelihood Scores')
    neptune.log_image('plot', fig)

    # error difference measure
    val_sum =  np.sum(np.abs(val_err))
    test_sum = np.sum(np.abs(test_err))
    neptune.set_property('error_difference', (test_sum - val_sum).item())

if __name__ == '__main__':
    neptune.init('cjlee/sandbox')
    neptune.create_experiment(name='Autoencoder')
    neptune.append_tag('run2')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, help='', default='/home/chl/eeg/data/eeg_test.csv')
    parser.add_argument('--val_path', type=str, help='', default='/home/chl/eeg/data/eeg_val.csv')
    parser.add_argument('--stat_file', type=str, help='', default='/home/chl/eeg/data/eeg.stat')
    parser.add_argument('--model_out_file', type=str, help='', default='/home/chl/eeg/lstm/add_fc_layer.pth')
    parser.add_argument('--model', type=str, help='', default='lstm')
    parser.add_argument('--dim_input', type=int, help='', default=2)
    args = parser.parse_args()

    test_main(args, neptune)
