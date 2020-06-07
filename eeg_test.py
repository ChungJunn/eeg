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
    model = torch.load(args.out_dir + '/' + args.exp_id).to('cpu')
    in_n = args.dim_input 

    fp = open(args.stat_file, 'r')
    lines = fp.readlines()
    x_avg= torch.tensor([float(s) for s in lines[0].split(',')])
    x_std= torch.tensor([float(s) for s in lines[1].split(',')])
    fp.close()

    # load test data
    test_data = np.loadtxt(args.test_path, delimiter=',')
    test_lbl = test_data[:,-1]
    test_data = torch.tensor(test_data[:,:-1]).type(torch.float32).detach() # last column is labels
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
    if args.use_smoothing == 1: 
        wsz = args.window_size # window size
        test_err = smooth(test_err.detach().numpy(), wsz)
        val_err = smooth(val_err.detach().numpy(), wsz)
    else:
        test_err = test_err.detach().numpy()
        val_err = val_err.detach().numpy()
    
    '''
    # scatter plot
    fig = plt.figure()
    plt.scatter(val_err[:,0], val_err[:,1], marker='o', color='b', label='Validation Error')
    plt.scatter(test_err[:,0], test_err[:,1], marker='o', color='r', label='Test Error')
    plt.title('Scatter Plot of Error Vectors')
    #neptune.log_image('Scatter', fig)
    sys.exit(0)
    '''
    from neptunecontrib.api import log_chart

    # 1. draw normal/reconstruct plot
    cols = ['sensor1', 'sensor2'] # features 
    ids_col = range(test_data.shape[0]) # for index
    
    for j, (data, recon, str) in enumerate([(val_data, val_recon, 'Validation'), (test_data, test_recon, 'Test')]):
        fig, axs = plt.subplots(len(cols),1, figsize=(12,3))
        for i, col in enumerate(cols):
            axs[i].plot(ids_col, data.numpy()[:,i], '-c', linewidth=2, label='Raw Data')
            axs[i].plot(ids_col, recon.detach().numpy()[:,i], '-b', linewidth=1, label='Reconstructed Data')
            axs[i].legend()
        fig.suptitle('Time Series of ' + str)
        log_chart('Data-Reconstruction', fig)
    
    # 2. draw error plot
    if args.use_smoothing == 1: 
        pad = np.empty((wsz-1, in_n)); pad[:] = np.nan # padding
        te = np.vstack([pad, test_err]) # test error
        ve = np.vstack([pad, val_err]) # valid error
    else:
        te = test_err
        ve = val_err

    for j, (err, str) in enumerate([(ve, 'Validation'), (te, 'Test')]):
        fig, axs = plt.subplots(len(cols),1, figsize=(32,16))
        for i, col in enumerate(cols):
            axs[i].plot(ids_col, err[:,i], '-k', linewidth=1)
            axs[i].set_title(col)
        fig.suptitle('Error of ' + str)
        neptune.log_image('plot', fig)

    # 3. draw likelihood plot
    glf = MultivariateGaussianLikelihood()
    glf.fit(val_err, args.sigma_scale)
    
    ts = glf.gaussian(test_err) # test score
    vs = glf.gaussian(val_err) # val score
    
    # retrieve threshold from vs
    T = np.min(vs) # Threshold
    
    pred = (ts < T)
    test_lbl = test_lbl[wsz-1:] # the first part should be gone

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score, recall_score
    
    acc = accuracy_score(test_lbl, pred)
    prec = precision_score(test_lbl, pred)
    rec = recall_score(test_lbl, pred)
    
    neptune.set_property('acc', acc)
    neptune.set_property('prec', prec)
    neptune.set_property('rec', rec) 

    if args.use_smoothing == 1: 
        pad = np.empty((wsz-1,)); pad[:] = np.nan # padding. score is scalar
        print('pad shape', pad.shape)
        print('test shape', ts.shape)
        ts = np.concatenate([pad, ts]) # test error
        vs = np.concatenate([pad, vs]) # valid error

    fig, axs = plt.subplots(len(cols), 1, figsize=(12,3))
    for i, (s, str) in enumerate([(vs, 'validation'),(ts, 'test')]):
        axs[i].plot(ids_col, s, '-g', linewidth=1, label='Likelihood Score')
        axs[i].legend()
    fig.suptitle('Likelihood Scores(valid and test respectively)')
    log_chart('Liklihood-scores', fig)

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
    parser.add_argument('--exp_id', type=str, help='', default='SAN-269')
    parser.add_argument('--out_dir', type=str, help='', default='/home/chl/eeg/lstm')
    parser.add_argument('--model', type=str, help='', default='lstm')
    parser.add_argument('--use_smoothing', type=int, help='', default=1)
    parser.add_argument('--sigma_scale', type=float, help='', default=8.0)
    parser.add_argument('--window_size', type=int, help='', default=80)
    parser.add_argument('--dim_input', type=int, help='', default=2)
    args = parser.parse_args()

    test_main(args, neptune)
