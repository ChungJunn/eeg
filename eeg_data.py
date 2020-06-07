import numpy as np
import torch
import random

class EEG_RNNIterator:
    def __init__(self, filename, batch_size=32, rnn_len=10, shuffle=False, m2m=False, args=None, train=True):
        self.b_size = batch_size
        self.rnn_len = rnn_len
        self.m2m = m2m

        in_fp = open(filename, 'r')
        in_lines = in_fp.readlines()
        self.in_n = len(in_lines[0].split(',')) # remove -1 for date
        self.data_len = len(in_lines)
        self.in_nums = np.asarray([[float(s) for s in line.split(',')[:]] for line in in_lines]) 

        if train:
            self.x_avg = np.mean(self.in_nums, axis=0)
            self.x_std = np.std(self.in_nums, axis=0)
            fp = open(args.stat_file, 'w')
            for i in range(self.x_avg.shape[0]):
                if i > 0:
                    fp.write(', ')
                fp.write('%.9f' % (self.x_avg[i]))
            fp.write('\n')
            for i in range(self.x_std.shape[0]):
                if i > 0:
                    fp.write(', ')
                fp.write('%.9f' % (self.x_std[i]))
            fp.write('\n')
            fp.close() 
        else:
            fp = open(args.stat_file, 'r')
            lines = fp.readlines()
            self.x_avg= np.asarray([float(s) for s in lines[0].split(',')])
            self.x_std= np.asarray([float(s) for s in lines[1].split(',')])
            fp.close()

        self.ids = list(range(self.data_len - self.rnn_len))
        if shuffle:
            random.shuffle(self.ids)
        self.ids_i = 0

    def __iter__(self):
        return self

    def reset(self):
        self.ids_i = 0

    def __next__(self):
        x_data = np.zeros((self.rnn_len, self.b_size, self.in_n))
        end_of_data=0
        b_len=0
        idx=0

        for i in range(self.b_size):
            if self.ids_i >= (self.data_len - self.rnn_len - 1):
                end_of_data=1
                self.reset()

            idx = self.ids[self.ids_i]
            
            data_range = range(idx, idx + self.rnn_len)
            x_data[:,i,:] = self.in_nums[data_range, :]
            b_len +=1
            self.ids_i += 1

            if end_of_data==1: break

        x_data = x_data[:, :b_len, :] 
        x_data = self.prepare_data(x_data)
        
        if self.m2m:
            y_data = np.empty_like(x_data)
            y_data = x_data[:, :b_len, :]
        else:
            y_data = np.empty_like(x_data[-1,:b_len,:])
            y_data = x_data[-1,:b_len, :] 
        
        x_data, y_data = torch.tensor(x_data).type(torch.float32), torch.tensor(y_data).type(torch.float32)
        return x_data, y_data, end_of_data # (T B E), (B E)

    def prepare_data(self, in_seq):
        #seq = (seq[:,:-1] + seq[:,1:])/2.0
        #seq_delta = seq[:,1:] - seq[:,:-1]
    
        x_data = (in_seq-self.x_avg)/self.x_std 
        return x_data

class EEG_AEIterator: 
    def __init__(self, filename, batch_size=32, train=True, args=None):
        self.b_size = batch_size # batch size?
        self.end_of_data = 0 # marks end_of_data

        in_fp = open(filename, 'r') # read input file
        in_lines = in_fp.readlines()
        self.in_n = len(in_lines[0].split(','))
        self.data_len = len(in_lines)
        self.in_nums = np.asarray([[float(s) for s in line.split(',')[:]] for line in in_lines]) 

        if train:
            self.x_avg = np.mean(self.in_nums, axis=0)
            self.x_std = np.std(self.in_nums, axis=0)
            fp = open(args.stat_file, 'w')
            for i in range(self.x_avg.shape[0]):
                if i > 0:
                    fp.write(', ')
                fp.write('%.9f' % (self.x_avg[i]))
            fp.write('\n')
            for i in range(self.x_std.shape[0]):
                if i > 0:
                    fp.write(', ')
                fp.write('%.9f' % (self.x_std[i]))
            fp.write('\n')
            fp.close() 
        
        else:
            fp = open(args.stat_file, 'r')
            lines = fp.readlines()
            self.x_avg= np.asarray([float(s) for s in lines[0].split(',')])
            self.x_std= np.asarray([float(s) for s in lines[1].split(',')])
            fp.close()

        self.idx = 0

    def __iter__(self):
        return self

    def reset(self):
        self.idx = 0

    def __next__(self):
        x_data = np.zeros((self.b_size, self.in_n)) # T B E
        end_of_data = 0

        if self.idx >= self.data_len:
            self.reset()
            end_of_data=1
            
        b_len = 0
        for i in range(self.b_size):
            if self.idx+i >= self.data_len: 
                break

            x_data[i,:] = self.in_nums[self.idx+i, :]
            b_len += 1

        x_data = x_data[:b_len, :]
        self.idx += self.b_size

        x_data = self.prepare_data(x_data)
        y_data = np.empty_like(x_data); y_data[:] = x_data # make deep copy

        x_data, y_data = torch.tensor(x_data).type(torch.float32), torch.tensor(y_data).type(torch.float32)
        return x_data, y_data, end_of_data # B E

    def prepare_data(self, in_seq):
        x_data = (in_seq-self.x_avg)/self.x_std 
        return x_data

if __name__ == "__main__":
    import os
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--stat_file", type=str, default='./data/eeg.stat')
    args = parser.parse_args()

    in_file = args.data_dir + '/eeg_tr.csv'
    bs = 4
    rlen = 1
    train_iter = EEGIterator(in_file, batch_size=bs, args=args, train=True)

    i = 0
    for epoch in range(3):
        for tr_x, tr_y, end_of_data in train_iter:
            i = i + 1
            if i==1:
                print('main', epoch, i * bs, tr_x.shape, tr_y.shape, tr_x, tr_y)
            
            if i == 1000:
                print('main', epoch, i * bs, tr_x.shape, tr_y.shape, tr_x, tr_y)
            
            if i == 2438:
                print('main', epoch, i * bs, tr_x.shape, tr_y.shape, tr_x, tr_y)
                break

        if epoch >= 0: 
            break
