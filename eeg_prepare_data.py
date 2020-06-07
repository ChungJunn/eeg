import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', type=str, help='', default='')
    parser.add_argument('--tr_out_path', type=str, help='', default='')
    parser.add_argument('--val_out_path', type=str, help='', default='')
    parser.add_argument('--test_out_path', type=str, help='', default='')
    args = parser.parse_args()
    
    val_test_split = 3000
    test_tr_split = 6000

    data = np.loadtxt(args.input_data_path)
    data_len = data.shape[-1] - 1 # exclude first column

    val_data = data[:val_test_split,1:].reshape(-1, data_len)
    test_data = data[val_test_split:test_tr_split,1:].reshape(-1, data_len)
    tr_data = data[test_tr_split:,1:].reshape(-1, data_len)

    # add labels for test data
    anomaly_range = range(1240, 1461) # anomaly range is [4240, 4460]
    lbl = np.zeros((test_data.shape[0], 1))
    lbl[anomaly_range] = 1

    test_data = np.hstack([test_data, lbl])

    np.savetxt(args.tr_out_path, tr_data, delimiter=',')
    np.savetxt(args.val_out_path, val_data, delimiter=',')
    np.savetxt(args.test_out_path, test_data, delimiter=',')
