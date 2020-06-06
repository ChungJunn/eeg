import argparse
import neptune
from eeg_test import test_main

parser = argparse.ArgumentParser()
parser.add_argument('--tr_path', type=str, help='', default='')
parser.add_argument('--val_path', type=str, help='', default='')
parser.add_argument('--test_path', type=str, help='', default='')
parser.add_argument('--stat_file', type=str, help='', default='')

parser.add_argument('--batch_size', type=int, help='')
parser.add_argument('--lr', type=float, help='')
parser.add_argument('--optimizer', type=str, help='')
parser.add_argument('--max_epoch', type=int, help='')
parser.add_argument('--valid_every', type=int, help='')
parser.add_argument('--patience', type=int, help='')

parser.add_argument('--model', type=str, help='{ae, lstm}', default='')
parser.add_argument('--out_dir', type=str, help='', default='')
parser.add_argument('--exp_id', type=str, help='models are saved in exp_id', default='')

parser.add_argument('--dim_input', type=int, help='')
parser.add_argument('--dim_layer', type=int, help='')
parser.add_argument('--dim_z', type=int, help='')

parser.add_argument('--init', type=str, help='', default='')
parser.add_argument('--name', type=str, help='', default='')
parser.add_argument('--tag', type=str, help='', default='')

parser.add_argument('--shuffle', type=int, help='')
parser.add_argument('--m2m', type=int, help='')
parser.add_argument('--rnn_len', type=int, help='')
parser.add_argument('--dim_hidden', type=int, help='')

# smoothing parameter
parser.add_argument('--use_smoothing', type=int, help='')
parser.add_argument('--window_size', type=int, help='')

parser.add_argument('--train', type=int, help='')
parser.add_argument('--test', type=int, help='')

args = parser.parse_args()

if __name__ == '__main__':
    params = vars(args)
    
    import neptune
    neptune.init(args.init)
    exp = neptune.create_experiment(name=args.name, params=params)
    neptune.append_tag(args.tag) 

    if args.model == 'ae':
        from ae.ae_main import train_main
    if args.model == 'lstm':
        from lstm.lstm_main import train_main
    
    if args.train==1:
        args.exp_id = exp._id
        train_main(args, neptune)
    if args.test==1:
        test_main(args, neptune)
