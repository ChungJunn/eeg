import argparse
import neptune
from eeg_test import test_main

parser = argparse.ArgumentParser()
parser.add_argument('--tr_path', type=str, help='', default='./data/eeg_tr.csv')
parser.add_argument('--val_path', type=str, help='', default='./data/eeg_val.csv')
parser.add_argument('--test_path', type=str, help='', default='./data/eeg_test.csv')
parser.add_argument('--batch_size', type=int, help='', default=32)
parser.add_argument('--lr', type=float, help='', default=0.001)
parser.add_argument('--optimizer', type=str, help='', default='RMSprop')
parser.add_argument('--max_epoch', type=int, help='', default=1000)
parser.add_argument('--valid_every', type=int, help='', default=5)
parser.add_argument('--stat_file', type=str, help='', default='./data/eeg.stat')
parser.add_argument('--model_out_file', type=str, help='', default='./AE_model.pth')
parser.add_argument('--patience', type=int, help='', default=5)

parser.add_argument('--dim_input', type=int, help='', default=1)
parser.add_argument('--dim_layer', type=int, help='', default=2)
parser.add_argument('--dim_z', type=int, help='', default=1)

parser.add_argument('--init', type=str, help='', default='')
parser.add_argument('--name', type=str, help='', default='')
parser.add_argument('--tag', type=str, help='', default='')

parser.add_argument('--model', type=str, help='{ae, lstm}', default='ae')


args = parser.parse_args()

if __name__ == '__main__':
    params = vars(args)
    
    import neptune
    neptune.init(args.init)
    neptune.create_experiment(name=args.name, params=params)
    neptune.append_tag(args.tag) 
    
    if args.model == 'ae':
        from ae.ae_main import train_main
    if args.model == 'lstm':
        from lstm.lstm_main import train_main

    train_main(args, neptune)

    test_main(args, neptune)