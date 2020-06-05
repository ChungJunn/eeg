# script for training the model

DATA_DIR='/home/chl/eeg/data'

INPUT_DATA_PATH=$DATA_DIR'/qtdbsel102.txt'
TR_OUT_PATH=$DATA_DIR'/eeg_tr.csv'
VAL_OUT_PATH=$DATA_DIR'/eeg_val.csv'
TEST_OUT_PATH=$DATA_DIR'/eeg_test.csv'

python3 eeg_prepare_data.py --input_data_path=$INPUT_DATA_PATH \
    --tr_out_path=$TR_OUT_PATH --val_out_path=$VAL_OUT_PATH \
    --test_out_path=$TEST_OUT_PATH
