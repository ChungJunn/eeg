# script for training the model
GPU=$1

# neptune
INIT='cjlee/sandbox'

TR_PATH='data/eeg_tr.csv'
VAL_PATH='./data/eeg_val.csv'
TEST_PATH='./data/eeg_test.csv'
OUT_DIR='/home/chl/eeg'
STAT_FILE='/data/eeg.stat'

BATCH_SIZE=64
LR=0.002
OPTIMIZER='RMSprop'
MAX_EPOCH=1000
VALID_EVERY=1
PATIENCE=1

MODEL='lstm' # ae or lstm

# AE params
DIM_INPUT=2
DIM_LAYER=32
DIM_Z=1

# LSTM params
SHUFFLE=1
M2M=1
RNN_LEN=16
DIM_HIDDEN=8

# smoothing
USE_SMOOTHING=1
WINDOW_SIZE=80 
SIGMA_SCALE=16 #TODO

##MUST CHANGE##
NAME='LSTM-redraw-graphics'
#TAG='scale'$SIGMA_SCALE'-0'
TAG='tag'

TRAIN=0
TEST=1
EXP_ID='SAN-305' # for reloading model for test
###############

#MODEL_OUT_FILE=$OUT_DIR'/optim.'$OPTIMIZER'.'$LR'.pth'
OUT_DIR=$OUT_DIR'/'$MODEL

export CUDA_VISIBLE_DEVICES=$GPU
python3 eeg_run.py --tr_path=$TR_PATH --val_path=$VAL_PATH \
    --test_path=$TEST_PATH --stat_file=$STAT_FILE\
    --batch_size=$BATCH_SIZE --lr=$LR --optimizer=$OPTIMIZER \
    --max_epoch=$MAX_EPOCH --valid_every=$VALID_EVERY \
    --out_dir=$OUT_DIR --patience=$PATIENCE \
    --init=$INIT --name=$NAME --tag=$TAG \
    --model=$MODEL --exp_id=$EXP_ID \
    --dim_input=$DIM_INPUT --dim_layer=$DIM_LAYER --dim_z=$DIM_Z \
    --shuffle=$SHUFFLE --m2m=$M2M --rnn_len=$RNN_LEN --dim_hidden $DIM_HIDDEN \
    --use_smoothing=$USE_SMOOTHING --window_size=$WINDOW_SIZE --sigma_scale=$SIGMA_SCALE\
    --train=$TRAIN --test=$TEST
