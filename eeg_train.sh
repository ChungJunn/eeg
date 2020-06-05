# script for training the model
GPU=$1
INIT='cjlee/sandbox'
NAME='Autoencoder'
TAG='hidden_dimensions'

TR_PATH='data/eeg_tr.csv'
VAL_PATH='./data/eeg_val.csv'
TEST_PATH='./data/eeg_test.csv'

BATCH_SIZE=32
LR=0.002
OPTIMIZER='RMSprop'
MAX_EPOCH=1000
VALID_EVERY=1
PATIENCE=1

DIM_INPUT=2
DIM_LAYER=32
DIM_Z=2

OUT_DIR='/home/chl/eeg'

STAT_FILE='/data/eeg.stat'

MODEL='ae' # ae or lstm

#MODEL_OUT_FILE=$OUT_DIR'/optim.'$OPTIMIZER'.'$LR'.pth'
MODEL_OUT_FILE=$OUT_DIR'/'$MODEL'/'$OPTIMIZER$LR'.dim'$DIM_Z'.pth'

export CUDA_VISIBLE_DEVICES=$GPU
python3 eeg_run.py --tr_path=$TR_PATH --val_path=$VAL_PATH \
    --test_path=$TEST_PATH \
    --batch_size=$BATCH_SIZE --lr=$LR --optimizer=$OPTIMIZER \
    --max_epoch=$MAX_EPOCH --valid_every=$VALID_EVERY \
    --model_out_file=$MODEL_OUT_FILE --patience=$PATIENCE \
    --init=$INIT --name=$NAME --tag=$TAG \
    --dim_input=$DIM_INPUT --dim_layer=$DIM_LAYER --dim_z=$DIM_Z \
    --model=$MODEL
