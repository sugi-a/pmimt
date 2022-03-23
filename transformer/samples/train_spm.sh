#!/bin/bash -e

V="./vocab.json"

SRC_TRAIN_DATA="./en-fr/train.en"
TRG_TRAIN_DATA="./en-fr/train.fr"
MODEL_PREFIX="spm_shared"

SIZE="$(cat $V | jq '.size')"
PAD_ID="$(cat $V | jq '.PAD_ID')"
SOS_ID="$(cat $V | jq '.SOS_ID')"
EOS_ID="$(cat $V | jq '.EOS_ID')"
UNK_ID="$(cat $V | jq '.UNK_ID')"

USER_SYMBOLS=""


spm_train \
    --input=$SRC_TRAIN_DATA,$TRG_TRAIN_DATA \
    --model_prefix=$MODEL_PREFIX \
    --vocab_size=$SIZE \
    --character_coverage=0.9995 \
    --pad_id=$PAD_ID \
    --bos_id=$SOS_ID \
    --eos_id=$EOS_ID \
    --unk_id=$UNK_ID \
    --user_defined_symbols=$USER_SYMBOLS || { echo 'SPM TRAIN FAILED'; exit 1; }
