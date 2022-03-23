#!/bin/bash

D=$(cd $(dirname $0); pwd)

MULTI_BLEU=~/ubuntu16/mosesdecoder/scripts/generic/multi-bleu.perl

spm_decode --model $D/data/OpenSubs18/ru_sp16k.model \
    | $MULTI_BLEU <(spm_decode --model $D/data/OpenSubs18/ru_sp16k.model < $1)
