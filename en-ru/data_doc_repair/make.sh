#!/bin/bash

SOS="<s>"
EOS="</s>"
N=4

ORIG_SRC_DIR=../round_trip_data
ORIG_TRG_DIR=../data_lm/one_sent_per_line

tmpd=$(mktemp -d _XXXXXX)

mkdir $tmpd/trg_naked
for f in $ORIG_TRG_DIR/train-*; do
    awk -f ./del_first_tok.awk < $f > $tmpd/trg_naked/$(basename $f) &
done
wait

mkdir $tmpd/min_filtered
for srcf in $ORIG_SRC_DIR/train-*; do
    bname=$(basename $srcf)
    trgf=$tmpd/trg_naked/$bname

    paste $srcf $trgf \
        | awk -f ./min_len_filter_dual.awk \
        | tee >(cut -f 1 > $tmpd/min_filtered/$bname.src) \
        | cut -f 2 > $tmpd/min_filtered/$bname.trg &
done
wait

mkdir $tmpd/concat

for f in $tmpd/min_filtered/train-*; do
    cat $f \
        | awk -v SOS="$SOS" -v EOS="$EOS" -v N=$N -f ./concat.awk \
        > $tmpd/concat/$(basename $f) &
done
wait

for srcf in $tmpd/concat/train-*.src; do
    bname=$(basename $srcf .src)
    trgf=$tmpd/concat/$bname.trg

    paste $srcf $trgf \
        | awk -v MAX=512 -f ./max_len_filter_dual.awk \
        | tee >(cut -f 1 > ./$bname.src) \
        | cut -f 2 > ./$bname.trg &
done
wait

