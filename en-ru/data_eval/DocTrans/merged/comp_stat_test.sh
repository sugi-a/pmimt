#!/bin/bash

TEST="../../paired-bootstrap.py"
N=1000
GOLD=../../test.trg.tok

tempd=$(mktemp -d _XXXXXX)

for k in 6 15 30; do
    sentlv="../../sentlv/output/da-fw_${k}M/test.out.tok"
    self="aided_${k}m/test.out.tok"
    echo $sentlv vs $self > $tempd/$k
    python $TEST --eval_type bleu --num_samples $N $GOLD $sentlv $self >> $tempd/$k &
done
wait
cat $tempd/*

rm -r $tempd
