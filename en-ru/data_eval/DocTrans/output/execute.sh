#!/bin/bash

function job {
    mkdir -p $3
    for f in ../test-?.src; do
        CUDA_VISIBLE_DEVICES=$1 python -m transformer.applications.doc_trans_zhang2018.inference \
                --debug \
                --beam_size 4 \
                -d ../../../DocTrans/$2 \
                --progress_frequency 100 \
                --length_penalty 0 \
                ${@:4} \
            < $f \
            > ./$3/$(basename $f .src).out
    done
}

job 0 orig_6m orig_6m 2> log_0 &
job 1 aided_6m aided_6m 2> log_1 &
job 2 aided_15m aided_15m 2> log_2 &
job 3 aided_30m aided_30m 2> log_3 &

wait
