function job {
    mkdir -p $1
    for f in ../test-?.src; do
        python -m transformer.vanilla.inference \
                --debug \
                --beam_size 4 \
                -d ../../../DocReps/$1 \
                --progress 100 \
            < $f \
            > ./$1/$(basename $f .src).out
    done
}


CUDA_VISIBLE_DEVICES=0 job  3m  2> log_0 &
CUDA_VISIBLE_DEVICES=1 job  6m  2> log_1 &
CUDA_VISIBLE_DEVICES=2 job  15m 2> log_2 & 
CUDA_VISIBLE_DEVICES=3 job  30m 2> log_3 &

wait
