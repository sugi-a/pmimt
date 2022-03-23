function job {
    CUDA_VISIBLE_DEVICES=$1 python -m transformer.vanilla.inference \
        --debug \
        --beam_size 4 \
        -d ../../../../TMs/fw_6M \
        --progress 100 \
        --length_penalty 1.0 \
        < $2 \
        > ./$(basename $2 .src).out
}

job 0 ../deixis_test.src       2> log_0 &
job 1 ../ellipsis_infl.src     2> log_1 &
job 2 ../ellipsis_vp.src       2> log_2 &
job 3 ../lex_cohesion_test.src 2> log_3 &

wait
