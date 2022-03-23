function job {
    mkdir -p $2
    for f in ../test-?.src; do
        CUDA_VISIBLE_DEVICES=$1 python -m transformer.applications.better_doc_level_trans_w_bayes.inference \
                ../../../TMs/fw_6M \
                ../../../TMs/bw_6M \
                ../../../LMs/$2 \
                1.5 0.5 0.8 \
                --n_ctx 0 \
                --lattice_width 20 \
                --lattice_beam_size 5 \
                --debug \
            < $f \
            > ./$2/$(basename $f .src).out
    done
}


job 4 3m  2> log_0 &
job 5 6m  2> log_1 &
job 6 15m 2> log_2 & 
job 7 30m 2> log_3 &

wait
