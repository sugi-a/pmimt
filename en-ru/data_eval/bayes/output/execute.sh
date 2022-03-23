function job {
    mkdir -p $2
    for f in ../test-?.src; do
        CUDA_VISIBLE_DEVICES=$1 python -m transformer.applications.better_doc_level_trans_w_bayes.inference \
                ../../../TMs/fw_6M \
                ../../../TMs/bw_6M \
                ../../../LMs/$2 \
                2.5 0.5 0.2 \
                --lattice_width 20 \
                --lattice_beam_size 5 \
                --debug \
            < $f \
            | tee ./$2/$(basename $f .src).out \
            | spm_decode --model ../../../data/OpenSubs18/ru_sp16k.model \
            > ./$2/$(basename $f .src).out.tok
    done
}


job 0 3m  2> log_0 &
job 1 6m  2> log_1 &
job 2 15m 2> log_2 & 
job 3 30m 2> log_3 &

wait
