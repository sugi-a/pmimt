function job {
    mkdir -p $2
    for f in ../test-?.src; do
        CUDA_VISIBLE_DEVICES=$1 python -m transformer.applications.pmi_fusion.inference \
                ../../../TMs/fw_6M \
                ../../../LMs/$2 \
                --beam_size 4 \
                --debug \
                --capacity 8000 \
                -T1 8 \
                -T2 8 \
            < $f \
            > ./$2/$(basename $f .src).out
    done
}


job 2 3m  2> log_0 &
job 3 6m  2> log_1 &
job 4 15m 2> log_2 & 
job 5 30m 2> log_3 &
wait
