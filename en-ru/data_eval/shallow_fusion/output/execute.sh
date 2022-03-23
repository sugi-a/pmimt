function job {
    mkdir -p $2
    for f in ../test-?.src; do
        CUDA_VISIBLE_DEVICES=$1 python -m transformer.applications.pmi_fusion.inference_ \
                ../../../TMs/fw_6M \
                ../../../LMs/$2 \
                --beam_size 4 \
                --debug \
                --capacity 8000 \
                --mode trans_shallow_fusion \
                --beta 0.1 \
            < $f \
            > ./$2/$(basename $f .src).out
    done
}


job 4 3m  2> log_0 &
job 5 6m  2> log_1 &
job 6 15m 2> log_2 & 
job 7 30m 2> log_3 &
wait
