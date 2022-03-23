function job {
    mkdir -p $2
    for f in ../test-?.src; do
        CUDA_VISIBLE_DEVICES=$1 python -m transformer.vanilla.inference \
                --debug \
                --beam_size 4 \
                -d ../../../TMs/$2 \
                --progress 100 \
                --length_penalty 1.0 \
            < $f \
            2> ./log_$1 \
            | tee ./$2/$(basename $f .src).out \
            | spm_decode --model ../../../data/OpenSubs18/ru_sp16k.model \
            > ./$2/$(basename $f .src).out.tok
    done
}


job 0 da-fw_3M &
job 1 da-fw_6M &
job 2 da-fw_15M &
job 3 da-fw_30M &
job 4 fw_6M &

wait
