function job() {
    for t in deixis_test ellipsis_vp ellipsis_infl lex_cohesion_test; do
        paste ../aligned/$t.out ../$t.ref \
            | python -m transformer.vanilla.inference \
                -d ../../../../DocReps/$1 \
                --mode logp \
                --debug \
            | awk '{print(-$0)}' \
            > ./${1}_$t
    done
}


CUDA_VISIBLE_DEVICES=0 job 3m  &
CUDA_VISIBLE_DEVICES=1 job 6m  &
CUDA_VISIBLE_DEVICES=2 job 15m &
CUDA_VISIBLE_DEVICES=3 job 30m &

wait
