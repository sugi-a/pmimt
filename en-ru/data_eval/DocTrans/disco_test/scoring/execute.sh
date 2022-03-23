function job() {
    for t in deixis_test ellipsis_vp ellipsis_infl lex_cohesion_test; do
        cat ../$t \
            | python -m transformer.applications.doc_trans_zhang2018.inference \
                -d ../../../../DocTrans/$1 \
                --mode logp \
                --debug \
            | awk '{print(-$0)}' \
            > ./${1}_$t
    done
}


CUDA_VISIBLE_DEVICES=4 job orig_6m  &
CUDA_VISIBLE_DEVICES=5 job aided_15m  &
#CUDA_VISIBLE_DEVICES=2 job 15m &
CUDA_VISIBLE_DEVICES=6 job aided_30m &

wait
