function job() {
    for t in deixis_test ellipsis_vp ellipsis_infl lex_cohesion_test; do
        python -m transformer.applications.better_doc_level_trans_w_bayes.inference \
                ../../../../TMs/fw_6M \
                ../../../../TMs/bw_6M \
                ../../../../LMs/$1 \
                2.5 0.5 0.2 \
                --mode rerank_score \
                --debug \
            < ../$t \
            | awk '{print(-$0)}' \
            > ./${1}_$t
    done
}


CUDA_VISIBLE_DEVICES=0 job 3m  2> log_0 &
CUDA_VISIBLE_DEVICES=1 job 6m  2> log_1 &
CUDA_VISIBLE_DEVICES=2 job 15m 2> log_2 &
CUDA_VISIBLE_DEVICES=3 job 30m 2> log_3 &

wait
