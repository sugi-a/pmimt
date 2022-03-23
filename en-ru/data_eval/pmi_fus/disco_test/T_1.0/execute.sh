T1=1.0
T2=1.0

function job() {
    for t in deixis_test ellipsis_vp ellipsis_infl lex_cohesion_test; do
        python -m transformer.applications.pmi_fusion.inference \
                ../../../../TMs/fw_6M \
                ../../../../LMs/$1 \
                --mode fscore \
                --debug \
                -T1 $T1 \
                -T2 $T2 \
                --capacity 8000 \
            < ../$t \
            | awk '{print(-$0)}' \
            > ./${1}_$t
        
        python ../../../disco_test/good-translation-wrong-in-context/scripts/evaluate_consistency.py \
                --repo-dir ../../../disco_test/good-translation-wrong-in-context \
                --test $t \
                --scores ./${1}_$t
    done
}


CUDA_VISIBLE_DEVICES=0 job 3m  &
CUDA_VISIBLE_DEVICES=1 job 6m  &
CUDA_VISIBLE_DEVICES=2 job 15m &
CUDA_VISIBLE_DEVICES=3 job 30m &

wait
