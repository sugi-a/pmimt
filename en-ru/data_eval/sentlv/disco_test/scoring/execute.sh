function job() {
    m=da-fw_30M 
    for t in deixis_test ellipsis_vp ellipsis_infl lex_cohesion_test; do
        python -m transformer.vanilla.inference \
                -d ../../../../TMs/$m \
                --mode logp \
                --debug \
                --capacity 8000 \
            < ../$t \
            | awk '{print(-$0)}' \
            > ./${m}_$t
    done
}


CUDA_VISIBLE_DEVICES=4 job

