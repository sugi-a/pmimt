for t in deixis_test ellipsis_vp ellipsis_infl lex_cohesion_test; do
    for d in da-fw_30M ; do
        echo $t $d
        python ../../../disco_test/good-translation-wrong-in-context/scripts/evaluate_consistency.py \
                --repo-dir ../../../disco_test/good-translation-wrong-in-context \
                --test $t \
                --scores ./${d}_$t
    done
done
