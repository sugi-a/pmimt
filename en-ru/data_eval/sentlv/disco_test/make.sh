for label in deixis_test lex_cohesion_test ellipsis_infl ellipsis_vp; do
    src=../../disco_test/tab_sep_scoring_data2/$label.src
    dst=../../disco_test/tab_sep_scoring_data2/$label.dst

    paste \
            <(awk -F '\t' '{print("<s> " $4 " </s>")}' < $src) \
            <(awk -F '\t' '{print("<s> " $4 " </s>")}' < $dst) \
        > ./$label
done
