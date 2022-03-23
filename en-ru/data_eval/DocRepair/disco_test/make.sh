for f in ../../disco_test/tab_sep_scoring_data2/*.src; do
    awk -F '\t' '{print("<s> " $1 " </s>"); print("<s> " $2 " </s>"); print("<s> " $3 " </s>"); print("<s> " $4 " </s>")}' \
        < $f \
        > $(basename $f)
done

for f in ../../disco_test/tab_sep_scoring_data2/*.dst; do
    awk -F '\t'  '{print("<s> " $1 " </s> " $2 " </s> " $3 " </s> " $4 " </s>")}' \
        < $f \
        > $(basename $f .dst).ref
done
