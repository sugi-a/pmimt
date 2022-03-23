for d in orig_6m aided_{6,15,30}m; do
    mkdir -p $d
    for i in 1 2 3; do
        a="../output/$d/test-$i.out"
        b="../../sentlv/output/fw_6M/test-$i.out"
        echo "$(wc -l < $a)"  "$(wc -l < $b)"

        paste $a $b  \
            | awk -f ./merge.awk \
            > $d/test-$i.out
    done
    cat $d/test-*.out > $d/test.out
done
