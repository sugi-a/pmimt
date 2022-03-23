for f in ../sentlv/output/fw_6M/test-?.out; do
    cat $f \
        | awk -f ./concat.awk \
        > ./$(basename $f .out).src
done
