for f in ../test-*.src; do
    cat $f \
        | awk -v SOS="<s>" -v EOS="</s>" -v N=4 -f ./concat.awk \
        > ./$(basename $f)
done

for f in ../test-*.trg; do
    cat $f \
        | awk -v SOS="<s>" -v EOS="</s>" -v N=4 -f ./wrap_trg.awk \
        > ./$(basename $f)
done
