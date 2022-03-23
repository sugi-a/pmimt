for t in test-{1,2,3}; do
    for d in {3,6,15,30}m; do
        echo $t $d
        ../../../decode_and_bleu.sh ../$t.trg < $d/$t.out
    done
done > result
