for t in test-{1,2,3} test; do
    for d in orig_6m aided_{6,15,30}m; do
        echo $t $d
        ../../../decode_and_bleu.sh ../../$t.trg < $d/$t.out
    done
done > result
