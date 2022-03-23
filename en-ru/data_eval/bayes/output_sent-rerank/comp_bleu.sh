BLEU=../../../decode_and_bleu.sh 
for d in {3,6,15,30}m; do
    for t in test-{1,2,3}; do
        echo $t $d
        $BLEU ../../$t.trg < $d/$t.out
    done
    echo Test
    cat $d/test-{1,2,3}.out | tee $d/test.out | $BLEU ../../test.trg
done > result
