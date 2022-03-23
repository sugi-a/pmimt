BLEU=../../../decode_and_bleu.sh
for d in da-fw_{3,6,15,30}M fw_6M; do
    for t in test-{1,2,3}; do
        echo $t $d
        $BLEU ../$t.trg < $d/$t.out
    done
    cat $d/test-{1,2,3}.out | tee $d/test.out | $BLEU ../test.trg
done > result
