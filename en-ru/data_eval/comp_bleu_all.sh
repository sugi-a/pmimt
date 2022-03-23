#!/bin/bash

REF="./test.trg.tok"

for d in \
        ./sentlv/output/{fw_6M,da-fw_{3,6,15,30}M} \
        ./DocRepair/merged/{3,6,15,30}m/ \
        ./DocTrans/merged/orig_6m/ \
        ./DocTrans/merged/aided_{15,30}m/ \
        ./bayes/output/{3,6,15,30}m \
        ./bayes/output_sent-rerank/{3,6,15,30}m \
        ./pmi_fus/output*/{3,6,15,30}m \
        ./shallow_fusion/output/{3,6,15,30}m \
        ./shallow_fusion/output/output_norm/{3,6,15,30}m \
        ./shallow_fusion/output_no_ctx/{3,6,15,30}m \
        ./shallow_fusion/output_no_ctx/output_norm/{3,6,15,30}m \
    ; do
    echo -n $(./bleu.sh ./test.trg.tok < $d/test.out.tok)
    echo -n -e '\t'
    echo $d
done
