#!/bin/bash


# Environment information
TEST_ORIG_SRC="../data/OpenSubs18/para/test.en"
TEST_ORIG_TRG="../data/OpenSubs18/para/test.ru"

DEV_ORIG_SRC="../data/OpenSubs18/para/dev.en"
DEV_ORIG_TRG="../data/OpenSubs18/para/dev.ru"


# Parameters
MIN_DOC_SIZE=10
MAX_LEN=60


tmpd=$(mktemp -d _XXXXXX)

paste $TEST_ORIG_SRC $TEST_ORIG_TRG \
    | awk -v MAX=$MAX_LEN -f ./filter_len_dual.awk \
    | tee >(cut -f 1 > $tmpd/test.src) \
    | cut -f 2 > $tmpd/test.trg

paste $DEV_ORIG_SRC $DEV_ORIG_TRG \
    | awk -v MAX=$MAX_LEN -f ./filter_len_dual.awk \
    | tee >(cut -f 1 > $tmpd/dev.src) \
    | cut -f 2 > $tmpd/dev.trg


for f in $tmpd/*; do
    cat $f \
        | awk -v MIN=$MIN_DOC_SIZE -f ./filter_doc_size.awk \
        | awk -f ./del_redundant_empty_lines.awk \
        > ./$(basename $f)
done

awk -v SUFFIX=.src -f ./split_test.awk < test.src
awk -v SUFFIX=.trg -f ./split_test.awk < test.trg
