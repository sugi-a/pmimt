#!/bin/bash -e

cd $(dirname $0)
TOKENIZE="../../scripts/text/tokenize.sh"

if [ "$1" = "en" ]; then
    $TOKENIZE $1 \
        | spm_encode --model ./en_sp16k.model
elif [ "$1" = "ru" ]; then
    $TOKENIZE $1 \
        | spm_encode --model ./ru_sp16k.model
else
    exit 1
fi
