#!/bin/bash

echo 'Usage: ./check_config_structure.sh spec.ts instance.json'

([ -n "$1" ] && [ -n "$2" ]) || { echo 'Invalid arguments'; exit 1; };

tmpd=$(mktemp -d tmp_XXXXXX)
tmp=$tmpd/a.ts

{
    cat $1
    echo 'const x: Config = '
    cat $2
    echo ';'
} > $tmp

npx tsc --strict $tmpd/a.ts \
    && echo 'Succeeded!' \
    || echo 'Illegal Config Structure!'

rm -r $tmpd