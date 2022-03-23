ORIG_SRC="./orig_train-000.src" 
ORIG_TRG="./orig_train-000.trg" 

if [ ! -e $ORIG_SRC ] || [ ! -e $ORIG_TRG ]; then
    echo "error" >&2
    exit 1
fi

N=5000
AWKSCRIPT='{if(NR <= N) print($0) > "/dev/stderr"; else print($0)}' 

awk -v N=$N "$AWKSCRIPT" < $ORIG_SRC 1> ./train-000.src 2> ./dev.src
awk -v N=$N "$AWKSCRIPT" < $ORIG_TRG 1> ./train-000.trg 2> ./dev.trg
