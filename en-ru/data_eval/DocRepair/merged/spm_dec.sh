for d in ./{3,6,15,30}m; do
    for f in $d/*.out; do
        echo $f
        spm_decode --model ../../../data/OpenSubs18/ru_sp16k.model < $f > $f.tok
    done
done
