for f in ../{test-*,dev.*}; do
    awk -f ./wrap.awk < $f > ./$(basename $f)
done
