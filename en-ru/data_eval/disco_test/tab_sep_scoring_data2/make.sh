ENC_SRC="spm_encode --model ../../../data/OpenSubs18/en_sp16k.model"
ENC_TRG="spm_encode --model ../../../data/OpenSubs18/ru_sp16k.model"

for f in ../good-translation-wrong-in-context/consistency_testsets/scoring_data/*.src; do
    sed -r -e 's| _eos |\t|g' < $f > ./_temp
    paste \
            <(cut -f 1 ./_temp | $ENC_SRC) \
            <(cut -f 2 ./_temp | $ENC_SRC) \
            <(cut -f 3 ./_temp | $ENC_SRC) \
            <(cut -f 4 ./_temp | $ENC_SRC) \
        > ./$(basename $f)
done

for f in ../good-translation-wrong-in-context/consistency_testsets/scoring_data/*.dst; do
    sed -r -e 's| _eos |\t|g' < $f > ./_temp
    paste \
            <(cut -f 1 ./_temp | $ENC_TRG) \
            <(cut -f 2 ./_temp | $ENC_TRG) \
            <(cut -f 3 ./_temp | $ENC_TRG) \
            <(cut -f 4 ./_temp | $ENC_TRG) \
        > ./$(basename $f)
done
