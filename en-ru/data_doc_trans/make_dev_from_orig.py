ORIG_SRC = "../data_tm/pure_3ctx/dev.en"
ORIG_TRG = "../data_tm/pure_3ctx/dev.ru"

DEST_SRC = "./dev_from_orig.src"
DEST_TRG = "./dev_from_orig.trg"

with open(ORIG_SRC) as orig_src, open(DEST_SRC, 'w') as dest_src:
    for line in orig_src:
        a = line.split('</s>')
        assert len(a) == 5
        a = '{}</s>{}</s>{}</s>\t<s>{}</s>\n'.format(*a[:4])
        dest_src.write(a)


with open(ORIG_TRG) as orig_trg, open(DEST_TRG, 'w') as dest_trg:
    for line in orig_trg:
        a = line.split('</s>')
        assert len(a) == 5
        a = f'{a[3]}</s>\n'
        dest_trg.write(a)
