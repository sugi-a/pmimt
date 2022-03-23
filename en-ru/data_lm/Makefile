SHELL = /bin/bash

# Parameters
MIN_TOKS_PER_DOC = 200
SEP = </s>
SOS = <s>
EOS = </s>


# Paths and names
ORIG_DATA_DIR = ../data/OpenSubs18/mono


# Tools
PIPE_FILTER_LONG_SENT = awk -v MAX=200 -f ./filter_long_sentence.awk
PIPE_FILTER_SHORT_DOC = awk -v MIN_TOKS=200 -f ./filter_short_doc.awk
PIPE_PUT_FRONT_SEP = awk '{if ($$0 == "") print(); else print("$(SEP)", $$0)}'
PIPE_FILTER_MULTI_EMPTY_LINE = awk -f ./filter_redundant_empty_line.awk
PIPE_PUT_DOC_CLOSING = awk '{if ($$0 == "") {print("$(SEP)"); print()} else print($$0)}'


all:;

one_sent_per_line:
	mkdir -p $@
	for f in $(ORIG_DATA_DIR)/{train-*,dev.ru,test.ru}; do \
		cat $$f \
			| $(PIPE_FILTER_LONG_SENT) \
			| $(PIPE_FILTER_SHORT_DOC) \
			| $(PIPE_FILTER_MULTI_EMPTY_LINE) \
			| $(PIPE_PUT_FRONT_SEP) \
			| $(PIPE_PUT_DOC_CLOSING) \
			> $@/$$(basename $$f); \
	done


PIPE_MAKE_DATA_FOR_BT = awk -v EOS='$(EOS)' -v SOS='$(SOS)' -f ./make_data_for_bt.awk

for_bt: one_sent_per_line
	mkdir -p $@
	for f in $</{train-*,dev.ru,test.ru}; do \
		cat $$f \
			| $(PIPE_MAKE_DATA_FOR_BT) \
			> $@/$$(basename $$f); \
	done
