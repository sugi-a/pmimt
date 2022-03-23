SHELL = /bin/bash

ORIG_DIR = ../data/OpenSubs18/para
SL = en
TL = ru

MAXLEN = 200
SEP = </s>

CONC = awk -f ./concat.awk
WRAP = awk -f ./wrap.awk
DUAL_FILTER = awk -F '\t' 'split($$1,a," ") < $(MAXLEN) && split($$2,a," ") < $(MAXLEN)'
LEN_DIFF_FILTER = awk -v SMOOTH=3 -v RATIO=1.4 -f ./len_diff_filter.awk

all:;

mixed_3ctx:
	mkdir -p $@
	for f in $(ORIG_DIR)/{test.*,dev.*,train-*}; do \
		$(CONC) < $$f | $(WRAP) > $@/$$(basename $$f); \
	done
	tmp=$$(mktemp _XXXXXX); \
	for f in $@/*.$(SL); do \
		pfx=$${f%.$(SL)}; \
		paste $${pfx}.$(SL) $${pfx}.$(TL) \
			| $(DUAL_FILTER) \
			| $(LEN_DIFF_FILTER) \
			> $$tmp ; \
		cut -f 1 $$tmp > $${pfx}.$(SL); \
		cut -f 2 $$tmp > $${pfx}.$(TL); \
	done; \
	rm $$tmp

# Contains 3 context sentences
pure_3ctx: mixed_3ctx
	mkdir -p $@
	for f in $</train-*; do \
		cat $$f \
			| awk -F "$(SEP)" 'NF==5' \
			> $@/$$(basename $$f); \
	done
	for f in $</{dev,test}.*; do \
		cat $$f \
			| awk -F "$(SEP)" 'NF==5' \
			> $@/$$(basename $$f); \
	done


single_from_pure_3ctx: pure_3ctx
	mkdir -p $@
	for f in $</{test.*,dev.*,train-*}; do \
		cat $$f \
			| awk -F "$(SEP)" '{print("<s>" $$4 "</s>")}' \
			> $@/$$(basename $$f); \
	done


