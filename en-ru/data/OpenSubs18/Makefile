SHELL = /bin/bash

# Languages
SL = en
TL = ru

# Position in OpenSubs18
SP = 1
TP = 2
LL = en
RL = ru

# Commands
SHUF = shuf --random-source <(while :; do echo 1;done)
SCRIPTS = ../../scripts
MULTI_PROCESS = $(SCRIPTS)/text/parallelize_pipe.sh
TOKENIZE = $(SCRIPTS)/text/tokenize.sh
SCATTER = python $(SCRIPTS)/text/evenly_split_docs.py
PREPROCESS = ./preprocess.sh

# Config file
CONFIG = ./config.json
JQC := cat $(CONFIG) | jq

# Put the data on Google Drive here
ORIG_MONO = orig/mono-txt-$(TL)
ORIG_PARA = orig/para-txt_ctx-$(LL)-$(RL)
ORIG_PARA_DENSE = orig/para-txt-$(LL)-$(RL)

ORIG_TEST_LIST = orig/en-ru_disco_testset_en

# Options
N_TEST_FILES = 500
N_TEST_FILES_MIN = 50
N_DEV_FILES = 50
N_SPM_TRAIN_FILES = 50000 # Avg #lines ~ 300
N_MONO_SCATTER = 50
N_PARA_SCATTER = 30

# Path to the target files and directories
SPM_SRC_PFX := $(shell $(JQC) -r '.spm.src_prefix')
SPM_TRG_PFX := $(shell $(JQC) -r '.spm.trg_prefix')


all: $(shell echo {mono,para,para_dense}/{train-000,dev,test}.$(TL));

test_file_list_mono:
	cat $(ORIG_TEST_LIST) \
		| sed -r -e 's|^en/(.+)$$|\1|' \
		| sed -r -e 's|/|-|g' \
		> $@

test_file_list_para: test_file_list_mono
	find $(ORIG_PARA) -type f -name '*.txt' \
		| sed -r -e 's|^.*/(.+)\.txt$$|\1|' \
		| grep -F -f <(awk '{print($$0 "_")}' < $<) \
		> $@

train_file_list_para: test_file_list_para
	find $(ORIG_PARA) -type f -name '*.txt' \
		| sed -r -e 's|^.*/(.+)\.txt$$|\1|' \
		| grep -v -F -f $< > $@

train_file_list_mono: test_file_list_mono
	find $(ORIG_MONO) -type f -name '*.txt' \
		| sed -r -e 's|^.*/(.+)\.txt$$|\1|' \
		| grep -v -F -f $< > $@

spm_train_file_list: train_file_list_para
	$(SHUF) -n $(N_SPM_TRAIN_FILES) < $< \
		> $@
	
spm_train_data_src: spm_train_data_trg;
spm_train_data_trg: spm_train_file_list
	cat $< \
		| awk '{print("$(ORIG_PARA_DENSE)/" $$0 ".txt")}' \
		| xargs -P 1 -n 100 cat \
		| tee >(cut -f $(SP) \
			| NPROC=8 $(MULTI_PROCESS) $(TOKENIZE) $(SL) \
			> spm_train_data_src) \
		| cut -f $(TP) \
		| NPROC=8 $(MULTI_PROCESS) $(TOKENIZE) $(TL) \
		> spm_train_data_trg

define _train_spm
	spm_train \
		--input=$< \
		--model_prefix=$1 \
		--vocab_size=$$($(JQC) '.spm.vocab_size') \
		--character_coverage=$2 \
		--pad_id=$$($(JQC) '.spm.pad_id') \
		--bos_id=$$($(JQC) '.spm.bos_id') \
		--eos_id=$$($(JQC) '.spm.eos_id') \
		--unk_id=$$($(JQC) '.spm.unk_id') \
		--user_defined_symbols=$$($(JQC) -r '.spm.user_defined_symbols[]' \
			| awk 'NR!=1{printf(",")} 1{printf($$0)}') \
		--control_symbols=$$($(JQC) -r '.spm.custom_control_symbols[]' \
			| awk 'NR!=1{printf(",")} 1{printf($$0)}')
endef

$(SPM_SRC_PFX).model:
	cp $$($(JQC) -r '.spm.src_origin').* ./
	#$(call _train_spm,$(shell $(JQC) '.spm.src_prefix'),$(shell $(JQC) .spm.src_char_coverage))

$(SPM_TRG_PFX).model: spm_train_data_trg
	$(call _train_spm,$(shell $(JQC) '.spm.trg_prefix'),$(shell $(JQC) .spm.trg_char_coverage))


mono/test_list: test_file_list_mono
	mkdir -p mono
	head -n $(N_TEST_FILES_MIN) $< > $@

mono/dev_list: mono/train_list;
mono/train_list: train_file_list_mono
	mkdir -p mono
	cat $< | $(SHUF) -n $(N_DEV_FILES) > mono/dev_list
	cat $< | grep -v -F -f mono/dev_list > mono/train_list

para/test_list: test_file_list_para
	mkdir -p para
	head -n $(N_TEST_FILES_MIN) $< > $@

para/dev_list: para/train_list;
para/train_list: train_file_list_para
	mkdir -p para
	cat $< | $(SHUF) -n $(N_DEV_FILES) > para/dev_list
	cat $< | grep -v -F -f para/dev_list > para/train_list


mono/%.$(TL): mono/%_list $(SPM_TRG_PFX).model
	cat $< \
		| awk '{print("$(ORIG_MONO)/" $$0 ".txt")}' \
		| xargs -P 1 -n 100 awk 'FNR==1 && NR > 1{print ""}; 1' \
		| NPROC=8 $(MULTI_PROCESS) $(PREPROCESS) $(TL) \
		> $@

para/%.$(SL): para/%.$(TL) $(SPM_SRC_PFX).model;
para/%.$(TL): para/%_list $(SPM_TRG_PFX).model
	cat $< \
		| awk '{print("$(ORIG_PARA)/" $$0 ".txt")}' \
		| xargs -P 1 -n 100 awk 'FNR==1 && NR > 1{print ""}; 1' \
		| tee >( \
			cut -f $(SP) \
			| NPROC=8 $(MULTI_PROCESS) $(PREPROCESS) $(SL) \
			> para/$*.$(SL)) \
		| cut -f $(TP) \
		| NPROC=8 $(MULTI_PROCESS) $(PREPROCESS) $(TL) \
		> para/$*.$(TL)

# Scatter
mono/train-%: mono/train-000.$(TL)
mono/train-000.$(TL): mono/train.$(TL)
	$(SCATTER) $(N_MONO_SCATTER) mono/train- < $<

para/train-%: para/train-000.$(TL)
para/train-000.$(TL): para/train.$(TL) para/train.$(SL)
	paste para/train.{$(SL),$(TL)} \
		| $(SCATTER) $(N_PARA_SCATTER) para/_train_
	for f in para/_train_*; do \
		cut -f $(SP) < $$f > para/train-$${f##*_}.$(SL); \
		cut -f $(TP) < $$f > para/train-$${f##*_}.$(TL); \
	done
	rm para/_train_*

para_dense/%: para_dense/train-000.$(TL);
para_dense/train-000.$(TL): para/train-000.$(TL) para/dev.$(TL) para/test.$(TL)
	mkdir -p para_dense
	for f in para/{dev.*,test.*,train-*}; do \
		grep -v '^$$' $$f > para_dense/$$(basename $$f); \
	done
