#!/bin/bash

CD=$(dirname $0)
MODEL="$CD/../data/OpenSubs18/ru_sp16k.model"
spm_decode --model $MODEL
