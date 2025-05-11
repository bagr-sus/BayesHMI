#!/bin/bash
mnt=$1
cfg_path=$2
head_address=$3

source venv/bin/activate
export RAY_ADDRESS=$head_address
python -m bp_simunek.scripts.sample $mnt $cfg_path

#cp -r $SCRATCH ~/bayes_output