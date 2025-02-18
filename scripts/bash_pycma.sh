#!/bin/bash
mnt=$1
cfg_path=$2

source venv/bin/activate
python -m bp_simunek.scripts.optimize $mnt $cfg_path

cp -r $SCRATCH ~/bayes_output