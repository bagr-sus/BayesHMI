#!/bin/bash
set -x
head_address=$1
temp_dir=$SCRATCHDIR
echo "tempdir: $temp_dir"

cd $PBS_O_WORKDIR &&
singularity instance start bp_simunek.sif cont &&
singularity shell instance://cont scripts/worker_node_script.sh &&
source venv/bin/activate
ray start --address=$head_address --temp-dir=$temp_dir --num-cpus $PBS_NCPUS
exit;
exit;