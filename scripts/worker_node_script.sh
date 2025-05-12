#!/bin/bash
set -x
head_address=$1
cfg_path=$2

echo `hostname -i`

echo `realpath .`

cd $PBS_O_WORKDIR

echo `realpath .`

singularity instance start bp_simunek.sif contw
echo $(singularity instance list)
#singularity shell instance://contw
singularity exec instance://contw scripts/worker_cont_script.sh $head_address $cfg_path

exit;