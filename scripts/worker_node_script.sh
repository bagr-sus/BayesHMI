#!/bin/bash
set -x
head_address=$1
temp_dir=$SCRATCHDIR
echo "tempdir: $temp_dir"
echo `hostname -i`

echo `realpath .`

cd $PBS_O_WORKDIR

echo `realpath .`

singularity instance start bp_simunek.sif contw
echo $(singularity instance list)
#singularity shell instance://contw
signularity exec instance://contw scripts/worker_cont_script.sh $head_address $temp_dir
exit;