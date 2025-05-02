#!/bin/bash
set -x
head_address=$1
temp_dir=$SCRATCHDIR
echo "tempdir: $temp_dir"
echo `hostname -i`

cd $PBS_O_WORKDIR
singularity instance start bp_simunek.sif cont
singularity shell instance://cont scripts/worker_node_script.sh 
source venv/bin/activate
export RAY_PYTHON=$(which python)
echo $(ls)
#echo $(env | sort | grep RAY_PYTHON)
"$RAY_PYTHON" -c pip list
"$RAY_PYTHON" -m pip show ray
"$RAY_PYTHON" -c "import ray; print(ray.__version__)"
#"$RAY_PYTHON" -m ray.scripts.scripts start --address=$head_address --temp-dir=$temp_dir --num-cpus $PBS_NCPUS
#echo $(env | sort | grep RAY_PYTHON)
exit;
exit;