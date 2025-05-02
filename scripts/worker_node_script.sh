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
singularity shell instance://contw

echo `realpath .`
echo "$SINGULARITY_CONTAINER"
source venv/bin/activate
export RAY_PYTHON=$(which python)
echo $(which python)


#echo $(env | sort | grep RAY_PYTHON)
#"$RAY_PYTHON" -c pip list
#"$RAY_PYTHON" -m pip show ray
#"$RAY_PYTHON" -c "import ray; print(ray.__version__)"
#"$RAY_PYTHON" -m ray.scripts.scripts start --address=$head_address --temp-dir=$temp_dir --num-cpus $PBS_NCPUS
#echo $(env | sort | grep RAY_PYTHON)

exit;
exit;