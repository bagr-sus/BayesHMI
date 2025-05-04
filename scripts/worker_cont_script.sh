#!/bin/bash

set -x
head_address=$1

echo `realpath .`
echo "$SINGULARITY_CONTAINER"
source venv/bin/activate
export RAY_PYTHON=$(which python)
echo $(which python)
#echo $(env | sort | grep RAY_PYTHON)
#"$RAY_PYTHON" -c pip list
#"$RAY_PYTHON" -m pip show ray
#"$RAY_PYTHON" -c "import ray; print(ray.__version__)"
ray start --address=$head_address --num-cpus $PBS_NCPUS
ray -m ray.scripts.scripts status

cat /tmp/ray/session_latest/logs/raylet.err
cat /tmp/ray/session_latest/logs/gcs_server.err

#echo $(env | sort | grep RAY_PYTHON)
