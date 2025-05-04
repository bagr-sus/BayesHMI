#!/bin/bash

#PBS -S /bin/bash
#PBS -N flow_bayes_multinode
#PBS -q charon

set -x

# get ip addreses of all nodes
NODES=`cat $PBS_NODEFILE`

# ip of head node
head_ip=`hostname -i`

# port to use
port=6379

# combine ip and port
head_address="$head_ip:$port"
echo "$head_address"

# change into workdir
cd "$PBS_O_WORKDIR"


# --- head node config ---
# create container instance

echo `realpath .`

echo "Starting container on head node"
singularity instance start bp_simunek.sif cont

# run head script inside singularity container
echo "Running head node script"
singularity exec instance://cont scripts/head_node_script.sh $port

# --- worker node config ---
# pbsdsh to all other nodes and run their scripts
# get head node to exclude it from the worker node list
# head node is the first unique record in $PBS_NODEFILE

#uniq "$PBS_NODEFILE" | tail -n +2 | while read node; do
#    echo "Running worker node script on $node"
#    echo "$node"
#    pbsdsh -h "$node" bash -c './scripts/worker_node_script.sh "$@"' $head_address &
#done
#wait

nodes=$(cat "$PBS_NODEFILE")
node_count=$(sort -u "$PBS_NODEFILE" | wc -l)
# assuming first node has the same number of cores as the rest
cores_per_node=$(grep -c "$(head -n1 "$PBS_NODEFILE")" "$PBS_NODEFILE")
total_cores=$(($node_count * $cores_per_node))

worker_script_path=`realpath scripts/worker_node_script.sh`

for (( n=$cores_per_node; n<$total_cores; n+=$cores_per_node )); do
    pbsdsh -n $n $worker_script_path $head_address &
    #pbsdsh -n $n 'echo 1'
done
