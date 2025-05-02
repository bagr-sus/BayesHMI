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

# --- worker node config ---
# pbsdsh to all other nodes and run their scripts
# get head node to exclude it from the worker node list
# head node is the first unique record in $PBS_NODEFILE
command="./scripts/worker_node_script.sh $head_address $SCRATCHDIR"

uniq "$PBS_NODEFILE" | tail -n +2 | while read node; do
    echo "Running worker node script on $node"
    echo "$node"
    echo "$command"
    pbsdsh -h "$node" bash -c "$command" &
done
wait

# --- head node config ---
# create container instance
echo "Starting container on head node"
singularity instance start bp_simunek.sif cont

# run head script inside singularity container
echo "Running head node script"
singularity exec instance://cont scripts/head_node_script.sh $port
