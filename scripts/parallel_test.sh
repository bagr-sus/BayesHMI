#!/bin/bash

#PBS -S /bin/bash
#PBS -N flow_bayes_multinode
#PBS -q charon

set -x

id=$(echo "$PBS_JOBID" | cut -d'.' -f1)
echo $id

hash=$(echo "$id" | cksum)
echo $hash

link="${HOME}"/"${hash:0:3}"
echo $link

if [ ! -f "${link}" ] ; then
    rm -r "${link}"
fi
ln -s $SCRATCHDIR $link

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
singularity exec instance://cont scripts/head_node_script.sh $port $link

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

# workers are blocked, so wait some time for them to start
sleep 30

cfg_path=$CFG_PATH

SCRATCHDIR=$SCRATCHDIR  # Your temporary working directory
TARGETDIR=$HOME/bayes_output/$PBS_JOBID  # Persistent backup location

mkdir -p $TARGETDIR

# Start periodic sync in the background
while true; do
    rsync -av --delete $SCRATCHDIR/ $TARGETDIR/
    sleep 600  # Wait 5 minutes before syncing again
done &
SYNC_PID=$!  # Store the process ID of the background sync

#cat "$(ls -la $SCRATCHDIR)"
#cd $SCRATCHDIR

cd $PBS_O_WORKDIR # potential cause of symlinks being overwritten
singularity exec bp_simunek.sif bash scripts/singularity_run_script.sh "${link}" "${cfg_path}"

# Kill the sync process when the job completes
kill $SYNC_PID
wait $SYNC_PID 2>/dev/null  # Ensure it's fully terminated

# Perform a final sync to capture the last changes
rsync -av $SCRATCHDIR/ $TARGETDIR/

rm -r $link

