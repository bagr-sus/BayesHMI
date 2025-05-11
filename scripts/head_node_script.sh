#!/bin/bash

#set -x

echo `realpath .`

# get head address from input
#head_address=$1

# get port from input
port=$1

# get temp dir from input
temp_dir=$2

# get worker node addresses from input
#NODES=$3

# get pass to use with ssh
#pass=$4

# get a copy of $PBS_O_WORKDIR, other nodes seem to not know its value
#workdir=$(echo $PBS_O_WORKDIR)
#echo "workdir $workdir"

# activate venv
source venv/bin/activate

# start ray on head node
echo "Starting Ray on head node"
ray start --head --port=$port --num-cpus=$PBS_NCPUS --temp-dir=$temp_dir
#python -m bp_simunek.scripts.ray_hack $port $PBS_NCPUS
sleep 5

# # install sshpass, later wont be neccessary
# #echo apt-get install sshpass

# # Generate SSH key if it doesn't exist
# if [ ! -f ~/.ssh/id_ed25519 ]; then
#     ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
# fi

# # Distribute SSH key to all allocated nodes
# for NODE in $NODES; do
#     ssh-copy-id -o StrictHostKeyChecking=no "$USER@$NODE"
# done

