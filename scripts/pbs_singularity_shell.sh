#!/bin/bash

#PBS -S /bin/bash
#PBS -N flow_bayes
#PBS -q charon
##PBS -l select=1:ncpus=10:mem=10gb:scratch_local=5gb
#PBS -l walltime=90:00:00

# run fterm sing atleast once to create the image
#./bin/fterm_sing

cfg_path=$CFG_PATH

export RAY_BACKEND_LOG_LEVEL=debug

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

rm -rf $SCRATCHDIR/*
