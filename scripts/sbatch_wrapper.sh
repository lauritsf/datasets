#!/bin/sh
# SLURM script for job scheduling.
# Usage: sbatch sbatch_wrapper.sh <python_script> <python_script_arguments>
# It is simply meant to be a wrapper for the python scripts that are run on the cluster.

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G


## INFO
echo "Node: $(hostname)"
ehco "Start: $(date)"
echo "Job Name: $SLURM_JOB_NAME"
echo "Working Directory: $(pwd)\n"



if [ -z "$1" ]
then
    echo "Error: No python script provided."
    echo "Usage: sbatch sbatch_wrapper.sh <python_script> <python_script_arguments>"
    exit 1
fi

PYTHON_SCRIPT=$1
shift # Shift all arguments to the left by one, i.e. remove the first argument.

echo "Running $PYTHON_SCRIPT with arguments $@"
python $PYTHON_SCRIPT "$@"


echo "Done: $(date)"
