#!/bin/bash
# Name of the job
#SBATCH --job-name=stochastic-implicits

# Use Wojciechs account
#SBATCH --account=jarosz-lab

# Number of compute nodes
#SBATCH --nodes=1

# Number of tasks per node
#SBATCH --ntasks-per-node=1

# Number of CPUs per task
#SBATCH --cpus-per-task=1


#SBATCH --array=1-32
# Request memory
#SBATCH --mem=1G
# Walltime (job duration)
#SBATCH --time=01:30:00
# Email notifications (comma-separated options: BEGIN,END,FAIL)
#SBATCH --mail-type=FAIL
./build/tungsten --spp 8 --seed $((${SLURM_ARRAY_TASK_ID})) -e "./data/examples-scenes/gp-medium-cornell-box/$((${SLURM_ARRAY_TASK_ID})).hdr" "./data/example-scenes/gp-medium-cornell-box/gp-medium-cornell-box.json"
