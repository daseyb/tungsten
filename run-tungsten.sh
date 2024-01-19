#!/bin/bash

# Use Wojciechs account
#SBATCH --account=jarosz-lab

# Number of compute nodes
#SBATCH --nodes=1

# Number of tasks per node
#SBATCH --ntasks-per-node=1

# Number of CPUs per task
#SBATCH --cpus-per-task=10

#SBATCH -o ./report/output-render.%A.%a.out # STDOUT

#SBATCH --array=1-10
# Request memory
#SBATCH --mem=12G
# Walltime (job duration)
#SBATCH --time=24:00:00
# Email notifications (comma-separated options: BEGIN,END,FAIL)
#SBATCH --mail-type=FAIL
export JOB_DIR="/dartfs-hpc/rc/lab/J/JaroszLab/dseyb/stimp/${SLURM_JOB_NAME}/$((${SLURM_ARRAY_JOB_ID}))"
mkdir -p $JOB_DIR
./build/tungsten --threads 10 --spp 1 --seed $((${SLURM_ARRAY_TASK_ID})) -d $JOB_DIR -o "$((${SLURM_ARRAY_TASK_ID})).png" -e "$((${SLURM_ARRAY_TASK_ID})).exr" "./data/example-scenes/${SLURM_JOB_NAME}.json" 
