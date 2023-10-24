#!/bin/bash

# Use Wojciechs account
#SBATCH --account=jarosz-lab

# Number of compute nodes
#SBATCH --nodes=1

# Number of tasks per node
#SBATCH --ntasks-per-node=1

# Number of CPUs per task
#SBATCH --cpus-per-task=2

#SBATCH -o ./report/output.%j.%a.out # STDOUT

#SBATCH --array=1-50
# Request memory
#SBATCH --mem=4G
# Walltime (job duration)
#SBATCH --time=06:00:00
# Email notifications (comma-separated options: BEGIN,END,FAIL)
#SBATCH --mail-type=FAIL
export JOB_DIR="/dartfs-hpc/rc/lab/J/JaroszLab/dseyb/stimp/${SLURM_JOB_NAME}"
mkdir -p $JOB_DIR
./build/tungsten --threads 1 --spp 1 --seed $((${SLURM_ARRAY_TASK_ID})) -d $JOB_DIR -o "$((${SLURM_ARRAY_TASK_ID})).png" -e "$((${SLURM_ARRAY_TASK_ID})).exr" "$abs_scene_path" 
