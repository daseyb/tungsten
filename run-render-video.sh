#!/bin/bash


# Use Wojciechs account
#SBATCH --account=jarosz-lab

# Number of compute nodes
#SBATCH --nodes=1

# Number of tasks per node
#SBATCH --ntasks-per-node=1

# Number of CPUs per task
#SBATCH --cpus-per-task=1

#SBATCH -o ./report/output.%j.%a.out # STDOUT

# Request memory
#SBATCH --mem=2G
# Walltime (job duration)
#SBATCH --time=06:00:00
# Email notifications (comma-separated options: BEGIN,END,FAIL)
#SBATCH --mail-type=FAIL
conda init bash
source ~/.bashrc
conda activate research
export JOB_DIR="/dartfs-hpc/rc/lab/J/JaroszLab/dseyb/stimp/${SLURM_JOB_NAME}/.."
ffmpeg -y -framerate 12 -pattern_type glob -i "$JOB_DIR/*.png" -c:v libx264 -pix_fmt yuv420p render.mp4
