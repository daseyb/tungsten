#!/bin/bash

#SBATCH --job-name install-deps

# Use Wojciechs account
#SBATCH --account=jarosz-lab

# Number of compute nodes
#SBATCH --nodes=1

# Number of tasks per node
#SBATCH --ntasks-per-node=1

# Number of CPUs per task
#SBATCH --cpus-per-task=8

#SBATCH -o ./report/output.%j.%a.out # STDOUT

# Request memory
#SBATCH --mem=8G
# Walltime (job duration)
#SBATCH --time=06:00:00
# Email notifications (comma-separated options: BEGIN,END,FAIL)
#SBATCH --mail-type=FAIL
scl enable devtoolset-10 bash
module load cmake/3.23.2
conda init bash
source ~/.bashrc
conda activate research
scl enable devtoolset-10 bash
module load cmake/3.23.2
conda activate research
../vcpkg/vcpkg install openvdb openexr
