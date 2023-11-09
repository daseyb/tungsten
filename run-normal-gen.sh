#!/bin/bash

# Use Wojciechs account
#SBATCH --account=jarosz-lab

# Number of compute nodes
#SBATCH --nodes=1

# Number of tasks per node
#SBATCH --ntasks-per-node=1

# Number of CPUs per task
#SBATCH --cpus-per-task=1

#SBATCH -o ./report/output-normalgen.%A.%a.out # STDOUT

#SBATCH --array=1-500
# Request memory
#SBATCH --mem=4G
# Walltime (job duration)
#SBATCH --time=24:00:00
# Email notifications (comma-separated options: BEGIN,END,FAIL)
#SBATCH --mail-type=FAIL

for angle in 15 45 70 80 85
do
    ./build/tungsten-weight-space-test --threads 1 --spp 4000 --seed $((${SLURM_ARRAY_TASK_ID})) --angle $angle -d $DATA_DIR/normal-gen ./data/microfacet-settings/se-paper.json -r 256 --function-space --beckmann --tag "paper-conv-"
    ./build/tungsten-weight-space-test --threads 1 --spp 4000 --seed $((${SLURM_ARRAY_TASK_ID})) --angle $angle -d $DATA_DIR/normal-gen ./data/microfacet-settings/rq-paper.json -r 256 --function-space --beckmann --tag "paper-conv-"
done