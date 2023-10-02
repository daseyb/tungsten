sbatch -J $1 run-tungsten.sh
sbatch --dependency=singleton -J $1 run-average-images-cleanup.sh
