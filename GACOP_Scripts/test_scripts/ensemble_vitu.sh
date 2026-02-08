#!/bin/bash
#
#SBATCH -p amdahl-cpu 
#SBATCH --chdir=/home/salvador/
#SBATCH -o /home/salvador/master_2025/slurm_outputs/vitu/slurm-%j.out
#SBATCH -J TestEnsemble
#SBATCH --mem=0
#SBATCH --cpus-per-task=1#16
#SBATCH --mail-type=ALL   # END/START/NONE
#SBATCH --mail-user=salvadorde.haroo@um.es

module load singularity/3.8.0

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

singularity exec --nv \
  -B /home/salvador:/home/salvador \
  --pwd /home/salvador/master_2025/vitunet-lvnc \
  /nas/hdd-0/singularity_images/blade_runner_2025v2.sif \
  bash -c "python ./test_ensemble2.py --experiment-dir ./logs/goku_1 --device cpu --precision 32 --batch-size 1"
