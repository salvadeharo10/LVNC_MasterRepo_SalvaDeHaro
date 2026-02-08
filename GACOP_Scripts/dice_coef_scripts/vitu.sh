#!/bin/bash
#
#SBATCH -p amdahl-gpu
#SBATCH --chdir=/home/salvador/
#SBATCH -o /home/salvador/master_2025/slurm_outputs/vitu/slurm-%j.out
#SBATCH -J diceViTU
#SBATCH --mem=0
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL   # END/START/NONE
#SBATCH --mail-user=salvadorde.haroo@um.es

module load singularity/3.8.0

cd ../vitunet-lvnc

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

singularity exec --nv \
  -B /home/salvador:/home/salvador \
  --pwd /home/salvador/master_2025/vitunet-lvnc \
  /nas/hdd-0/singularity_images/blade_runner_2025v2.sif \
  bash -c "python ./dice_coefficients_patients.py"