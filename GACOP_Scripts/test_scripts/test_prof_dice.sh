#!/bin/bash

#SBATCH -p amdahl-cpu
#SBATCH --chdir=/home/salvador/
#SBATCH -o /home/salvador/master_2025/slurm_outputs/vitu/slurm-%j.out
#SBATCH -J prof_dice
#SBATCH --mem=0
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=salvadorde.haroo@um.es

module load singularity/3.8.0

find . | grep -E "(__pycache__|\\.pyc|\\.pyo$)" | xargs rm -rf

singularity exec --nv \
  -B /home/salvador:/home/salvador \
  --pwd /home/salvador/master_2025/vitunet-lvnc \
  /nas/hdd-0/singularity_images/blade_runner_2025v2.sif \
  bash -c "python ./test_ensemble_profiling_dice.py --experiment-dir ./logs/vitu_normal --device cpu --precision bf16-mixed --batch-size 1"
  #bash -c "python ./test_ensemble_profiling_dice.py --experiment-dir ./logs/vitunet_lvnc_detector_cleaned_via --device gpu --precision 16-mixed --batch-size 6"
  #bash -c "python ./test_ensemble_profiling_dice.py --experiment-dir ./logs/vitunet_lvnc_detector_cleaned_via --device cpu --precision bf16-mixed --batch-size 1"
  

  