#!/bin/bash

#SBATCH -p amdahl-gpu
#SBATCH --chdir=/home/salvador/
#SBATCH -o /home/salvador/master_2025/slurm_outputs/vitu/slurm-%j.out
#SBATCH -J vituBeta
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
  bash -c "python ./train.py run \
      --train-cfg ./config_files/config.yaml \
      --train-cfg ./config_files/config_best_HCM.yaml \
      --split-file ../../LVNC_dataset/via_cleaned_split_5_cv_HCM_X_Hebron_Titina_groupedby_patient.json \
      --augmentation-file ./config_files/augmentation/rotate_025.json
  "
