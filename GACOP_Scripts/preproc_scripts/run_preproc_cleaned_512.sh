#!/bin/bash
#
#SBATCH -p amdahl-gpu
#SBATCH --chdir=/home/salvador/
#SBATCH -o /home/salvador/slurmOutputs/slurm-%j.out
#SBATCH -J norm_preproc
#SBATCH --mem=0
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL   # END/START/NONE
#SBATCH --mail-user=salvadorde.haroo@um.es

module load singularity/3.8.0


find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

singularity exec --nv -B $home_dir:/home/salvador/vitunet-lvnc --pwd /home/salvador/vitunet-lvnc /nas/hdd-0/singularity_images/aquilesV4.sif bash -c "python ./new_generate_preproc_folder.py --data-folder ../LVNC_dataset/ --preproc-name Cleaned_New_Preproc_2d_512_combinedresize_zscore --target-size 512 512"

