#!/bin/bash
#SBATCH --array=1-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=16G                  # memory (per node)
#SBATCH --time=0-1:00
#SBATCH --output=out.log          # output file
#SBATCH --error=err.log           # error file

dataset=reuters
cuda_on=$true
if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
  name=eda
  python3.8 -u main.py -dataset $dataset -name $name -cuda_on $cuda_on
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then
  name=mrmp
  mrmp_on=$true
  python3.8 -u main.py -dataset $dataset -name $name -mrmp_on $mrmp_on -cuda_on $cuda_on
fi