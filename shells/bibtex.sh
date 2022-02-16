#!/bin/bash
#SBATCH --array=1-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=8G                  # memory (per node)
#SBATCH --time=0-1:00
#SBATCH --output=out.log          # output file
#SBATCH --error=err.log           # error file

dataset=bibtex
cuda_on=$true
name=mrmp
mrmp_on=$true
if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
  n_layers_mrmp=4
  python3.8 -u main.py -dataset $dataset -name $name -mrmp_on $mrmp_on -cuda_on $cuda_on -n_layers_mrmp $n_layers_mrmp
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then
  n_layers_mrmp=6
  python3.8 -u main.py -dataset $dataset -name $name -mrmp_on $mrmp_on -cuda_on $cuda_on -n_layers_mrmp $n_layers_mrmp
fi