#!/bin/bash
#SBATCH --job-name=fullrecountaggregation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=11:20:30
#SBATCH --mem=400G

module load anaconda
conda activate scbert

python3 recount3_preprocessing.py
mv slurm-$SLURM_JOB_ID.out reports/
sacct -l -j $SLURM_JOB_ID