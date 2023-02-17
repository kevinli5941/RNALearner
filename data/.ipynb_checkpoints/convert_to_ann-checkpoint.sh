#!/bin/bash
#SBATCH --job-name=fullrecountaggregation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:30
#SBATCH --mem=400G

module load anaconda
conda activate scbert

python3 convert_to_ann.py
mv slurm-$SLURM_JOB_ID.out reports/
sacct -l -j $SLURM_JOB_ID