#!/bin/bash
#SBATCH --job-name=scbertpretraining
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -G1
#SBATCH --time=11:30:00
#SBATCH --mem=720G

module load anaconda
module load nvidia
conda activate scbert

dataPath="data/osd99/osd99_preprocessed_ensembl_15117.h5ad"
geneNum=15117
# modelPath="ckpts/finetune_no_guide_lr10e5_training2_best.pth"
modelPath="ckpts/finetune_guide15_withpred_lr10e4_training12_best.pth"
# modelPath="ckpts/denovo_lr10e5_training2_best.pth"
binNum=7
epoch=1000
batchSize=2
lr=1e-5

numGPUs=1

python3 predict.py --data_path $dataPath --gene_num $geneNum --bin_num $binNum --epoch $epoch --model_path $modelPath

mv slurm-$SLURM_JOB_ID.out reports/
sacct -l -j $SLURM_JOB_ID