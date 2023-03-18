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

dataPath="data/osd105/osd105_preprocessed_ensembl_15117.h5ad"
geneNum=15117
modelName="denovo_from_27_lr10e4"
modelPath="ckpts/recount3_mouse_15117_bulk_pretrain_continuation_1_bs_3_lr1e5_17.pth"
binNum=7
epoch=500
batchSize=12
lr=1e-4

numGPUs=1

python3 -m torch.distributed.launch --nproc_per_node=$numGPUs de_novo_finetune.py --data_path $dataPath --gene_num $geneNum --model_name $modelName --bin_num $binNum --learning_rate $lr --epoch $epoch --batch_size $batchSize --model_path $modelPath

mv slurm-$SLURM_JOB_ID.out reports/
sacct -l -j $SLURM_JOB_ID