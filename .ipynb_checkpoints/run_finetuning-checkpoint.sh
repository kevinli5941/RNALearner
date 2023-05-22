#!/bin/bash
#SBATCH --job-name=scbertfinetuning
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
dataPath2="data/osd104/osd104_preprocessed_ensembl_15117.h5ad"
geneNum=15117
modelName="finetune_guide15_withpred_lr10e3_training12"
modelPath="ckpts/recount3_mouse_15117_bulk_pretrain_continuation_1_bs_3_lr10e4_16.pth"
binNum=7
epoch=1000
batchSize=2
lr=1e-3
maskPath="masks/mask17guidepred.pt"

numGPUs=1

### 2 dataset problem

gradAcc=6

python3 -m torch.distributed.launch --nproc_per_node=$numGPUs new_finetune.py --data_path $dataPath --gene_num $geneNum --model_name $modelName --bin_num $binNum --learning_rate $lr --epoch $epoch --batch_size $batchSize --model_path $modelPath --grad_acc $gradAcc --data_path2 $dataPath2 --mask_path $maskPath

### 1 dataset problem

# gradAcc=1

# python3 -m torch.distributed.launch --nproc_per_node=$numGPUs new_finetune.py --data_path $dataPath --gene_num $geneNum --model_name $modelName --bin_num $binNum --learning_rate $lr --epoch $epoch --batch_size $batchSize --model_path $modelPath --grad_acc $gradAcc --mask_path $maskPath

mv slurm-$SLURM_JOB_ID.out reports/
sacct -l -j $SLURM_JOB_ID