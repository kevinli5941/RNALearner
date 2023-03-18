#!/bin/bash
#SBATCH --job-name=scbertpretraining
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -G4
#SBATCH --time=119:30:00
#SBATCH --mem=720G

module load anaconda
module load nvidia
conda activate scbert

dataPath="data/recount3/preprocessed_mouse_gene2vec_15117_compatible_bulk_only.h5ad"
geneNum=15117
modelName="recount3_mouse_15117_bulk_pretrain_continuation_1_bs_3_lr1e2"
binNum=7
epoch=20
batchSize=3
lr=1e-2

numGPUs=4

python3 -m torch.distributed.launch --nproc_per_node=$numGPUs pretrain.py --data_path $dataPath --gene_num $geneNum --model_name $modelName --bin_num $binNum --learning_rate $lr --epoch $epoch --batch_size $batchSize --model_path "ckpts/recount3_mouse_15117_bulk_pretrain_10.pth"

mv slurm-$SLURM_JOB_ID.out reports/
sacct -l -j $SLURM_JOB_ID