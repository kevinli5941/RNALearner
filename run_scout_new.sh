#!/bin/bash
#SBATCH --job-name=scout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -G4
#SBATCH --time=11:30:00
#SBATCH --mem=720G

module load anaconda
module load nvidia
conda activate scbert

dataPath="data/recount3/kmeans30/pairs/preprocessed_mouse_gene2vec_15117_compatible_bulk_only_seed_132_pair_30.h5ad"
geneNum=15117
modelName="scout_pairs_30"
modelPath="ckpts/recount3_mouse_15117_bulk_pretrain_continuation_1_bs_3_lr10e4_16.pth"
binNum=7
epoch=100
batchSize=3
lr=1e-4
gradAcc=60
patience=10

numGPUs=4

python3 -m torch.distributed.launch --nproc_per_node=$numGPUs scout.py --data_path $dataPath --gene_num $geneNum --model_name $modelName --bin_num $binNum --learning_rate $lr --epoch $epoch --batch_size $batchSize --model_path $modelPath --grad_acc $gradAcc --patience $patience

mv slurm-$SLURM_JOB_ID.out reports/
sacct -l -j $SLURM_JOB_ID