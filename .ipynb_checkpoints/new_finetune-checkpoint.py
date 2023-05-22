# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad', help='Path of data for finetune.')
parser.add_argument("--data_path2", type=str, default=None, help='2nd dataset, optional.')
parser.add_argument("--model_path", type=str, default=None, help='Path of pretrained model.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='finetune', help='Finetuned model name.')
parser.add_argument("--mask_path", type=str, default=None, help='Path to guidance mask.')

args = parser.parse_args()
rank = int(os.environ["RANK"])
local_rank = args.local_rank
is_master = local_rank == 0

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every

PATIENCE = 10
UNASSIGN_THRES = 0.0

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

model_name = args.model_name
ckpt_dir = args.ckpt_dir

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()

seed_all(SEED + torch.distributed.get_rank())


class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        #rand_start = random.randint(0, self.data.shape[0]-1)
        rand_start = index
        
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]


class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

if not args.data_path2:
    data = sc.read_h5ad(args.data_path)
    label_dict, label = np.unique(np.array(data.obs['y']), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
    class_num = np.unique(label, return_counts=True)[1].tolist()
    class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
    label = torch.from_numpy(label)
    data = data.X

else:
    data1 = sc.read_h5ad(args.data_path)
    data2 = sc.read_h5ad(args.data_path2)
    data = data1.concatenate(data2)
    label_dict, label = np.unique(np.array(data.obs['y']), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
    class_num = np.unique(label, return_counts=True)[1].tolist()
    class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
    label = torch.from_numpy(label)
    data = data.X

acc = []
f1 = []
f1w = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pred_list = pd.Series(['un'] * data.shape[0])

if not args.data_path2: #if no separate dataset for validation
    sss = StratifiedShuffleSplit(n_splits=1, train_size=2, test_size=10, random_state=SEED)
    for index_train, index_val in sss.split(data, label):
        print("Test:", index_train, index_val)
        data_train, label_train = data[index_train], label[index_train]
        data_val, label_val = data[index_val], label[index_val]
        print("Train label: ", label_train, " Val label: ", label_val)
        train_dataset = SCDataset(data_train, label_train)
        val_dataset = SCDataset(data_val, label_val)  
else: #if second dataset for validation
    total_len = data.shape[0]
    data_train, label_train = data[:total_len // 2], label[:total_len // 2]
    data_val, label_val = data[total_len // 2:], label[total_len // 2:]
    train_dataset = SCDataset(data_train, label_train)
    val_dataset = SCDataset(data_val, label_val) 
    
print(train_dataset.data.shape)
print(val_dataset.data.shape) 

# train_sampler = DistributedSampler(train_dataset, shuffle = False)
# val_sampler = DistributedSampler(val_dataset, shuffle = False)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

train_sampler = SequentialSampler(train_dataset)
val_sampler = SequentialSampler(val_dataset)
 

model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
    g2v_position_emb = POS_EMBED_USING
)

path = args.model_path
ckpt = torch.load(path)
model.load_state_dict(ckpt['model_state_dict'])
for param in model.parameters():
    param.requires_grad = False
for param in model.norm.parameters():
    param.requires_grad = True
for param in model.performer.net.layers[-2].parameters():
    param.requires_grad = True
model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
model = model.to(device)
# model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
# model = model.to(device)
#model = DDP(model, device_ids=[local_rank], output_device=local_rank)

if args.mask_path:
    print("yes path!")
    guidance_mask = torch.load(args.mask_path, map_location = device)
else:
    guidance_mask = {}
    for name, param in model.named_parameters():
        guidance_mask[name] = 1

# optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)
loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)
dist.barrier()
trigger_times = 0
max_acc = 0.0

### testing ###
# print("enumerate train_loader")
# for i in train_sampler:
#     print(i)
# for i in train_sampler:
#     print(i)
# for i in train_sampler:
#     print(i)

def save_best_ckpt(epoch, model, optimizer, scheduler, losses, model_name, ckpt_folder):
    """
    保存模型checkpoint
    """
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'losses': losses,
        },
        f'{ckpt_folder}{model_name}_best.pth'
    )

print("begin training")
train_losses = []
val_losses = []
train_accs = []
val_accs = []
for i in range(1, EPOCHS+1):
    #train_loader.sampler.set_epoch(i)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.train()
    dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0
    for index, (data, labels) in enumerate(train_loader):
        index += 1
        data, labels = data.to(device), labels.to(device)
        if index % GRADIENT_ACCUMULATION != 0:
            # with model.no_sync():
            #     logits = model(data)
            #     loss = loss_fn(logits, labels)
            #     loss.backward()
            logits = model(data)
            loss = loss_fn(logits, labels)
            loss.backward()
        if index % GRADIENT_ACCUMULATION == 0:
            logits = model(data)
            loss = loss_fn(logits, labels)
            loss.backward()
            #GTL mask appliance
            for name, param in model.named_parameters():
                if param.grad is not None and name in guidance_mask:
                    mask_val = guidance_mask[name]
                    param.grad *= mask_val
                else:
                    print(f"{name} has no guide value")
                    print("Name in guidance?: ", name in guidance_mask)
                    print("gradient val: ", param.grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
        softmax = nn.Softmax(dim=-1)
        final = softmax(logits)
        final = final.argmax(dim=-1)
        pred_num = labels.size(0)
        correct_num = torch.eq(final, labels).sum(dim=-1)
        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
    epoch_loss = running_loss / index
    epoch_acc = 100 * cum_acc / index
    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
    dist.barrier()
    scheduler.step()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        dist.barrier()
        running_loss = 0.0
        predictions = []
        truths = []
        with torch.no_grad():
            for index, (data_v, labels_v) in enumerate(val_loader):
                index += 1
                data_v, labels_v = data_v.to(device), labels_v.to(device)
                logits = model(data_v)
                loss = loss_fn(logits, labels_v)
                running_loss += loss.item()
                softmax = nn.Softmax(dim=-1)
                final_prob = softmax(logits)
                final = final_prob.argmax(dim=-1)
                final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                predictions.append(final)
                truths.append(labels_v)
            del data_v, labels_v, logits, final_prob, final
            # gather
            # predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
            # truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
            predictions = torch.cat(predictions, dim=0)
            truths = torch.cat(truths, dim=0)
            
            no_drop = predictions != -1
            predictions = np.array((predictions[no_drop]).cpu())
            truths = np.array((truths[no_drop]).cpu())
            cur_acc = accuracy_score(truths, predictions)
            f1 = f1_score(truths, predictions, average='macro')
            val_loss = running_loss / index
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
            val_acc = 100 * cur_acc
            val_acc = get_reduced(val_acc, local_rank, 0, world_size)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            if is_master:
                print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | Accuracy: {val_acc:6.4f}%  ==')
                print(confusion_matrix(truths, predictions))
                #print(classification_report(truths, predictions, target_names=label_dict.tolist(), digits=4))
            if cur_acc >= max_acc:
                max_acc = cur_acc
                trigger_times = 0
                save_best_ckpt(i, model, optimizer, scheduler, val_loss, model_name, ckpt_dir)
            else:
                trigger_times += 1
                # if trigger_times > PATIENCE:
                #     break
    del predictions, truths

                        
plt.figure(figsize=(10,5))
plt.title("Training and Validation Accuracies")
plt.plot(val_accs,label="val")
plt.plot(train_accs,label="train")
plt.xlabel("iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("figures/" + args.model_name + "_acc")

plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(val_losses,label="val")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("figures/" + args.model_name + "_val")