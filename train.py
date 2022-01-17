import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import random
from tqdm import tqdm
from typing import Tuple
from torch import Tensor
import time

### global settings
# model
IN = 2048
MID = 64
OUT = 2048

# train
GPU_ID = 0
LEARNING_RATE = 1e-4
MOMENTUM = 0.0005
WEIGHT_DECAY = 0.9
BATCH_SIZE = 32 
MARGIN = 0.5
ITERS = 120000

MAX_ITERS = 1200000
NUM_WORKERS = 0

# save
SAVE_EVERY = 200
SNAPSHOT = 'snapshots'

### dataset
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self):
        with open('train/train_list.txt') as f:
            self.data = [[line.strip().split()[0], int(line.strip().split()[1])] for line in f.readlines()]
        self.ref = {}
        for item in self.data:
            if item[1] not in self.ref:
                self.ref[item[1]] = [item[0]]
            else:
                self.ref[item[1]].append(item[0])
        self.data = self.data * int(np.ceil(float(MAX_ITERS) / len(self.data)))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_idx = self.data[idx][1] 
        pos_list = self.ref[anchor_idx].copy()
        # anchor
        anchor_pth = random.choice(pos_list)
        # remove anchor from the list
        pos_list.remove(anchor_pth)
        # pos
        if len(pos_list) > 1:
            pos_pth = random.choice(pos_list)
        else:
            pos_pth = anchor_pth
        # neg list
        neg_idx_list = list(self.ref.keys())
        neg_idx_list.remove(anchor_idx)
        neg_idx = random.choice(neg_idx_list)
        neg_list = self.ref[neg_idx]
        # neg
        neg_pth = random.choice(neg_list)
        # load data
        anchor = torch.from_numpy(np.fromfile(f'train/train_feature/{anchor_pth}', dtype=np.float32))
        pos = torch.from_numpy(np.fromfile(f'train/train_feature/{pos_pth}', dtype=np.float32))
        neg = torch.from_numpy(np.fromfile(f'train/train_feature/{neg_pth}', dtype=np.float32))
        return anchor, pos, neg

### model
class Model(nn.Module):

    def __init__(self, mid, out):
        super(Model, self).__init__()
        self.linear = nn.Linear(IN, mid)
        self.regen = nn.Linear(mid, out)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, input):
        out = self.linear(input)
        out = self.relu(out)
        out = self.regen(out)
        out = self.relu(out)        
        return out

### training objective
def loss_calc(anchor, pos, neg, margin):
    triplet_loss = nn.TripletMarginLoss(margin=margin)
    return triplet_loss(anchor, pos, neg)


### adaptive mining
def ada_mine(model, dataloader_iter, device) -> Tuple[Tensor, Tensor, Tensor, float]:
    # record time   
    start = time.time()

    anchor_lst = []
    pos_lst = []
    neg_lst = []
    
    while len(anchor_lst) < BATCH_SIZE:
        with torch.no_grad():
            _, data = dataloader_iter.__next__()
            anchor, pos, neg = data
            # propagate
            anchor_out = model(anchor.cuda(device))
            pos_out = model(pos.cuda(device))
            neg_out = model(neg.cuda(device))
            
            if loss_calc(anchor_out, pos_out, neg_out, margin=MARGIN) > 0:
                anchor_lst.append(anchor)
                pos_lst.append(pos)
                neg_lst.append(neg)
    
    anchor = torch.cat(anchor_lst)
    pos = torch.cat(pos_lst)
    neg = torch.cat(neg_lst)
    
    assert anchor.shape[0] == BATCH_SIZE
    assert pos.shape[0] == BATCH_SIZE
    assert neg.shape[0] == BATCH_SIZE
    
    ada_t = time.time() - start
    return anchor, pos, neg, ada_t

### main
def main():
    device = GPU_ID

    # dataset
    dataset = Dataset()
    
    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
					     batch_size=1,
					     num_workers=NUM_WORKERS,
 					     shuffle=True,
					     pin_memory=True)
    dataloader_iter = enumerate(dataloader)
    
    # create model & start training
    model = Model(MID, OUT)
    model.train()
    model.to(device)
    # cudnn.benchmark = True
    # cudnn.enabled = True
    
    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_RATE,
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    
    # train
    for i_iter in tqdm(range(ITERS + 1)):
        # reset optimizer
        optimizer.zero_grad()
        # adjust learning rate if needed
        # adjust_learning_rate(optimizer, i_iter)
        # load data
        _, data = dataloader_iter.__next__()
        # adaptive mining
        anchor, pos, neg, ada_t = ada_mine(model, dataloader_iter, device)
        # propagate
        anchor_out = model(anchor.cuda(device))
        pos_out = model(pos.cuda(device))
        neg_out = model(neg.cuda(device))
        # loss
        loss = loss_calc(anchor_out, pos_out, neg_out, margin=MARGIN)
        loss.backward()
        # print
        print('Iter {}: {:.4f}, Mining Cost: {:.4f} sec'.format(i_iter, loss.item(), ada_t))
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        # save
        if i_iter % SAVE_EVERY == 0 and i_iter:
            torch.save(model.state_dict(), f'{SNAPSHOT}/model_{i_iter}.pth')
        sys.stdout.flush()

if __name__ == '__main__':
    main()
