import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import random
from tqdm import tqdm

### global settings
# model
IN = 2048
OUT = 1024

# train
GPU_ID = 0
LEARNING_RATE = 1e-4
MOMENTUM = 0.0005
WEIGHT_DECAY = 0.9 
BATCH_SIZE = 1  
MARGIN = 0.3
ITERS = 120000
NUM_WORKERS = 0

# save
SAVE_EVERY = 2000
SNAPSHOT = 'snapshots'

### dataset
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self):
        with open('train/train_list.txt') as f:
            items = [[line.strip().split()[0], int(line.strip().split()[1])] for line in f.readlines()]
        self.data = {}
        for item in items:
            if item[1] not in self.data:
                self.data[item[1]] = [item[0]]
            else:
                self.data[item[1]].append(item[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        pos_list = self.data[idx]
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
        neg_idx_list = self.data.keys()
        neg_idx_list.remove(idx)
        neg_idx = random.choice(neg_idx_list)
        neg_list = self.data[neg_idx]
        # neg
        neg_pth = random.choice(neg_list)
        # load data
        anchor = torch.from_numpy(np.fromfile(f'train/train_feature/{anchor_pth}', dtype=np.float32))
        pos = torch.from_numpy(np.fromfile(f'train/train_feature/{pos_pth}', dtype=np.float32))
        neg = torch.from_numpy(np.fromfile(f'train/train_feature/{neg_pth}', dtype=np.float32))
        return anchor, pos, neg

### model
class Model(nn.Module):

    def __init__(self, out):
        super(Model, self).__init__()
        self.linear = nn.Linear(IN, out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, anchor, pos, neg):
        anchor_out = self.linear(anchor)
        anchor_out = self.relu(anchor_out)
        pos_out = self.linear(pos)
        pos_out = self.relu(pos_out)
        neg_out = self.linear(neg)
        neg_out = self.relu(neg_out)
        return anchor_out, pos_out, neg_out

### training objective
def loss_calc(anchor, pos, neg, margin):
    triplet_loss = nn.TripletMarginLoss(margin=margin)
    return triplet_loss(anchor, pos, neg)


### main
if __name__ == '__main__':
    device = GPU_ID

    # dataset
    dataset = Dataset()
    
    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
					     batch_size=BATCH_SIZE,
					     num_workers=NUM_WORKERS,
 					     shuffle=True,
					     pin_memory=True)
    dataloader_iter = enumerate(dataloader)
    
    # create model & start training
    model = Model(OUT)
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
        anchor, pos, neg = data
        # propagate
        anchor_out, pos_out, neg_out = model(anchor.cuda(device), pos.cuda(device), neg.cuda(device))
        # loss
        loss = loss_calc(anchor_out, pos_out, neg_out, margin=MARGIN)
        loss.backward()
        # print
        print('Iter {}: {:.4f}'.format(i_iter, loss.item()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        # save
        if i_iter % SAVE_EVERY == 0 and i_iter:
            torch.save(model.state_dict(), f'{SNAPSHOT}/model_{i_iter}.pth')
        sys.stdout.flush()
    
