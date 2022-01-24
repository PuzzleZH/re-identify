#%% import
from numpy import average
import torch.nn as nn
from torch.utils.data import DataLoader as torchDataLoader
import torch.optim as optim
from dataset import my_dataset
from model import Model
import torch
import nni
from visualdl import LogWriter
#%% Constant
IN=2048
MID=64
OUT=2048
LEARNING_RATE = 1e-4
MOMENTUM = 0.0005 
WEIGHT_DECAY = 0.9
BATCH_SIZE = 32 
MARGIN = 0.5
ITERS = 120000

my_cos=torch.nn.CosineSimilarity(dim=1)
#%% initial


def loss_calc(anchor, pos, neg, margin):
    triplet_loss = nn.TripletMarginLoss(margin=margin)
    return triplet_loss(anchor, pos, neg)

def main():
    dataset=my_dataset()
    model=Model(MID,OUT)

    dataloader = torchDataLoader(dataset,
					     batch_size=32,
					     num_workers=0,
 					     shuffle=True,
					     pin_memory=True)
    
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_RATE,
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    num_epochs = 10000
    with LogWriter(logdir="./log/scalar_test/train") as writer:
        for i_iter in range(num_epochs):
            # reset optimizer
            optimizer.zero_grad()
            # adjust learning rate if needed
            # adjust_learning_rate(optimizer, i_iter)
            # load data
            anchor,pos,neg,anchor_id,anchor_choosen_code,neg_id,index = next(iter(dataloader))
            anchor_with_positive=my_cos(anchor,pos)
            anchor_with_negative=my_cos(anchor,neg)
            # print(anchor_with_positive,anchor_with_negative)
            # print("anchor_id={},anchor_choosen_cod={},neg_id={}".format(anchor_id,anchor_choosen_code,neg_id))
            # print("index_list={}".format(index))
            # adaptive mining
            # anchor, pos, neg, ada_t = ada_mine(model, dataloader_iter, device)
            # propagate
            anchor_out = model(anchor)
            pos_out = model(pos)
            neg_out = model(neg)
            # loss
            loss = loss_calc(anchor_out, pos_out, neg_out, margin=MARGIN)
            loss.backward()
            # print
            # print('Iter {}: {:.4f}'.format(i_iter, loss.item()))
            writer.add_scalar(tag="loss", step=i_iter, value=loss.item())
        optimizer.step()

if __name__ == "__main__":
    main()

