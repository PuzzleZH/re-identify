#%% import part
from os import replace
import torch
import numpy as np
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchDataLoader
import random

MAX_ITERS = 1200000
num_per_picture=20
#%% Dataset
my_cos=torch.nn.CosineSimilarity(dim=1)
class my_dataset(torchDataset):
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

    def __getitem__(self,idx):
        #TODO:get p people finish
        #TODO:each person get K pictures finish
        #TODO:pick hardest positive and hardest nagetive from pictures finish
        index = np.random.choice(range(0,130),10,replace=False)
        # anchor_idx = self.data[index[0]][1] 
        # pos_list = self.ref[anchor_idx].copy()
        anchor_idx=index[0]
        pos_list = self.ref[anchor_idx].copy()
        # anchor
        anchor_pth = random.choice(pos_list)
        anchor_id=anchor_idx
        anchor_choosen_code=anchor_pth

        anchor_tensor=torch.from_numpy(np.fromfile(f'train/train_feature/{anchor_pth}',dtype=np.float32))
        # remove anchor from the list
        pos_list.remove(anchor_pth)
        # pos
        if len(pos_list) > num_per_picture:
            pos_pth = np.random.choice(pos_list,num_per_picture,replace=False)
        else:
            pos_pth = np.random.choice(pos_list,num_per_picture,replace=True)

        positive_tensor=torch.Tensor()
        for i in pos_pth:
            tem_positive = torch.from_numpy(np.fromfile(f'train/train_feature/{i}', dtype=np.float32))
            positive_tensor=torch.cat((positive_tensor,tem_positive),0)
        positive_tensor=positive_tensor.reshape(-1,2048)
        distance_cos=my_cos(anchor_tensor,positive_tensor)
        topk_index=distance_cos.topk(k=num_per_picture)[1]
        hard_positive_index=topk_index[-1]



        negative_tensor=torch.Tensor()
        for nagetive_people_index in index[1:]:
            nag_list=self.ref[nagetive_people_index]
            if len(nag_list)>num_per_picture:
                nag_pth=np.random.choice(nag_list,num_per_picture,replace=False)
            else:
                nag_pth=np.random.choice(nag_list,num_per_picture,replace=True)
            for i in nag_pth:
                tem_negative= torch.from_numpy(np.fromfile(f'train/train_feature/{i}', dtype=np.float32))
                negative_tensor=torch.cat((negative_tensor,tem_negative),0)
        negative_tensor=negative_tensor.reshape(-1,2048)
        distance_cos=my_cos(anchor_tensor,negative_tensor)
        topk_index=distance_cos.topk(k=num_per_picture*(len(index)-1))[1]
        hard_negative_index=topk_index[0]
        # for find neg neg_index
        list_neg_id=index[1:]
        neg_id=list_neg_id[hard_negative_index//num_per_picture]

        pos_tensor = positive_tensor[hard_positive_index]
        neg_tensor =negative_tensor[hard_negative_index]
        return anchor_tensor, pos_tensor, neg_tensor,anchor_id,anchor_choosen_code,neg_id,index
        # return anchor_tensor, pos_tensor, neg_tensor
#%%


# dataset=my_dataset()
# dataloader = torchDataLoader(dataset,
# 					     batch_size=1,
# 					     num_workers=0,
#  					     shuffle=True,
# 					     pin_memory=True)
# print(dataloader_iter.__next__())
#%% test




