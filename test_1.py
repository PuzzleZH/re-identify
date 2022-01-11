from typing import List
import torch
from train import Model
import json
import torch.nn as nn
import numpy as np
from torch import Tensor

IN = 2048
OUT = 1024
MAX_ITERAS=1200000

if __name__ == '__main__':


    # load
    model = Model(OUT)
    def forward_mini(value_out):
        #直接用forward会出错，因为你的代码中的forward是关于三个向量的 
        value_out=model.linear(value_out)
        value_out=model.relu(value_out)
        return value_out
    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            with open('train/train_list.txt') as f:
                self.data= [[line.strip().split()[0], int(line.strip().split()[1])] for line in f.readlines()]
            self.ref= {}
            for item in self.data:
                if item[1] not in self.ref:
                    self.ref[item[1]] = [item[0]]
                else:
                    self.ref[item[1]].append(item[0])
            self.data=self.data*int(np.ceil(float(MAX_ITERAS)/len(self.data)))
    
        def __len__(self):
            return len(self.ref)

    dataset=Dataset()
    # recall函数
    def recall_k(input_index:Tensor,person_index:int,gallery_people_flag:list,k:int):
        same_people=0
        for i in input_index:
            if gallery_people_flag[person_index]<= i  and i <gallery_people_flag[person_index+1] :
                same_people +=1
        recallk=same_people/(gallery_people_flag[person_index+1]-gallery_people_flag[person_index])
        preceise=same_people/k
        return recallk,preceise

    Mycos=torch.nn.CosineSimilarity(dim=-1)

    gallery_total_list=torch.Tensor()
    gallery_people_flag=[0] #用于存储每个人的图像索引界限
    k_value=20
    for gallery_samepeople in dataset.ref.values():
        gallery_list=torch.Tensor()

        for gallery_out in gallery_samepeople:
            gallery_out= np.fromfile(f'train/train_feature/{gallery_out}', dtype=np.float32)
            gallery_out=torch.from_numpy(gallery_out)
            gallery_out=forward_mini(gallery_out)
            gallery_list=torch.cat((gallery_list,gallery_out),-1)
        gallery_list=gallery_list.reshape(-1,1024)
        gallery_total_list=torch.cat((gallery_total_list,gallery_list),0)
        gallery_people_flag.append(gallery_total_list.shape[0])
        # print(gallery_total_list[0].shape)
    wich_person_flag=0
    for query_people in dataset.ref.keys():
        query_out=np.fromfile(f'train/train_feature/{dataset.ref[query_people][0]}',dtype=np.float32)
        query_out=torch.from_numpy(query_out)
        query_out=forward_mini(query_out)
        query_out=torch.Tensor(query_out)
        cos_value=Mycos(query_out,gallery_total_list)
        top_k_index=cos_value.topk(k=k_value)[1]
        recall_k_value,preceise_value=\
                recall_k(input_index=top_k_index,person_index=wich_person_flag,gallery_people_flag=gallery_people_flag,k=k_value)
        print("人物id:{},recall@k:{:.4f},准确度:{}".format( wich_person_flag,float(recall_k_value),preceise_value))
        wich_person_flag+=1

