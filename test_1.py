import torch
from train import Model
import json
import torch.nn as nn
from train import Dataset

IN = 2048
OUT = 1024

if __name__ == '__main__':
    
    # load
    model = Model(OUT)
    model.load_state_dict('snapshots/model.pth')
    # generate queries and gallery
    dataset=Dataset
    queries={}
    gallery={}
    # 抽取每一个人物的第一张图片
    for i in len(dataset):
        queries[i].append(dataset.data[i][0])
    # 重组dataset剔除queries中的图片
    for i in dataset.data:
        if i not in queries:
            gallery.append(i)
    
    # test mode
    model.eval()
    cos=nn.CosineSimilarity(dim=1)
    '''
        给定一个 query 特征，计算 gallery 中所有特征和该特征的
        余弦相似度，返回排名前一百的文件
    '''
    
    res = {}

    def forward_mini(object_aim):
        object_out=model.linear(object_aim)
        object_out=model.relu(object_out)
        return object_out

    def recall_k(k=100,sortList,aim,relative_total):
        sortList.sort()
        count=0
        aim_label=aim.label
        for i in k:
            if sortList[i].label==aim_label :
                count+=1
        recall=count/relative_total
        precision=count/k
        return recall,precision

    for query in queries:
        query_out=forward_mini(query)
        sort_list = []
        relativeTotal=0
        for value in gallery:
            value_out=forward_mini(value)
            cos_value=cos(query_out,value_out)
            if value.label==query.label:
                relativeTotal+=1
            sort_list.append([cos_value,value])
        print(recall_k(k=100,sortList=sort_list,aim=query,relative_total=relativeTotal))
    # res = {'00000.png': [], '00001.png': [], ...}
    json.dump(res, open('res.json', 'w', encoding='utf-8'), indent=4)
    
