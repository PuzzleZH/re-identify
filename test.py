import torch
from train import Model
import json

IN = 2048
OUT = 1024

if __name__ == '__main__':
    
    # load
    model = Model(OUT)
    model.load_state_dict('snapshots/model.pth')
    
    # test mode
    model.eval()

    '''
        给定一个 query 特征，计算 gallery 中所有特征和该特征的
        余弦相似度，返回排名前一百的文件
    '''
    
    res = {}
    for query in queries:
        for value in gallery:
            pass

    # res = {'00000.png': [], '00001.png': [], ...}
    json.dump(res, open('res.json', 'w', encoding='utf-8'), indent=4)
    
