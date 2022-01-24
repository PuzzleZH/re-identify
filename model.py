#%% import
import torch.nn as nn
#%%
IN = 2048

#%%
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
