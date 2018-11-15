import torch.nn as nn

##############
# classes
##############

class Flatten(nn.Module):
    '''
    Flatten layer for conv to fc
    '''
    def forward(self, x):
        N, _, _, _, = x.size()
        return x.view(N, -1)

class Unflatten(nn.Module):
    '''
    Unflatten layer for fc to conv
    '''
    def __init__(self, N, C, H, W):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)
