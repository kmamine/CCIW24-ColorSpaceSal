import torch.nn as nn 

class SalLoss(nn.Module): 
    def __init__(self): 
        self.mse = ... 
        self.nss = ... 
        self.kld = ... 
        self.congruence = ... 

    def forward(self, pred, gt): 

        loss = ...
        return loss
    

class NSS(nn.Module): 
    def __init__(self): 
        pass

    def forward(self, pred, gt): 

        loss = ...
        return loss    


class KLD(nn.Module): 
    def __init__(self): 
        pass

    def forward(self, pred, gt): 

        loss = ...
        return loss
    
class Congruence(nn.Module): 
    def __init__(self): 
        pass

    def forward(self, pred, gt): 

        loss = ...
        return loss