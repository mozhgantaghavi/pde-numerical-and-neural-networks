import torch
import torch.nn as nn

class PINNNet(nn.Module):
    def __init__(self, layers):
        super(PINNNet, self).__init__()
        
        # making layers dynamicaly 
        net_layers = []
        for i in range(len(layers) - 1):
            net_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                net_layers.append(nn.Tanh())
                
        self.net = nn.Sequential(*net_layers)

    def forward(self, x):
        return self.net(x)