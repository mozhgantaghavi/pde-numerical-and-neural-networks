import torch
import numpy as np

def generate_burgers_data(config):
    h = config['domain']['h']
    k = config['domain']['k']
    x_min, x_max = config['domain']['x_min'], config['domain']['x_max']
    t_min, t_max = config['domain']['t_min'], config['domain']['t_max']
    
    x = torch.arange(x_min, x_max + h, h)
    t = torch.arange(t_min, t_max + k, k)
    
    # PDE Loss (whole domain)
    x_pde = torch.stack(torch.meshgrid(x, t, indexing='ij')).reshape(2, -1).T
    
    # boundary and initial condition for Train Data
    bc_1 = torch.stack(torch.meshgrid(x[0], t, indexing='ij')).reshape(2, -1).T
    bc_2 = torch.stack(torch.meshgrid(x[-1], t, indexing='ij')).reshape(2, -1).T 
    ic   = torch.stack(torch.meshgrid(x, t[0], indexing='ij')).reshape(2, -1).T
    
    x_train = torch.cat([bc_1, bc_2, ic])
    
    # boundary and initial conditions
    y_bc_1 = torch.zeros(len(bc_1))
    y_bc_2 = torch.zeros(len(bc_2))
    y_ic   = -torch.sin(np.pi * ic[:, 0])
    
    y_train = torch.cat([y_bc_1, y_bc_2, y_ic]).unsqueeze(1)
    
    return x_pde, x_train, y_train