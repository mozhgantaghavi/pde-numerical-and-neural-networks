import torch
import numpy as np
from models import PINNNet

class BurgersPINN:
    def __init__(self, config, x_pde, x_train, y_train, device):
        self.config = config
        self.device = device
        
        #  (CPU/GPU)
        self.x_pde = x_pde.to(device).requires_grad_(True)
        self.x_train = x_train.to(device)
        self.y_train = y_train.to(device)
        
        
        self.nu = config['domain']['nu']
        
        
        self.model = PINNNet(config['network']['layers']).to(device)
        
        # optimizers
        self.adam = torch.optim.Adam(self.model.parameters(), lr=config['training']['adam_lr'])
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=config['training']['lbfgs_lr'],
            max_iter=config['training']['lbfgs_max_iter'],
            history_size=config['training']['history_size'],
            tolerance_grad=float(config['training']['tolerance_grad']),
            tolerance_change=1e-16,  # <-- استفاده از یک عدد مشخص و کوچک اعشاری
            line_search_fn="strong_wolfe"
        )
        
        self.criterion = torch.nn.MSELoss()
        self.iter = 1

    def loss_function(self):
        self.adam.zero_grad()
        self.optimizer.zero_grad()

        # 1. data Loss 
        y_predict = self.model(self.x_train)
        loss_data = self.criterion(y_predict, self.y_train)

        # 2. physical Loss 
        u = self.model(self.x_pde)
        
        du_dxt = torch.autograd.grad(
            u, self.x_pde,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        du_dx = du_dxt[:, 0]
        du_dt = du_dxt[:, 1]

        du_dxx = torch.autograd.grad(
            du_dx, self.x_pde,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True, retain_graph=True
        )[0][:, 0]

        # Burgers equation: u_t + u*u_x = nu*u_xx
        loss_pde = self.criterion(du_dt + u.squeeze() * du_dx, self.nu * du_dxx)
        
        total_loss = loss_pde + loss_data
        total_loss.backward()

        if self.iter % self.config['training']['print_every'] == 0:
            print(f"Iteration {self.iter}, Total Loss: {total_loss.item():.6e}")
        self.iter += 1

        return total_loss

    def train(self):
        self.model.train()
        print("--- Training with Adam ---")
        for epoch in range(self.config['training']['adam_epochs']):
            self.adam.step(self.loss_function)
            
        print("--- Training with LBFGS ---")
        self.optimizer.step(self.loss_function)