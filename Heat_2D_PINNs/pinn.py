import torch
import torch.nn as nn
from dataset import (
    generate_training_data, 
    generate_boundary_training_data, 
    initial_condition, 
    boundary_condition
)

def compute_pde_residual(x, y, t, model):
    input_data = torch.cat([x, y, t], dim=1)
    u = model(input_data)
    
    # First derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]

    # Heat Equation Residual: u_xx + u_yy - u_t = 0
    heat_eq_residual = u_xx + u_yy - u_t
    return heat_eq_residual

def calculate_loss(model, config):
    num_pts = config['training']['number_points']
    b_val = config['physics']['custom_boundary_value']
    
    # 1. Interior Domain Points & PDE Residual
    x, y, t = generate_training_data(num_pts)
    residual = compute_pde_residual(x, y, t, model)
    loss_pde = nn.MSELoss()(residual, torch.zeros_like(residual))
    
    # 2. Initial Condition Points
    t_initial = torch.zeros_like(t)
    u_init_pred = model(torch.cat([x, y, t_initial], dim=1))
    u_init_true = initial_condition(x, y)
    loss_init = nn.MSELoss()(u_init_pred, u_init_true)
    
    # 3. Boundary Condition Points
    x_b, y_b, t_b = generate_boundary_training_data(num_pts)
    
    u_b_x_pred = model(torch.cat([x_b, y_b, t_b], dim=1))
    u_b_x_true = boundary_condition(x_b, y_b, t_b, b_val)
    loss_boundary_x = nn.MSELoss()(u_b_x_pred, u_b_x_true)
    
    u_b_y_pred = model(torch.cat([y_b, x_b, t_b], dim=1))
    u_b_y_true = boundary_condition(y_b, x_b, t_b, b_val)
    loss_boundary_y = nn.MSELoss()(u_b_y_pred, u_b_y_true)
    
    # Total Loss
    total_loss = loss_pde + loss_init + loss_boundary_x + loss_boundary_y
    return total_loss