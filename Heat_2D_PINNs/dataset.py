import torch

def initial_condition(x, y):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

def boundary_condition(x, y, t, custom_value=0.0):
    return torch.full_like(x, custom_value)

def generate_training_data(number_points):
    x = torch.rand(number_points, 1, requires_grad=True)
    y = torch.rand(number_points, 1, requires_grad=True)
    t = torch.rand(number_points, 1, requires_grad=True)
    return x, y, t

def generate_boundary_points(number_points):
    x_boundary = torch.tensor([0.0, 1.0]).repeat(number_points // 2)
    y_boundary = torch.rand(number_points)

    if torch.rand(1) > 0.5:
        x_boundary, y_boundary = y_boundary, x_boundary

    return x_boundary.view(-1, 1), y_boundary.view(-1, 1)

def generate_boundary_training_data(number_points):
    x_b, y_b = generate_boundary_points(number_points)
    t_b = torch.rand(number_points, 1, requires_grad=True)
    return x_b, y_b, t_b