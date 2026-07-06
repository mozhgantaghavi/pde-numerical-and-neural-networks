import yaml
import torch
import torch.optim as optim
from model import PINN
from pinn import calculate_loss
from plot import evaluate_and_plot

def main():
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        

    net_config = config.get('network', {})
    model = PINN(
        input_dim=net_config.get('input_dim', 3),
        hidden_dim=net_config.get('hidden_dim', 64),
        output_dim=net_config.get('output_dim', 1)
    )
    
    # Optimizer
    lr = float(config['training']['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    iterations = config['training']['iterations']
    print_every = config['training']['print_every']

    
    try:
        model.load_state_dict(torch.load("heat_2d_pinn.pt"))
        print("Previous weights loaded successfully! Continuing training...")
    except FileNotFoundError:
        print("No previous weights found. Starting training from scratch.")
    
        model_path = "heat_2d_pinn.pt"
    best_loss = float('inf')  
    
    print("\nStarting 2D Heat Equation PINN Training...")
    
    
    for iteration in range(iterations):
        optimizer.zero_grad()
        
        loss = calculate_loss(model, config)
        
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        
        
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), model_path) 
        
    
        if iteration % print_every == 0:
            print(f"Iteration {iteration:4d} | Current Loss: {current_loss:.4e} | Best Loss: {best_loss:.4e}")
            
    print(f"\nTraining finished! Best achieved loss was: {best_loss:.4e}")

    
    print("Generating 2D Heat Equation plots...")
    evaluate_and_plot(model_path) 

if __name__ == "__main__":
    main()
