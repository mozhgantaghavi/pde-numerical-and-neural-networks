import yaml
import torch
import torch.optim as optim
from model import PINN
from pinn import calculate_loss
from plot import evaluate_and_plot

def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Initialize Model using configurations
    model = PINN(
        input_dim=config['network']['input_dim'],
        hidden_dim=config['network']['hidden_dim'],
        output_dim=config['network']['output_dim']
    )
    
    # Setup Optimizer
    lr = float(config['training']['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    iterations = config['training']['iterations']
    print_every = config['training']['print_every']

    
    print("Starting 2D Heat Equation PINN Training...")
    
    for iteration in range(iterations):
        optimizer.zero_grad()
        
        # Calculate loss
        loss = calculate_loss(model, config)
        
        loss.backward()
        optimizer.step()
        
        if iteration % print_every == 0:
            print(f"Iteration {iteration:4d} | Loss: {loss.item():.4f}")
            
    model_path = "heat_2d_pinn.pt"
    
    # Save the trained model weights securely
    torch.save(model.state_dict(), model_path)
    print("Training finished and model saved!")

    # plot
    print("Generating 2D Heat Equation plots...")
    evaluate_and_plot(model_path) 

if __name__ == "__main__":
    main()
