import torch
import yaml
from dataset import generate_burgers_data
from pinn import BurgersPINN
from plots import plot_results

def main():

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    x_pde, x_train, y_train = generate_burgers_data(config)
    
    
    pinn = BurgersPINN(config, x_pde, x_train, y_train, device)
    pinn.train()
    
    torch.save(pinn.model.state_dict(), "burgers_pinn_model.pt")
    print("Model weights saved successfully!")

    print("Generating plots...")
    plot_results(config, pinn.model, device)

if __name__ == "__main__":
    main()
