import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_results(config, model, device):
    model.eval()
    

    h = 0.01 
    k = 0.01
    x_min, x_max = config['domain']['x_min'], config['domain']['x_max']
    t_min, t_max = config['domain']['t_min'], config['domain']['t_max']
    
    x = np.arange(x_min, x_max + h, h)
    t = np.arange(t_min, t_max + k, k)
    X, T = np.meshgrid(x, t)
    
    x_test = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
    t_test = torch.tensor(T.flatten(), dtype=torch.float32).unsqueeze(1)
    X_test = torch.cat([x_test, t_test], dim=1).to(device)
    
    with torch.no_grad():
        u_pred = model(X_test).cpu().numpy()
    
    U_pred = u_pred.reshape(X.shape)
    
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(T, X, U_pred, cmap='rainbow', shading='gouraud')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('Time (t)')
    plt.ylabel('Space (x)')
    plt.title('PINN Solution for 1D Burgers Equation')
    
   
    plt.savefig('burgers_contour.png', dpi=300)
    print("Contour plot saved as 'burgers_contour.png'")
    
    
    plt.figure(figsize=(12, 4))
    time_slices = [0.25, 0.50, 0.75]
    
    for i, t_slice in enumerate(time_slices):
       
        t_idx = np.abs(t - t_slice).argmin()
        
        plt.subplot(1, 3, i+1)
        plt.plot(x, U_pred[t_idx, :], 'r--', linewidth=2, label='PINN')
        plt.title(f't = {t_slice}')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.grid(True)
        if i == 0:
            plt.legend()
            
    plt.tight_layout()
    plt.savefig('burgers_slices.png', dpi=300)
    print("Slices plot saved as 'burgers_slices.png'")
    plt.show()