import matplotlib
matplotlib.use('TkAgg') # Forces matplotlib to open an interactive window
import matplotlib.pyplot as plt
import numpy as np

def plot_fem_mesh(nodes, connectivity, inner_r, outer_r, save_path):
    """
    Plots the structured 2D isoparametric finite element mesh grid.
    """
    plt.figure(figsize=(7, 6))
    for tri in connectivity:
        pts = nodes[tri]
        # Draw element edges
        plt.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), 'k-', alpha=0.3, linewidth=0.8)
    
    # Plot circular analytical boundaries
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(inner_r*np.cos(theta), inner_r*np.sin(theta), 'r--', label='Inner Boundary (r=1)')
    plt.plot(outer_r*np.cos(theta), outer_r*np.sin(theta), 'b--', label='Outer Boundary (r=2)')
    
    plt.scatter(nodes[:, 0], nodes[:, 1], c='black', s=20, zorder=3)
    plt.gca().set_aspect('equal')
    plt.title("FEM Mesh with Circular Boundaries")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_fem_solution(nodes, connectivity, u_sol, save_path):
    """
    Generates a continuous filled contour map of the estimated solution.
    """
    plt.figure(figsize=(7, 6))
    contour = plt.tricontourf(nodes[:, 0], nodes[:, 1], connectivity, u_sol.ravel(), levels=50, cmap='inferno')
    plt.colorbar(contour, label='u_approximate')
    plt.title("Solution Contour (Iso-FEM)")
    plt.savefig(save_path, dpi=300)
    plt.show()

def print_errors(rms_err, l2_err, linf_err):
    """
    Prints validation metrics comparing numerical output against exact solution.
    """
    print(f"=== FEM Solver Error Metrics ===")
    print(f"RMS error  = {rms_err:.6e}")
    print(f"L2 error   = {l2_err:.6e}")
    print(f"Linf error = {linf_err:.6e}")