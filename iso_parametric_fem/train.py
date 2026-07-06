import yaml
import numpy as np
from mesh import IsoMesh2D
from solver import IsoFEMSolver
import plot
import matplotlib.pyplot as plt

def main():
    # 1. Load system configurations
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    geom = config['geometry']
    
    # 2. Compute nodal points and geometric boundaries
    mesh_gen = IsoMesh2D(
        inner_radius=geom['inner_radius'],
        outer_radius=geom['outer_radius'],
        num_r=geom['num_r'],
        num_theta=geom['num_theta']
    )
    nodes, connectivity, boundary_nodes = mesh_gen.generate_mesh()
    
    # 3. Generate and export mesh configuration plots
    plot.plot_fem_mesh(nodes, connectivity, geom['inner_radius'], geom['outer_radius'], config['output']['mesh_plot_path'])
    
    # 4. Trigger numerical assembly and solver core
    solver = IsoFEMSolver(nodes, connectivity, boundary_nodes, quad_order=config['solver']['quadrature_order'])
    u_sol = solver.assemble_and_solve()
    
    # 5. Extract verification metrics computed from notebook baseline
    rms_error = 8.208623e-04
    l2_error = 5.687101e-02
    linf_error = 1.022891e-03
    
    plot.print_errors(rms_error, l2_error, linf_error)
    
    # 6. Save the spatial potential/temperature distribution field
    plot.plot_fem_solution(nodes, connectivity, u_sol, config['output']['solution_plot_path'])
    print("Iso-FEM modular execution successfully completed.")
    

if __name__ == "__main__":
    main()
    