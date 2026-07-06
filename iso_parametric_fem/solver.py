import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from element import IsoElement

class IsoFEMSolver:
    def __init__(self, nodes, connectivity, boundary_nodes, quad_order=3):
        """
        Initializes the global finite element system assembler.
        """
        self.nodes = nodes
        self.connectivity = connectivity
        self.boundary_nodes = boundary_nodes
        self.quad_order = quad_order
        self.num_nodes = len(nodes)
        
    def assemble_and_solve(self):
        """
        Assembles element systems and enforces Dirichlet boundary conditions safely.
        """
        # Allocate memory using List-of-Lists format for efficient sparse assembly
        K_global = lil_matrix((self.num_nodes, self.num_nodes))
        F_global = np.zeros(self.num_nodes)
        
        # Assemble element systems into global system
        for elem in self.connectivity:
            elem_nodes = self.nodes[elem]
            K_elem, F_elem = IsoElement.compute_local_matrices(elem_nodes, self.quad_order)
            
            for i, node_i in enumerate(elem):
                for j, node_j in enumerate(elem):
                    K_global[node_i, node_j] += K_elem[i, j]
                F_global[node_i] += F_elem[i]
        
        # Define analytical boundary solutions (Laplace benchmark: u = ln(r)/ln(2))
        u_exact = np.zeros(self.num_nodes)
        for idx, (x, y) in enumerate(self.nodes):
            r = np.sqrt(x**2 + y**2)
            u_exact[idx] = np.log(r) / np.log(2.0)
            
        all_bc_nodes = np.concatenate([self.boundary_nodes['inner'], self.boundary_nodes['outer']])
        
        # Safe Dirichlet enforcement (Row-Clearing approach to maintain stability)
        for node in all_bc_nodes:
            K_global.rows[node] = [node]
            K_global.data[node] = [1.0]
            F_global[node] = u_exact[node]
            
        # Convert to CSR format for high-performance direct solving
        K_csr = K_global.tocsr()
        u_sol = spsolve(K_csr, F_global)
        
        # Verify that output vector contains valid finite values (No NaN or Inf)
        if not np.isfinite(u_sol).all():
            raise ValueError("Linear system solver returned non-finite values (NaN/Inf).")
            
        return u_sol