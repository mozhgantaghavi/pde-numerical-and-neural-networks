import numpy as np

class IsoMesh2D:
    def __init__(self, inner_radius, outer_radius, num_r, num_theta):
        """
        Initializes the mesh generator for circular geometries.
        """
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.num_r = num_r
        self.num_theta = num_theta
        self.nodes = None
        self.connectivity = None
        self.boundary_nodes = None

    def generate_mesh(self):
        """
        Generates 2D coordinates mapping from polar to Cartesian domain.
        """
        r_vals = np.linspace(self.inner_radius, self.outer_radius, self.num_r)
        theta_vals = np.linspace(0, 2 * np.pi, self.num_theta, endpoint=False)
        
        # Create node coordinates
        nodes_list = []
        for r in r_vals:
            for theta in theta_vals:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                nodes_list.append([x, y])
        self.nodes = np.array(nodes_list)
        
        # Build connectivity matrix for triangular isoparametric elements
        connectivity_list = []
        for i in range(self.num_r - 1):
            for j in range(self.num_theta):
                n1 = i * self.num_theta + j
                n2 = i * self.num_theta + (j + 1) % self.num_theta
                n3 = (i + 1) * self.num_theta + j
                n4 = (i + 1) * self.num_theta + (j + 1) % self.num_theta
                
                # Split each quadrilateral patch into two quadratic/linear triangles
                connectivity_list.append([n1, n2, n4])
                connectivity_list.append([n1, n4, n3])
                
        self.connectivity = np.array(connectivity_list)
        
        # Extract boundary node indices for Dirichlet boundary conditions
        self.boundary_nodes = {
            'inner': np.arange(self.num_theta),
            'outer': np.arange((self.num_r - 1) * self.num_theta, self.num_r * self.num_theta)
        }
        return self.nodes, self.connectivity, self.boundary_nodes