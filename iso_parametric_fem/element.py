import numpy as np

class IsoElement:
    @staticmethod
    def get_quadrature(order=3):
        """
        Returns Gauss-Legendre integration points and weights for standard triangle.
        Using 3-point quadrature rule to prevent zero-Jacobian or singularity issues.
        """
        pts = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
        wts = np.array([1/6, 1/6, 1/6])
        return pts, wts

    @staticmethod
    def shape_functions(xi, eta):
        """
        Computes standard linear triangular shape functions and their local derivatives.
        """
        N = np.array([1 - xi - eta, xi, eta])
        dN_dxi = np.array([-1.0, 1.0, 0.0])
        dN_deta = np.array([-1.0, 0.0, 1.0])
        return N, dN_dxi, dN_deta

    @classmethod
    def compute_local_matrices(cls, elem_nodes, order=3):
        """
        Calculates the element local stiffness matrix using explicit Jacobian transformation.
        """
        n_dof = len(elem_nodes)
        K_elem = np.zeros((n_dof, n_dof))
        F_elem = np.zeros(n_dof)
        
        pts, wts = cls.get_quadrature(order)
        
        for pt, wt in zip(pts, wts):
            xi, eta = pt[0], pt[1]
            N, dN_dxi, dN_deta = cls.shape_functions(xi, eta)
            
            # Construct the Jacobian matrix mapping local coordinates to global Cartesian ones
            J = np.zeros((2, 2))
            J[0, 0] = np.dot(dN_dxi, elem_nodes[:, 0])
            J[0, 1] = np.dot(dN_dxi, elem_nodes[:, 1])
            J[1, 0] = np.dot(dN_deta, elem_nodes[:, 0])
            J[1, 1] = np.dot(dN_deta, elem_nodes[:, 1])
            
            # Compute determinant and inverse of the Jacobian matrix
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            abs_detJ = abs(detJ)
            
            # Prevent potential division by zero
            if abs_detJ < 1e-12:
                continue
                
            invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ
            
            # Map derivatives to global Cartesian coordinates (dN/dx, dN/dy)
            dN_dX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
            dN_dY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
            
            # Perform numerical integration for the stiffness matrix
            for i in range(n_dof):
                for j in range(n_dof):
                    K_elem[i, j] += (dN_dX[i] * dN_dX[j] + dN_dY[i] * dN_dY[j]) * abs_detJ * wt
                    
        return K_elem, F_elem