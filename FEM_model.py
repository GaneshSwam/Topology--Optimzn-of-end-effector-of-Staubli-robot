import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

class FEMModel:
    def __init__(self, nelx, nely, volfrac, penal, rmin):
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.nel = nelx * nely
        self.nn = (nelx + 1) * (nely + 1)
        self.x = volfrac * np.ones(self.nel)
        self.ke = self.element_stiffness()
        self.fixed_nodes = self.get_fixed_nodes()
        self.f = np.zeros(self.nn * 2)
        self.u = np.zeros(self.nn * 2)

    def element_stiffness(self):
        k = np.array([[ 4,  2, -4,  2],
                      [ 2,  4,  2, -4],
                      [-4,  2,  4, -2],
                      [ 2, -4, -2,  4]]) / 12
        ke = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                ke[i, j] = k[i, j] * self.x / (1 + self.penal * (1 - self.x))
        return ke

    def get_fixed_nodes(self):
        fixed_nodes = []
        for i in range(nelx + 1):
            j = 0
            fixed_nodes.append(j * (nelx + 1) + i)
            j = nely
            fixed_nodes.append(j * (nelx + 1) + i)
        for j in range(nely + 1):
            i = 0
            fixed_nodes.append(j * (nelx + 1) + i)
            i = nelx
            fixed_nodes.append(j * (nelx + 1) + i)
        return fixed_nodes

    def get_global_indices(self, i, j):
        n1 = j * (self.nelx + 1) + i
        n2 = (j + 1) * (self.nelx + 1) + i
        n3 = (j + 1) * (self.nelx + 1) + i + 1
        n4 = j * (self.nelx + 1) + i + 1
        return n1, n2, n3, n4

    def get_element_indices(self, el):
        i = el % self.nelx
        j = el // self.nelx
        return i, j

    def get_element_dofs(self, i, j):
        n1, n2, n3, n4 = self.get_global_indices(i, j)
        return [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1,
                2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1]

    def element_density(self, i, j):
        el = j * self.nelx + i
        return self.x[el]

    def element_stress(self, i, j):
        el = j * self.nelx + i
        edof = self.get_element_dofs(i, j)

def fixed_nodes(nelx, nely):
    fixed = np.zeros((2, (nelx+1)*(nely+1)), dtype=int)
    k = 0
    for i in range(nelx+1):
        fixed[0, k] = i*(nely+1)
        fixed[1, k] = 0
        k += 1
    for i in range(nelx+1):
        fixed[0, k] = i*(nely+1) + nely
        fixed[1, k] = 0
        k += 1
    return fixed

def plot_results(x, nelx, nely):
    x_phys = np.zeros(nelx*nely)
    x_phys[x.flatten()>0.5] = 1.
    x_phys = x_phys.reshape((nely, nelx))
    plt.imshow(x_phys.T, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.show()


def topology_optimization(nelx, nely, volfrac, penal, rmin, ft, x, rho_min, **kwargs):
    # Define constants
    E = 1.
    nu = 0.3
    lb = np.zeros(nely * nelx)
    ub = np.ones(nely * nelx)

    # Define filter kernel
    rmin *= np.sqrt(2)

    # Define finite element model
    fem = FEMModel(nelx, nely, E, nu, ft)

    # Define density constraint
    def constraint(x):
        return volfrac * nelx * nely - np.sum(x)

    # Define objective function
    def obj_fun(x):
        return fem.compute_compliance(x)

    # Define derivative of objective function
    def obj_fun_deriv(x):
        return fem.compute_compliance_deriv(x)

    # Define derivative of density filter
    def filter_density_deriv(x, H):
        return H @ (x - rho_min)

    # Define fixed nodes
    fixed_nodes = fem.get_fixed_nodes()

    # Define optimization problem
    problem = OptimizationProblem(
        obj_fun,
        obj_fun_deriv,
        lb=lb,
        ub=ub,
        constraint=constraint,
        constraint_deriv=None,
        filter_fun=density_filter,
        filter_deriv=filter_density_deriv,
        filter_args=(nelx, nely, rmin),
        fixed_nodes=fixed_nodes,
        **kwargs
    )

    # Solve optimization problem
    optimizer = MMA(x0=x, lb=lb, ub=ub, problem=problem, penal=penal)
    x_opt = optimizer.optimize()

    return x_opt.reshape((nely, nelx))

def main():
    nelx = 60
    nely = 30
    volfrac = 0.4
    rmin = 2.5
    ft = 1
    penal = 3.0
    maxiter = 50

    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones(nely * nelx, dtype=float)
    dc = np.zeros(nely * nelx, dtype=float)
    adj = np.zeros(nely * nelx, dtype=float)
    passive = np.zeros(nely * nelx, dtype=bool)
    for i in range(nely):
        for j in range(nelx):
            if (i + j) % 2 == 0:
                passive[i * nelx + j] = True

    # Define fixed points
    fixed_pts = fixed_nodes(nelx, nely)

    # Perform topology optimization
    x_opt = topology_optimization(nelx, nely, volfrac, rmin, ft, penal, passive, maxiter, x, fixed_pts, dc, adj)

    # Plot the optimized results
    plot_results(x_opt, nelx, nely)












