import ott # type: ignore
from ott.solvers import linear # type: ignore
import jax.numpy as jnp
import jax 

def project_to_psd(matrix):
    """
    Project a matrix to the nearest positive semidefinite matrix.
    
    Args:
    matrix: A square matrix
    
    Returns:
    The nearest positive semidefinite matrix to the input matrix
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    return eigenvectors @ jnp.diag(jnp.maximum(eigenvalues, 0)) @ eigenvectors.T

def matrix_sqrt(A):
    """
    Compute the matrix square root using eigendecomposition.
    
    Args:
    A: A symmetric positive definite matrix
    
    Returns:
    The matrix square root of A
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(A)
    eigenvalues = jax.nn.relu(eigenvalues)
    return eigenvectors @ jnp.diag(jnp.sqrt(eigenvalues)) @ eigenvectors.T

def gaussian_monge_map(Nx, Ny):
    """
    Compute the Gaussian Monge map from N(mu_x, sigma_x) to N(mu_y, sigma_y).
    
    Args:
    Nx: Mean and covariance of the source distribution (shape: (d,))
    Ny: Mean and covariance of the target distribution (shape: (d,))

    Returns:
    Parameters for function T(x) that maps points from the source to the target distribution
    """
    
    # Compute A = sigma_y^(1/2) (sigma_y^(1/2) sigma_x sigma_y^(1/2))^(-1/2) sigma_y^(1/2)
    
    mu_x, sigma_x = Nx
    mu_y, sigma_y = Ny

    sigma_y_sqrt = matrix_sqrt(sigma_y)
    inner_sqrt = matrix_sqrt(sigma_y_sqrt @ sigma_x @ sigma_y_sqrt.T)

    
    # Compute A
    A = sigma_y_sqrt @ jnp.linalg.pinv(inner_sqrt) @ sigma_y_sqrt.T
    b = mu_y - A @ mu_x
    # Define the Monge map function
    return A,b

def mccann_interpolation(Nx, T, t):

    
    mu_x, sigma_x = Nx
    A, b = T

    d = mu_x.shape[0]
    Iden = jnp.eye(d)

    mu_t = (1 - t) * mu_x + t * (A @ mu_x + b)
    M = (1 - t) * Iden + t * A
    sigma_t = M @ sigma_x @ M.T
    
    return mu_t, sigma_t

def mccann_derivative(Nx, T, t):
 
    mu_x, sigma_x = Nx
    A, b = T

    d = mu_x.shape[0]
    Iden = jnp.eye(d)

    mu_t_dot = (A-Iden) @ mu_x + b
    sigma_t_dot = (A-Iden) @ sigma_x @ ((1-t) * Iden + t * A).T + ((1-t) * Iden + t * A) @ sigma_x @ (A-Iden).T
    
    return mu_t_dot, sigma_t_dot


def frechet_distance(Nx, Ny):
    """
    Compute the Fréchet distance between two Gaussian distributions.
    
    Args:
    Nx: Mean and covariance of the source distribution (shape: (d,))
    Ny: Mean and covariance of the target distribution (shape: (d,))

    Returns:
    The Fréchet distance between the two distributions
    """
    mu_x, sigma_x = Nx
    mu_y, sigma_y = Ny

    mean_diff_squared = jnp.sum((mu_x - mu_y)**2)
    
    # Compute the sum of the square roots of the eigenvalues of sigma_x @ sigma_y
    sigma_x_sqrt =matrix_sqrt(sigma_x)
    product = sigma_x_sqrt @ sigma_y @ sigma_x_sqrt
    eigenvalues = jnp.linalg.eigvalsh(product)
    trace_term = jnp.sum(jnp.sqrt(jnp.maximum(eigenvalues, 0)))  # Ensure non-negative
    
    # Compute the trace of the sum of covariances
    trace_sum = jnp.trace(sigma_x + sigma_y)
    
    # Compute the Fréchet distance
    return(mean_diff_squared + trace_sum - 2 * trace_term)
    


def sinkhorn_from_distance(distance_matrix, eps = 0.1, lse_mode = False): #produces deltas from x to y


    ot_solve = linear.solve(
        ott.geometry.geometry.Geometry(cost_matrix = distance_matrix, epsilon = eps, scale_cost = 'mean'),
        lse_mode = lse_mode,
        min_iterations = 0,
        max_iterations = 100)
    return(ot_solve.matrix)