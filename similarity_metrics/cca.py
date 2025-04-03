import torch
import numpy as np

def cca_torch(X, Y, epsilon=1e-9, common_type=torch.float32, device="cpu", 
              regularize=True, reg_lambda=1e-4, return_directions=False):
    """
    Calculate Canonical Correlation Analysis (CCA) between two matrices.
    
    Parameters:
    - X: torch.Tensor of shape (n_samples, n_features_x)
    - Y: torch.Tensor of shape (n_samples, n_features_y)
    - epsilon: Small value to prevent division by zero
    - common_type: Data type for computation
    - device: Device to perform computation on
    - regularize: Whether to apply regularization (recommended for high-dim data)
    - reg_lambda: Regularization strength
    - return_directions: Whether to return canonical directions in addition to correlations
    
    Returns:
    - correlations: Canonical correlation coefficients
    - [optional] x_directions, y_directions: Canonical directions if return_directions=True
    """
    if X.dtype != Y.dtype:
        print("Warning: Input tensors have different dtypes. Casting to common type.")
        X = X.to(common_type)
        Y = Y.to(common_type)
    elif X.dtype != common_type:
        X = X.to(common_type)
        Y = Y.to(common_type)
    
    X = X.to(device)
    Y = Y.to(device)
    
    n = X.shape[0]
    dx = X.shape[1]
    dy = Y.shape[1]
    
    if n == 0:
        return torch.tensor(0.0, device=device, dtype=common_type)
    
    X_c = X - torch.mean(X, dim=0, keepdim=True)
    Y_c = Y - torch.mean(Y, dim=0, keepdim=True)
    
    Cxx = (X_c.T @ X_c) / (n - 1)
    Cyy = (Y_c.T @ Y_c) / (n - 1)
    Cxy = (X_c.T @ Y_c) / (n - 1)
    
    if regularize:
        Cxx = Cxx + reg_lambda * torch.eye(dx, device=device, dtype=common_type)
        Cyy = Cyy + reg_lambda * torch.eye(dy, device=device, dtype=common_type)
    
    eigenvalues_x, eigenvectors_x = torch.linalg.eigh(Cxx)
    valid_indices_x = eigenvalues_x > epsilon
    sqrt_eigenvalues_x = torch.sqrt(eigenvalues_x[valid_indices_x])
    eigenvectors_x = eigenvectors_x[:, valid_indices_x]
    Cxx_inv_sqrt = eigenvectors_x @ torch.diag(1.0 / sqrt_eigenvalues_x) @ eigenvectors_x.T
    
    eigenvalues_y, eigenvectors_y = torch.linalg.eigh(Cyy)
    valid_indices_y = eigenvalues_y > epsilon
    sqrt_eigenvalues_y = torch.sqrt(eigenvalues_y[valid_indices_y])
    eigenvectors_y = eigenvectors_y[:, valid_indices_y]
    Cyy_inv_sqrt = eigenvectors_y @ torch.diag(1.0 / sqrt_eigenvalues_y) @ eigenvectors_y.T
    
    T = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
    
    U, S, Vh = torch.linalg.svd(T, full_matrices=False)
    
    correlations = S
    
    if return_directions:
        x_directions = Cxx_inv_sqrt @ U
        y_directions = Cyy_inv_sqrt @ Vh.T
        return correlations, x_directions, y_directions
    else:
        return correlations

def cca_similarity(X, Y, k=None, **kwargs):
    """
    Compute CCA similarity score based on average of top-k canonical correlations.
    
    Parameters:
    - X: torch.Tensor of shape (n_samples, n_features_x)
    - Y: torch.Tensor of shape (n_samples, n_features_y)
    - k: Number of top correlations to use. If None, uses all.
    - **kwargs: Additional arguments to pass to cca_torch
    
    Returns:
    - similarity: Average of top-k canonical correlations
    """
    correlations = cca_torch(X, Y, **kwargs)
    
    if k is None:
        k = min(correlations.shape[0], 10)  # Default to at most 10 components
    else:
        k = min(k, correlations.shape[0])
    
    similarity = torch.mean(correlations[:k])
    
    return similarity.item()