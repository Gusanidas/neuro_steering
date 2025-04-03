import torch

def centered_kernel_alignment(X, Y):
    """
    Calculate Centered Kernel Alignment between two representation matrices.
    
    Parameters:
    - X: torch.Tensor of shape (n_samples, n_features_x)
    - Y: torch.Tensor of shape (n_samples, n_features_y)
    
    Returns:
    - cka: The CKA similarity score between 0 and 1
    """
    # Handle different data types by casting to the same type
    if X.dtype != Y.dtype:
        # Cast to double (float64) for maximum precision
        X = X.to(torch.float64)
        Y = Y.to(torch.float64)
    
    # Calculate linear kernel matrices
    K = X @ X.T
    L = Y @ Y.T
    
    # Center the kernel matrices
    n = K.shape[0]
    H = torch.eye(n, device=X.device, dtype=X.dtype) - torch.ones((n, n), device=X.device, dtype=X.dtype) / n
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    
    # Calculate HSIC
    hsic_kl = torch.sum(K_centered * L_centered) / (n-1)**2
    hsic_kk = torch.sum(K_centered * K_centered) / (n-1)**2
    hsic_ll = torch.sum(L_centered * L_centered) / (n-1)**2
    
    # Calculate CKA
    cka = hsic_kl / torch.sqrt(hsic_kk * hsic_ll)
    
    return cka