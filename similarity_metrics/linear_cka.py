import torch

def linear_cka_efficient(X, Y, epsilon=1e-9, common_type=torch.float32, device="mps"):
    """
    Calculate Centered Kernel Alignment (CKA) efficiently for linear kernels.

    This version avoids computing the full n x n kernel matrices, making it
    suitable for large n when feature dimensions are smaller.

    Parameters:
    - X: torch.Tensor of shape (n_samples, n_features_x)
    - Y: torch.Tensor of shape (n_samples, n_features_y)
    - epsilon: Small value to prevent division by zero.

    Returns:
    - cka: The CKA similarity score between 0 and 1
    """
    if X.dtype != Y.dtype:
        print("Warning: Input tensors have different dtypes. Casting to float64.")
        X = X.to(common_type)
        Y = Y.to(common_type)
    elif X.dtype != common_type:
         X = X.to(common_type)
         Y = Y.to(common_type)

    X = X.to(device)
    Y = Y.to(device)

    n = X.shape[0]
    if n == 0:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)

    X_c = X - torch.mean(X, dim=0, keepdim=True)
    Y_c = Y - torch.mean(Y, dim=0, keepdim=True)


    C = X_c.T @ Y_c

    cka_numerator = torch.sum(C * C)

    X_cov = X_c.T @ X_c
    Y_cov = Y_c.T @ Y_c

    cka_denominator_term1 = torch.sum(X_cov * X_cov)
    cka_denominator_term2 = torch.sum(Y_cov * Y_cov)

    cka_denominator = torch.sqrt(cka_denominator_term1 * cka_denominator_term2 + epsilon)

    if cka_denominator < epsilon:
         print("Warning: CKA denominator is close to zero. Representations might be constant.")
         return torch.tensor(0.0, device=X.device, dtype=X.dtype)


    cka = cka_numerator / cka_denominator
    cka = torch.clamp(cka, min=0.0, max=1.0)

    return cka