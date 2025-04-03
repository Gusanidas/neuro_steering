import torch
import numpy as np
from sklearn.decomposition import PCA

def pca_pytorch(data_tensor: torch.Tensor, k: int) -> torch.Tensor:
    """
    Performs Principal Component Analysis (PCA) on a PyTorch tensor.

    Args:
        data_tensor: A 2D PyTorch tensor of shape (n_samples, n_features) 
                     containing the input data.
        k: The desired number of principal components (output dimensions). 
           Must be a positive integer less than or equal to the minimum 
           of n_samples and n_features.

    Returns:
        A 2D PyTorch tensor of shape (n_samples, k) containing the data
        projected onto the top k principal components. The tensor will
        be on the same device and have the same float dtype as the 
        input tensor.

    Raises:
        ValueError: If data_tensor is not 2D, k is not positive, or k 
                    is greater than the number of features.
        TypeError: If the input tensor cannot be converted to NumPy.
        RuntimeError: If the PCA computation fails.
    """

    n_samples, n_features = data_tensor.shape
    
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer. Got {k}.")
        
    if k > n_features:
        raise ValueError(f"k ({k}) cannot be greater than the number of "
                         f"features ({n_features}).")
        


    original_device = data_tensor.device
    original_dtype = data_tensor.dtype
    
    if not torch.is_floating_point(data_tensor):
        data_tensor = data_tensor.to(torch.float32) 
        print(f"Warning: Input tensor was not float, converting to {data_tensor.dtype}.")
        original_dtype = data_tensor.dtype

    try:
        np_data = data_tensor.detach().cpu().numpy()
    except Exception as e:
        raise TypeError(f"Failed to convert PyTorch tensor to NumPy array. Error: {e}")

    try:
        pca = PCA(n_components=k)
        pca_result_np = pca.fit_transform(np_data) 
    except Exception as e:
        raise RuntimeError(f"Scikit-learn PCA failed. Error: {e}")

    try:
        pca_tensor = torch.from_numpy(pca_result_np)
        pca_tensor = pca_tensor.to(device=original_device, dtype=original_dtype) 
    except Exception as e:
        raise TypeError(f"Failed to convert PCA result from NumPy back to PyTorch tensor. Error: {e}")

    return pca_tensor