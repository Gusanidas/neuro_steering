import torch
import numpy as np
from scipy.spatial import procrustes

def procrustes_pytorch(tensor_a: torch.Tensor, 
                       tensor_b: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Performs Procrustes analysis between two PyTorch tensors.

    Procrustes analysis finds the optimal translation, rotation, and 
    uniform scaling that maps points in tensor_b to points in tensor_a, 
    minimizing the sum of squared differences.

    Args:
        tensor_a: A 2D PyTorch tensor of shape (n_samples, n_features) 
                  representing the reference set of points.
        tensor_b: A 2D PyTorch tensor of shape (n_samples, n_features)
                  representing the set of points to be transformed. 
                  Must have the same shape as tensor_a.

    Returns:
        A tuple containing:
        - standardized_a (np.ndarray): Standardized version of tensor_a 
                                       (centered, unit scaling).
        - transformed_b (np.ndarray): tensor_b after applying the optimal 
                                      translation, rotation, and scaling 
                                      to align it with standardized_a.
        - disparity (float): The sum of squared differences between 
                             standardized_a and transformed_b after alignment.
                             Lower values indicate better similarity.

    Raises:
        ValueError: If the input tensors are not 2D or do not have the 
                    same shape.
    """
    
    if tensor_a.ndim != 2 or tensor_b.ndim != 2:
        raise ValueError("Input tensors must be 2D (n_samples, n_features).")
    if tensor_a.shape != tensor_b.shape:
        raise ValueError(f"Input tensors must have the same shape. " 
                         f"Got {tensor_a.shape} and {tensor_b.shape}.")
    if tensor_a.shape[0] < tensor_a.shape[1]:
         print(f"Warning: Number of samples ({tensor_a.shape[0]}) is less than "
               f"number of features ({tensor_a.shape[1]}). Procrustes may be less stable.")

    try:
        np_a = tensor_a.detach().cpu().numpy()
        np_b = tensor_b.detach().cpu().numpy()
    except Exception as e:
        raise TypeError(f"Failed to convert PyTorch tensors to NumPy arrays. Error: {e}")

    try:
        standardized_a, transformed_b, disparity = procrustes(np_a, np_b)
    except Exception as e:
        raise RuntimeError(f"SciPy procrustes analysis failed. Error: {e}")

    return standardized_a, transformed_b, disparity
