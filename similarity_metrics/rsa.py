import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import pdist, squareform

def rsa_torch(X, Y, metric='correlation', comparison='pearson', batch_size=None, 
              common_type=torch.float32, device="cpu"):
    """
    Perform Representational Similarity Analysis (RSA) between two matrices.
    
    Parameters:
    - X: torch.Tensor of shape (n_samples, n_features_x)
    - Y: torch.Tensor of shape (n_samples, n_features_y)
    - metric: Distance metric to use for computing RDMs
               Options: 'correlation', 'cosine', 'euclidean'
    - comparison: Method to compare RDMs
                  Options: 'pearson', 'spearman', 'kendall'
    - batch_size: If not None, compute distance matrices in batches to save memory
    - common_type: Data type for computation
    - device: Device to perform computation on
    
    Returns:
    - similarity: Similarity score between the two RDMs
    - rdm_x, rdm_y: The representational dissimilarity matrices if return_rdms=True
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
    
    n_samples = X.shape[0]
    
    if n_samples < 2:
        raise ValueError("RSA requires at least 2 samples")
    
    # Compute RDMs (Representational Dissimilarity Matrices)
    rdm_x = compute_rdm(X, metric=metric, batch_size=batch_size)
    rdm_y = compute_rdm(Y, metric=metric, batch_size=batch_size)
    
    # Compare RDMs using the specified method
    similarity = compare_rdms(rdm_x, rdm_y, method=comparison)
    
    return similarity

def compute_rdm(X, metric='correlation', batch_size=None):
    """
    Compute Representational Dissimilarity Matrix (RDM) for a given tensor.
    
    Parameters:
    - X: torch.Tensor of shape (n_samples, n_features)
    - metric: Distance metric ('correlation', 'cosine', 'euclidean')
    - batch_size: If not None, compute in batches to save memory
    
    Returns:
    - rdm: RDM as numpy array of shape (n_samples, n_samples)
    """
    X_np = X.cpu().numpy()
    
    if batch_size is None or X.shape[0] <= batch_size:
        distances = pdist(X_np, metric=metric)
        rdm = squareform(distances)
    else:
        n = X.shape[0]
        rdm = np.zeros((n, n))
        
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            X_batch_i = X_np[i:end_i]
            
            for j in range(0, n, batch_size):
                end_j = min(j + batch_size, n)
                X_batch_j = X_np[j:end_j]
                
                for bi, sample_i in enumerate(X_batch_i):
                    for bj, sample_j in enumerate(X_batch_j):
                        if i + bi != j + bj:
                            if metric == 'correlation':
                                rdm[i + bi, j + bj] = 1 - np.corrcoef(sample_i, sample_j)[0, 1]
                            elif metric == 'cosine':
                                rdm[i + bi, j + bj] = 1 - np.dot(sample_i, sample_j) / (np.linalg.norm(sample_i) * np.linalg.norm(sample_j))
                            elif metric == 'euclidean':
                                rdm[i + bi, j + bj] = np.linalg.norm(sample_i - sample_j)
        
        rdm = (rdm + rdm.T) / 2
    
    return rdm

def compare_rdms(rdm1, rdm2, method='pearson'):
    """
    Compare two RDMs using the specified correlation method.
    
    Parameters:
    - rdm1, rdm2: Two representational dissimilarity matrices
    - method: 'pearson', 'spearman', or 'kendall'
    
    Returns:
    - similarity: Correlation coefficient between the vectorized RDMs
    """
    triu_indices = np.triu_indices(rdm1.shape[0], k=1)
    rdm1_vec = rdm1[triu_indices]
    rdm2_vec = rdm2[triu_indices]
    
    if method == 'pearson':
        similarity, _ = pearsonr(rdm1_vec, rdm2_vec)
    elif method == 'spearman':
        similarity, _ = spearmanr(rdm1_vec, rdm2_vec)
    elif method == 'kendall':
        similarity, _ = kendalltau(rdm1_vec, rdm2_vec)
    else:
        raise ValueError(f"Unknown comparison method: {method}")
    
    return similarity

def rsa_with_rdms(X, Y, **kwargs):
    """
    Perform RSA and return both the similarity score and the RDMs.
    
    Parameters:
    - Same as rsa_torch
    
    Returns:
    - similarity: Similarity score between the two RDMs
    - rdm_x, rdm_y: The representational dissimilarity matrices
    """
    metric = kwargs.get('metric', 'correlation')
    batch_size = kwargs.get('batch_size', None)
    comparison = kwargs.get('comparison', 'pearson')
    X = X.to(kwargs.get('common_type', torch.float32))
    Y = Y.to(kwargs.get('common_type', torch.float32))
    device = kwargs.get('device', 'cpu')
    
    X = X.to(device)
    Y = Y.to(device)
    
    rdm_x = compute_rdm(X, metric=metric, batch_size=batch_size)
    rdm_y = compute_rdm(Y, metric=metric, batch_size=batch_size)
    
    similarity = compare_rdms(rdm_x, rdm_y, method=comparison)
    
    return similarity, rdm_x, rdm_y