from misc.get_subject_and_story_from_filename import get_subject_and_story_from_filename
from misc.pca_pytorch import pca_pytorch
import os
from time_interval_iterator.time_interval_neural_data import TimeIntervalNeuralData
from time_interval_iterator.combined_iterator import CombinedIterator
from time_interval_iterator.time_interval_iterator_list import TimeIntervalIteratorList
from typing import List, Tuple
import torch
from similarity_metrics import linear_cka_efficient, procrustes_pytorch, rsa_with_rdms, cca_similarity
# Common variables
story_base_dir = "gentle"
neural_base_dir = ""
tr_value = 1.5
max_files = 10

# Neural 1 variables
brain_region_1 = "temporal"
nifti_dir_1 = "selected_sm6_data"
temporal_bias_1 = 0.0

# Function to filter files in nifti_dir_1
def filter_nifti_files_1(filename: str) -> bool:
    return True

# Neural 2 variables
brain_region_2 = "frontal"
nifti_dir_2 = "selected_sm6_data"
temporal_bias_2 = 0.0

# Function to filter files in nifti_dir_2
def filter_nifti_files_2(filename: str) -> bool:
    return True

def get_valid_filenames(directory: str, filter_func) -> List[str]:
    """
    Get a list of valid filenames in the given directory.
    
    Args:
        directory: The directory path relative to neural_base_dir
        filter_func: Function to filter filenames
        
    Returns:
        List of valid filenames with full paths
    """
    full_dir_path = os.path.join(neural_base_dir, directory)
    result = []
    
    try:
        for filename in os.listdir(full_dir_path):
            if filename.endswith('.nii.gz') and filter_func(filename):
                result.append(os.path.join(directory, filename))
    except FileNotFoundError:
        print(f"Directory not found: {full_dir_path}")
    
    return result


def create_neural_iterators() -> TimeIntervalIteratorList:
    """
    Create a TimeIntervalIteratorList from neural data files.
    Pairs files 1-to-1 in order (first with first, second with second, etc.)
    
    Returns:
        TimeIntervalIteratorList combining all the iterators
    """
    files_1 = get_valid_filenames(nifti_dir_1, filter_nifti_files_1)
    files_2 = get_valid_filenames(nifti_dir_2, filter_nifti_files_2)
    
    print(f"Found {len(files_1)} valid files in neural 1 directory")
    print(f"Found {len(files_2)} valid files in neural 2 directory")
    
    combined_iterators = []
    
    for i, (file_1, file_2) in enumerate(zip(files_1, files_2)):
        if i >= max_files:
            break
        subject_id, story_name = get_subject_and_story_from_filename(file_1)
        
        neural_data_1 = TimeIntervalNeuralData(
            story_name=story_name,
            subject=subject_id,
            region=brain_region_1,
            file_path=file_1,
            TR=tr_value,
            temporal_bias=temporal_bias_1,
            base_dir=neural_base_dir
        )
        
        neural_data_2 = TimeIntervalNeuralData(
            story_name=story_name,
            subject=subject_id,
            region=brain_region_2,
            file_path=file_2,
            TR=tr_value,
            temporal_bias=temporal_bias_2,
            base_dir=neural_base_dir
        )
        
        combined_iterator = CombinedIterator(neural_data_1, neural_data_2)
        combined_iterators.append(combined_iterator)
        
        print(f"Created combined iterator for {file_1} and {file_2}")
    
    if combined_iterators:
        return TimeIntervalIteratorList(combined_iterators)
    else:
        raise ValueError("No valid iterator pairs found")

if __name__ == "__main__":
    # Create TimeIntervalIteratorList with all combined iterators
    iterator_list = create_neural_iterators()
    print(f"Created TimeIntervalIteratorList with combined iterators")
    neural_data_1 = []
    neural_data_2 = []
    for t1, t2, x in iterator_list:
        n1, n2 = x
        neural_data_1.append(n1)
        neural_data_2.append(n2)
    neural_data_1 = torch.stack(neural_data_1, dim=0)
    neural_data_2 = torch.stack(neural_data_2, dim=0)
    print(f"shape of neural_data_1: {neural_data_1.shape}")
    print(f"shape of neural_data_2: {neural_data_2.shape}")
    d = 64
    pca_1 = pca_pytorch(neural_data_1, d)
    pca_2 = pca_pytorch(neural_data_2, d)
    print(f"shape of pca_1: {pca_1.shape}")
    print(f"shape of pca_2: {pca_2.shape}")
    cka = linear_cka_efficient(pca_1, pca_2)
    print(f"CKA: {cka}")
    _, _, procrustes_result = procrustes_pytorch(pca_1, pca_2)
    print(f"Procrustes: {procrustes_result}")
    rsa_result, _, _ = rsa_with_rdms(pca_1, pca_2)
    print(f"RSA: {rsa_result}")
    cca_result = cca_similarity(pca_1, pca_2)
    print(f"CCA: {cca_result}")
