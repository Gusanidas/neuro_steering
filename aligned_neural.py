import nibabel as nib
import numpy as np
from nilearn import datasets, image, plotting
import matplotlib.pyplot as plt
import os
import torch
from typing import Union, List, Tuple, Optional, Callable, Dict
from map_region import map_region_to_harvard_oxford

from aligned_iterator import AlignedIterator

class AlignedNeuralData(AlignedIterator):
    def __init__(
        self,
        story_name: str,
        subject: str,
        region: Union[str, List[str]],
        voxels: np.ndarray,
        filename: str,
        TR: float = 1.5,
        affine: np.ndarray = None,
        mask: np.ndarray = None
    ):
        """
        Initialize an AlignedNeuralData object for brain imaging analysis.
        
        Args:
            story_name (str): Name of the story stimulus
            subject (str): Subject ID
            region (str or list): Brain region(s) of interest
            voxels (numpy.ndarray): Extracted voxel time series data (time × voxels)
            filename (str): Path to the NIfTI file
            TR (float, optional): Repetition time in seconds. Defaults to 1.5.
            affine (numpy.ndarray, optional): Affine transformation matrix. Defaults to None.
            mask (numpy.ndarray, optional): Binary mask for the selected brain region. Defaults to None.
        """
        # Create element to time mapping based on TR
        element_to_time_map = {}
        n_timepoints = voxels.shape[0]
        
        for i in range(n_timepoints):
            start_time = i * TR
            end_time = (i + 1) * TR
            element_to_time_map[i] = (start_time, end_time)
        
        # Convert numpy array to tensor for consistency with AlignedIterator
        voxels_tensor = torch.tensor(voxels)
        
        # Initialize the parent class
        super().__init__(data=voxels_tensor, element_to_time_map=element_to_time_map)
        
        # Store neural data specific attributes
        self.story_name = story_name
        self.subject = subject
        self.region = region
        self.filename = filename
        self.TR = TR
        self.affine = affine
        self.mask = mask
        self.n_timepoints = n_timepoints
        self.n_voxels = voxels.shape[1]
    
    # Alias methods for backward compatibility
    def time_to_index(self, seconds: float) -> int:
        """
        Convert time in seconds to the appropriate scan index.
        Alias for get_element_for_time_seconds for backward compatibility.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            int: The index of the scan at that time
        """
        return self.get_element_for_time_seconds(seconds)
    
    def index_to_time(self, index: int) -> Tuple[float, float]:
        """
        Convert scan index to time interval in seconds.
        Alias for get_time_seconds_for_element for backward compatibility.
        
        Args:
            index (int): Scan index
            
        Returns:
            Tuple[float, float]: The (start_time, end_time) interval for the scan
        """
        return self.get_time_seconds_for_element(index)
    
    def get_voxel_timeseries(self, voxel_idx: int) -> np.ndarray:
        """
        Get time series for a specific voxel.
        
        Args:
            voxel_idx (int): Voxel index
            
        Returns:
            numpy.ndarray: Time series data for the voxel
        """
        if voxel_idx < 0 or voxel_idx >= self.n_voxels:
            raise ValueError(f"Voxel index {voxel_idx} out of range (0-{self.n_voxels-1})")
        
        # Convert the tensor slice back to numpy for backward compatibility
        return self.data[:, voxel_idx].numpy()
    
    
    def __iter__(self):
        """
        Override the parent's __iter__ method to yield scan data by timepoint.
        Each iteration produces a tuple of (start_time, end_time, voxel_data).
        Voxel data is the entire array of voxel values for one timepoint.
        """
        sorted_indices = sorted(self.element_to_time_map.keys())
        for element_idx in sorted_indices:
            start_time, end_time = self.element_to_time_map[element_idx]
            # For neural data, we return the entire row (all voxels at this timepoint)
            yield start_time, end_time, self.data[element_idx]
    
    @classmethod
    def build_from_file(
        cls,
        story_name: str,
        subject: str,
        region: Union[str, List[str]],
        file_path: str,
        region_to_label_func: Callable,
        TR: float = 1.5,
        base_dir: str = ""
    ):
        """
        Build an AlignedNeuralData object from a NIfTI file.
        
        Args:
            story_name (str): Name of the story
            subject (str): Subject ID
            region (str or list): Brain region(s) of interest
            file_path (str): Path to the NIfTI file
            region_to_label_func (callable): Function that maps brain region names to atlas labels
            TR (float, optional): Repetition time in seconds. Defaults to 1.5.
            base_dir (str, optional): Base directory. Defaults to "".
            
        Returns:
            AlignedNeuralData: A new AlignedNeuralData object
        """
        # Construct full path
        full_path = os.path.join(base_dir, file_path)
        
        # Load the NIfTI file
        img = nib.load(full_path)
        data = img.get_fdata()
        affine = img.affine
        
        # Get atlas and create mask for the specified region
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = atlas['maps']
        atlas_data = atlas_img.get_fdata()
        
        # Get region labels
        region_labels = {}
        if isinstance(region, str):
            regions = [region]
        else:
            regions = region
            
        for r in regions:
            label_indices = region_to_label_func(r, atlas)
            for label, idx in label_indices.items():
                region_labels[label] = idx
        
        # Create binary mask for the region
        region_mask = np.zeros_like(atlas_data)
        for label, idx in region_labels.items():
            region_mask[atlas_data == idx] = 1
        
        # Resample mask to match functional data dimensions
        mask_img = nib.Nifti1Image(region_mask, atlas_img.affine)
        resampled_mask = image.resample_to_img(mask_img, img, interpolation='nearest')
        mask_resampled = resampled_mask.get_fdata() > 0
        
        # Extract the voxels within the region of interest
        n_timepoints = data.shape[3]
        n_voxels = np.sum(mask_resampled)
        
        # Create time series for all voxels in the region
        voxel_timeseries = np.zeros((n_timepoints, n_voxels))
        for t in range(n_timepoints):
            voxel_timeseries[t, :] = data[:,:,:,t][mask_resampled > 0]
        
        return cls(
            story_name=story_name,
            subject=subject,
            region=region,
            voxels=voxel_timeseries,
            filename=full_path,
            TR=TR,
            affine=affine,
            mask=mask_resampled
        )

if __name__ == "__main__":
    # Example usage
    story_name = "black"
    subject = "sub001"
    region = "temporal"
    file_path = "derivatives/afni-smooth/sub-015/func/sub-015_task-pieman_run-1_space-MNI152NLin2009cAsym_res-native_desc-sm6_bold.nii.gz"
    TR = 1.5
    base_dir = "narratives"
    
    # Map region to label indices
    region_to_label_func = map_region_to_harvard_oxford
    
    print(f"Testing AlignedNeuralData with story: {story_name}, subject: {subject}, region: {region}")
    
    try:
        # Build the AlignedNeuralData object
        neural_data = AlignedNeuralData.build_from_file(
            story_name=story_name,
            subject=subject,
            region=region,
            file_path=file_path,
            region_to_label_func=region_to_label_func,
            TR=TR,
            base_dir=base_dir
        )
        
        # Print basic information about the object
        print(f"\nSuccessfully created AlignedNeuralData object:")
        print(f"  - Story: {neural_data.story_name}")
        print(f"  - Subject: {neural_data.subject}")
        print(f"  - Region: {neural_data.region}")
        print(f"  - File: {neural_data.filename}")
        print(f"  - Dimensions: {neural_data.n_timepoints} timepoints × {neural_data.n_voxels} voxels")
        print(f"  - TR: {neural_data.TR} seconds")
        
        # Test time conversion methods
        test_time = 10.5  # seconds
        test_index = neural_data.time_to_index(test_time)
        print(f"\nTime conversion test:")
        print(f"  - Time {test_time}s maps to index {test_index}")
        start_time, end_time = neural_data.index_to_time(test_index)
        print(f"  - Index {test_index} maps to time interval [{start_time:.2f}s, {end_time:.2f}s]")
        
        # Test voxel timeseries extraction
        if neural_data.n_voxels > 0:
            print(f"\nVoxel timeseries test:")
            voxel_idx = 0  # First voxel
            timeseries = neural_data.get_voxel_timeseries(voxel_idx)
            print(f"  - Voxel {voxel_idx} timeseries shape: {timeseries.shape}")
            print(f"  - First 5 values: {timeseries[:5]}")
        
        # Test iteration
        print(f"\nIteration test:")
        for i, (start, end, voxels) in enumerate(neural_data):
            print(f"  - Timepoint {i}: [{start:.2f}s, {end:.2f}s], voxels shape: {voxels.shape}")
            if i >= 2:  # Only show first few iterations
                print("  - ...")
                break
                
    except FileNotFoundError:
        print(f"Error: Could not find the file at {os.path.join(base_dir, file_path)}")
        print("This is expected if running the example without the actual data files.")
        print("To use this class, provide valid paths to your neuroimaging data.")
    except Exception as e:
        print(f"Error: {str(e)}")