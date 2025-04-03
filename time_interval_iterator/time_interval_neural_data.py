import nibabel as nib
from nilearn import image, datasets
import numpy as np
import os
import torch
from typing import Union, List, Tuple, Optional, Callable, Dict
from misc.map_region_to_harvard_oxford import map_region_to_harvard_oxford

from time_interval_iterator.time_interval_iterator import TimeIntervalIterator


class TimeIntervalNeuralData(TimeIntervalIterator):
    def __init__(
        self,
        story_name: str,
        subject: str,
        region: Union[str, List[str]],
        file_path: str,
        TR: float = 1.5,
        temporal_bias: float = 0.0,
        region_to_label_func: Callable = map_region_to_harvard_oxford,
        base_dir: str = "",
        voxels: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ):
        """
        Initialize an TimeIntervalNeuralData object for brain imaging analysis.

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

        self.story_name = story_name
        self.subject = subject
        self.region = region
        self.file_path = file_path
        self.TR = TR
        self.affine = affine
        self.mask = mask
        self.region_to_label_func = region_to_label_func
        self.temporal_bias = temporal_bias
        self.base_dir = base_dir

        self.initialized = False

        if voxels is not None:
            self._set_voxels(voxels)

    def _set_voxels(self, voxels: np.ndarray):
        print(f"set voxels, voxels shape: {voxels.shape}")
        TR = self.TR
        temporal_bias = self.temporal_bias
        self.voxels = voxels
        self.initialized = True

        element_to_time_map = {}
        n_timepoints = voxels.shape[0]
        for i in range(n_timepoints):
            start_time = i * TR
            end_time = (i + 1) * TR
            element_to_time_map[i] = (start_time, end_time)
        voxels_tensor = torch.tensor(voxels)
        self.n_timepoints = n_timepoints
        self.n_voxels = voxels.shape[1]
        super().__init__(
            data=voxels_tensor,
            element_to_time_map=element_to_time_map,
            temporal_bias=temporal_bias,
            )

    def _initialize(self):
        if self.initialized:
            return
        self._build_from_file(
            story_name=self.story_name,
            subject=self.subject,
            region=self.region,
            file_path=self.file_path,
            region_to_label_func=self.region_to_label_func,
            TR=self.TR,
            base_dir=self.base_dir,
            temporal_bias=self.temporal_bias,
        )
        self.initialized = True

    def _ensure_initialized(self):
        if not self.initialized:
            self._initialize()

    def time_to_index(self, seconds: float) -> int:
        print(f"initialized? {self.initialized}")
        self._ensure_initialized()
        print(f"initialized? {self.initialized}")
        return self.get_element_index_for_time_seconds(seconds)

    def get_element_index_for_time_seconds(self, seconds: float) -> int:
        self._ensure_initialized()
        return super().get_element_index_for_time_seconds(seconds)

    def index_to_time(self, index: int) -> Tuple[float, float]:
        self._ensure_initialized()
        return self.get_time_seconds_for_element(index)

    def get_time_seconds_for_element(self, element_idx: int) -> Tuple[float, float]:
        self._ensure_initialized()
        return super().get_time_seconds_for_element(element_idx)

    def get_voxel_timeseries(self, voxel_idx: int) -> np.ndarray:
        """
        Get time series for a specific voxel.

        Args:
            voxel_idx (int): Voxel index

        Returns:
            numpy.ndarray: Time series data for the voxel
        """
        self._ensure_initialized()
        if voxel_idx < 0 or voxel_idx >= self.n_voxels:
            raise ValueError(
                f"Voxel index {voxel_idx} out of range (0-{self.n_voxels-1})"
            )

        return self.data[:, voxel_idx].numpy()

    def __iter__(self):
        """
        Override the parent's __iter__ method to yield scan data by timepoint.
        Each iteration produces a tuple of (start_time, end_time, voxel_data).
        Voxel data is the entire array of voxel values for one timepoint.
        """
        self._ensure_initialized()
        sorted_indices = sorted(self.element_to_time_map.keys())
        for element_idx in sorted_indices:
            start_time, end_time = self.element_to_time_map[element_idx]
            # For neural data, we return the entire row (all voxels at this timepoint)
            yield start_time, end_time, self.data[element_idx]

    def __next__(self):
        self._ensure_initialized()
        return super().__next__()

    def _build_from_file(
        self,
        story_name: str,
        subject: str,
        region: Union[str, List[str]],
        file_path: str,
        region_to_label_func: Callable,
        TR: float = 1.5,
        base_dir: str = "",
        temporal_bias: float = 0.0,
    ):
        """
        Build an TimeIntervalNeuralData object from a NIfTI file.

        Args:
            story_name (str): Name of the story
            subject (str): Subject ID
            region (str or list): Brain region(s) of interest
            file_path (str): Path to the NIfTI file
            region_to_label_func (callable): Function that maps brain region names to atlas labels
            TR (float, optional): Repetition time in seconds. Defaults to 1.5.
            base_dir (str, optional): Base directory. Defaults to "".
            temporal_bias (float, optional): Temporal bias to apply to the time intervals. Defaults to 0.0.
        Returns:
            TimeIntervalNeuralData: A new TimeIntervalNeuralData object
        """
        full_path = os.path.join(base_dir, file_path)

        # Load the NIfTI file
        img = nib.load(full_path)
        data = img.get_fdata()
        affine = img.affine

        # Get atlas and create mask for the specified region
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        atlas_img = atlas["maps"]
        atlas_data = atlas_img.get_fdata()

        # Get region labels
        region_labels = {}
        if isinstance(region, str):
            regions = [region]
        else:
            regions = region

        print(f"regions: {regions}")
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
        resampled_mask = image.resample_to_img(mask_img, img, interpolation="nearest")
        mask_resampled = resampled_mask.get_fdata() > 0

        print(f"mask_resampled.shape: {mask_resampled.shape}")

        # Extract the voxels within the region of interest
        n_timepoints = data.shape[3]
        n_voxels = np.sum(mask_resampled)

        # Create time series for all voxels in the region
        voxel_timeseries = np.zeros((n_timepoints, n_voxels))
        for t in range(n_timepoints):
            voxel_timeseries[t, :] = data[:, :, :, t][mask_resampled > 0]

        self._set_voxels(voxel_timeseries)


if __name__ == "__main__":
    story_name = "black"
    subject = "sub001"
    region = "temporal"
    file_path = "selected_data_1/sub-001_task-pieman_run-1_space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz"
    TR = 1.5
    base_dir = ""

    region_to_label_func = map_region_to_harvard_oxford

    print(
        f"Testing AlignedNeuralData with story: {story_name}, subject: {subject}, region: {region}"
    )

    try:
        neural_data = TimeIntervalNeuralData(
            story_name=story_name,
            subject=subject,
            region=region,
            file_path=file_path,
            region_to_label_func=region_to_label_func,
            TR=TR,
            base_dir=base_dir,
        )

        print(f"\nSuccessfully created TimeIntervalNeuralData object:")
        print(f"  - Story: {neural_data.story_name}")
        print(f"  - Subject: {neural_data.subject}")
        print(f"  - Region: {neural_data.region}")
        print(f"  - File: {neural_data.file_path}")
        print(f"  - TR: {neural_data.TR} seconds")

        # Test time conversion methods
        test_time = 10.5  # seconds
        test_index = neural_data.time_to_index(test_time)
        print(f"\nTime conversion test:")
        print(f"  - Time {test_time}s maps to index {test_index}")
        start_time, end_time = neural_data.index_to_time(test_index)
        print(
            f"  - Index {test_index} maps to time interval [{start_time:.2f}s, {end_time:.2f}s]"
        )

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
            print(
                f"  - Timepoint {i}: [{start:.2f}s, {end:.2f}s], voxels shape: {voxels.shape}"
            )
            if i >= 2:  # Only show first few iterations
                print("  - ...")
                break

    except FileNotFoundError:
        print(f"Error: Could not find the file at {os.path.join(base_dir, file_path)}")
        print("This is expected if running the example without the actual data files.")
        print("To use this class, provide valid paths to your neuroimaging data.")
    except Exception as e:
        print(f"Error: {str(e)}")

