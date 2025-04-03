#!/usr/bin/env python3
"""
Script to average multiple NIfTI files.
Usage: python average_nifti_files.py <input_directory> <output_file> [--pattern PATTERN]

Example:
python average_nifti_files.py selected_sm6_data/ averaged_data.nii.gz --pattern "*sm6_bold.nii.gz"
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from misc import get_subject_and_story_from_filename

def average_nifti_files(input_dir, output_dir, story_name, pattern="*.nii.gz"):
    """
    Average multiple NIfTI files in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing NIfTI files
    output_file : str
        Path to save the averaged NIfTI file
    pattern : str
        Glob pattern to match files (default: "*.nii.gz")
    """
    file_pattern = os.path.join(input_dir, pattern)
    files = glob.glob(file_pattern)
    output_file = os.path.join(output_dir, f"{story_name}_averaged.nii.gz")
    
    if not files:
        raise ValueError(f"No files found matching pattern '{pattern}' in directory '{input_dir}'")
    
    print(f"Found {len(files)} files to average")
    
    files = [f for f in files if story_name in f]
    print(f"Loading first file to get dimensions: {files[0]}")
    first_nii = nib.load(files[0])
    shape = first_nii.shape
    affine = first_nii.affine
    header = first_nii.header.copy()
    
    data_sum = np.zeros(shape, dtype=np.float32)
    
    print(f"Averaging {len(files)} files")
    for file_path in tqdm(files):
        try:
            img = nib.load(file_path)
            if img.shape != shape:
                print(f"Warning: Skipping {file_path} - shape {img.shape} doesn't match expected shape {shape}")
                continue
            data_sum += img.get_fdata(dtype=np.float32)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    data_avg = data_sum / len(files)
    
    avg_img = nib.Nifti1Image(data_avg, affine, header)
    print(f"Saving averaged image to {output_file}")
    nib.save(avg_img, output_file)
    print("Done!")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Average multiple NIfTI files')
    parser.add_argument('--input_dir', default="selected_sm6_data", help='Directory containing NIfTI files')
    parser.add_argument('--output_dir', default="averaged", help='Directory to save the averaged NIfTI file')
    parser.add_argument('--story_name', default="slumlordreach", help='Story name')
    parser.add_argument('--pattern', default="*.nii.gz", help='Glob pattern to match files (default: "*.nii.gz")')
    
    args = parser.parse_args()
    average_nifti_files(args.input_dir, args.output_dir, args.story_name, args.pattern)