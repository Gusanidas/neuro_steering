import torch
# Make sure necessary classes are imported
from datasets import Dataset, DatasetInfo, Features, Value, Sequence
import numpy as np # Needed for dtype checking if refining features further
import os
from typing import List, Dict, Any, Tuple
import datetime # To add a timestamp to the description

# Assume a default version for the dataset structure
DATASET_VERSION = "1.0.0"

def save_dataset(neural_data: torch.Tensor, activations_data: torch.Tensor,
                 metadata: Dict[str, Any] = None, # Dataset-level metadata dictionary
                 time_intervals: List[Tuple[float, float]] = None,
                 dataset_name: str = "neural_llm_alignment",
                 hf_token: str = None) -> None:
    """
    Save neural data and activations as a Hugging Face dataset and upload it.
    Includes dataset-level metadata (DatasetInfo).

    Args:
        neural_data: Tensor containing neural data (expected shape: [n_samples, ...])
        activations_data: Tensor containing activation data (expected shape: [n_samples, ...])
        metadata: Dictionary containing dataset-level metadata to be included in DatasetInfo
        time_intervals: List of time intervals (start, end) for each datapoint
        dataset_name: Base name for the dataset on Hugging Face
        hf_token: Hugging Face API token for authentication
    """
    print(f"Preparing to save dataset '{dataset_name}'...")

    neural_np = neural_data.detach().cpu().numpy()
    activations_np = activations_data.detach().cpu().numpy()

    n_samples = len(neural_np)

    data_dict = {
        "neural_data": [neural_np[i] for i in range(n_samples)],
        "activation_data": [activations_np[i] for i in range(n_samples)],
    }

    if time_intervals:
        if len(time_intervals) != n_samples:
            print(f"Warning: Mismatch between number of samples ({n_samples}) and time intervals ({len(time_intervals)}). Omitting time intervals.")
        else:
            data_dict["start_time"] = [interval[0] for interval in time_intervals]
            data_dict["end_time"] = [interval[1] for interval in time_intervals]

    data_dict["index"] = list(range(n_samples))

    feature_dict = {
        "neural_data": Sequence(feature=Value(dtype=str(neural_np.dtype))), # Use actual dtype
        "activation_data": Sequence(feature=Value(dtype=str(activations_np.dtype))), # Use actual dtype
        "index": Value(dtype='int64')
    }
    if "start_time" in data_dict:
        feature_dict["start_time"] = Value(dtype='float64')
        feature_dict["end_time"] = Value(dtype='float64')

    features_schema = Features(feature_dict)
    print("\nDefined Dataset Features:")
    print(features_schema)

    # Create dataset info as a dictionary
    info_dict = {
        "version": DATASET_VERSION,
        "homepage": "https://example.com/your_project_homepage",
        "citation": "@misc{your_citation_key_here,\n author = {Your Name/Lab},\n title = {Neural-LLM Alignment Data},\n year = {2025}\n}",
        "features": features_schema
    }
    
    # Directly use the metadata dictionary for dataset info
    if metadata:
        # Add all provided metadata keys to the info_dict
        for key, value in metadata.items():
            info_dict[key] = value
            
        # If description is not provided in metadata, create one from available metadata
        if 'description' not in metadata:
            description_parts = []
            for key, value in metadata.items():
                if key not in ['version', 'homepage', 'citation', 'features']:
                    if isinstance(value, list):
                        formatted_value = ', '.join(str(v) for v in value)
                    else:
                        formatted_value = str(value)
                    description_parts.append(f"{key}: {formatted_value}")
            
            if description_parts:
                info_dict['description'] = "Dataset containing neural recordings and model activations.\n" + "\n".join(description_parts)
    
    print("\nCreated DatasetInfo Dictionary:")
    print(info_dict)

    # Create DatasetInfo from dictionary
    info = DatasetInfo.from_dict(info_dict)
    print("\nConverted to DatasetInfo object.")

    try:
        dataset = Dataset.from_dict(data_dict, info=info, features=features_schema)
        print("\nDataset object created successfully.")
    except Exception as e:
        print(f"\nError creating Hugging Face Dataset object: {e}")
        return 


    if hf_token:
        print(f"\nAttempting to upload to Hugging Face Hub as '{dataset_name}'...")
        try:
            dataset.push_to_hub(
                repo_id=dataset_name,
                token=hf_token,
                private=True
            )
            print(f"Dataset successfully uploaded to Hugging Face Hub as '{dataset_name}'")
        except Exception as e:
            print(f"Error uploading dataset to Hugging Face Hub: {e}")
            print("Dataset was saved locally but not uploaded.")
    else:
        print("\nNo Hugging Face token provided.")
