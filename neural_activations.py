from misc import save_dataset, get_subject_and_story_from_filename, pca_pytorch, map_region_to_harvard_oxford
import os
import torch
from transformer_lens import HookedTransformer
from time_interval_iterator.time_interval_neural_data import TimeIntervalNeuralData
from time_interval_iterator.time_interval_tokenized_story import TimeIntervalTokenizedStory
from time_interval_iterator.time_interval_activations import TimeIntervalActivations
from time_interval_iterator.combined_iterator import CombinedIterator
from time_interval_iterator.time_interval_iterator_list import TimeIntervalIteratorList
from typing import List, Tuple, Callable
from similarity_metrics import linear_cka_efficient, procrustes_pytorch, rsa_with_rdms, cca_similarity
from transformers import AutoTokenizer
from functools import partial

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face token from environment variables
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    print("Hugging Face token loaded successfully")
else:
    print("Warning: HUGGINGFACE_TOKEN not found in .env file")


# Model definition
model_name = "gemma-2-2b"
tokenizer_name = "google/gemma-2-2b"
model = HookedTransformer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
temporal_bias_activations = 0.0
hook_names = model.hook_dict.keys()
for i, hook_name in enumerate(hook_names):
    print(f"{i}: {hook_name}")
print("--------------------------------")

def extract_mlp_layers(cache, layer_indices):
    layers = [cache[f"blocks.{i}.mlp.hook_pre"] for i in layer_indices]
    return torch.cat(layers, dim=-1)

def extract_layers(cache, layer_indices):
    layers = [cache[f"blocks.{i}.hook_resid_post"] for i in layer_indices]
    return torch.cat(layers, dim=-1)

def extract_attn_layers(cache, layer_indices):
    layers = [cache[f"blocks.{i}.hook_attn_out"] for i in layer_indices]
    return torch.cat(layers, dim=-1)

def extract_combined_layers(cache, layer_indices):
    layers = [cache[f"blocks.{i}.hook_resid_post"] for i in layer_indices]
    layers += [cache[f"blocks.{i}.hook_attn_out"] for i in layer_indices]
    layers += [cache[f"blocks.{i}.hook_mlp_out"] for i in layer_indices]
    return torch.cat(layers, dim=-1)

activation_extractor = partial(extract_attn_layers, layer_indices=[21])

# Common variables
story_base_dir = "gentle"
neural_base_dir = ""
tr_value = 1.5
max_files = 6

# Neural variables
brain_region = "temporal"
nifti_dir = "selected_sm6_data"
#nifti_dir = "averaged"
temporal_bias_neural = 0.0


# Function to filter files in nifti_dir
def filter_nifti_files(filename: str) -> bool:
    return "sub-111" in filename

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

def create_neural_activation_iterators() -> TimeIntervalIteratorList:
    """
    Create a TimeIntervalIteratorList from neural data and model activations.
    For each neural data file, process the corresponding story and extract activations.
    
    Returns:
        TimeIntervalIteratorList combining all iterators
    """
    files = get_valid_filenames(nifti_dir, filter_nifti_files)
    print(f"Found {len(files)} valid files in neural directory")
    
    combined_iterators = []
    
    for i, file_path in enumerate(files):
        if i >= max_files:
            break
        
        try:    
            subject_id, story_name = get_subject_and_story_from_filename(file_path)
            #story_name = "black"
            
            neural_data = TimeIntervalNeuralData(
                story_name=story_name,
                subject=subject_id,
                region=brain_region,
                file_path=file_path,
                TR=tr_value,
                temporal_bias=temporal_bias_neural,
                base_dir=neural_base_dir
            )
            
            tokenized_story = TimeIntervalTokenizedStory(
                story_name=story_name,
                tokenizer=tokenizer,
                base_dir=story_base_dir
            )
            
            activations = TimeIntervalActivations(
                activations=None,
                activation_to_time_map=None,
                story_name=story_name,
                activation_name="layers_4_5_6_resid_post",
                activation_extractor=activation_extractor,
                story=tokenized_story,
                model=model,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            combined_iterator = CombinedIterator(neural_data, activations)
            combined_iterators.append(combined_iterator)
            
            print(f"Created combined iterator for {file_path} and story '{story_name}'")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    if combined_iterators:
        return TimeIntervalIteratorList(combined_iterators)
    else:
        raise ValueError("No valid iterator pairs found")

if __name__ == "__main__":
    iterator_list = create_neural_activation_iterators()
    print(f"Created TimeIntervalIteratorList with combined iterators")
    
    neural_data = []
    activations_data = []
    intervals = []
    
    for t1, t2, x in iterator_list:
        n, a = x
        neural_data.append(n)
        activations_data.append(a)
        intervals.append((t1, t2))
    
    neural_data = torch.stack(neural_data, dim=0)
    activations_data = torch.stack(activations_data, dim=0)

    metadata = {"model": model_name, "region": brain_region, "activation": "attn 21", "subject": "111"}
    save_dataset(neural_data=neural_data, activations_data=activations_data, time_intervals=intervals, dataset_name="neural_activations_1", hf_token=hf_token, metadata=metadata)
    
    print(f"Shape of neural_data: {neural_data.shape}")
    print(f"Shape of activations_data: {activations_data.shape}")
    
    # Apply PCA to reduce dimensionality
    d = 64
    neural_pca = pca_pytorch(neural_data, d)
    activations_pca = pca_pytorch(activations_data, d)
    
    print(f"Shape of neural_pca: {neural_pca.shape}")
    print(f"Shape of activations_pca: {activations_pca.shape}")
    
    # Compute similarity metrics
    cka = linear_cka_efficient(neural_pca, activations_pca)
    print(f"CKA: {cka}")
    
    _, _, procrustes_result = procrustes_pytorch(neural_pca, activations_pca)
    print(f"Procrustes: {procrustes_result}")
    
    rsa_result, _, _ = rsa_with_rdms(neural_pca, activations_pca)
    print(f"RSA: {rsa_result}")
    
    cca_result = cca_similarity(neural_pca, activations_pca)
    print(f"CCA: {cca_result}")
