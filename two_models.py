from misc.pca_pytorch import pca_pytorch
import os
import torch
from transformer_lens import HookedTransformer
from time_interval_iterator.time_interval_tokenized_story import TimeIntervalTokenizedStory
from time_interval_iterator.time_interval_activations import TimeIntervalActivations
from time_interval_iterator.combined_iterator import CombinedIterator
from time_interval_iterator.time_interval_iterator_list import TimeIntervalIteratorList
from typing import List, Callable
from similarity_metrics import linear_cka_efficient, procrustes_pytorch, rsa_with_rdms, cca_similarity
from transformers import AutoTokenizer
from functools import partial

# Model 1 definition
model1_name = "gemma-2-2b"
tokenizer1_name = "google/gemma-2-2b"
model1 = HookedTransformer.from_pretrained(model1_name)
tokenizer1 = AutoTokenizer.from_pretrained(tokenizer1_name)

# Model 2 definition
model2_name = "Qwen/Qwen2.5-1.5B"
tokenizer2_name = "Qwen/Qwen2.5-1.5B"
model2 = HookedTransformer.from_pretrained(model2_name)
tokenizer2 = AutoTokenizer.from_pretrained(tokenizer2_name)

# Activation extraction functions
def extract_mlp_layers(cache, layer_indices):
    layers = [cache[f"blocks.{i}.mlp.hook_pre"] for i in layer_indices]
    return torch.cat(layers, dim=-1)

def extract_layers(cache, layer_indices):
    layers = [cache[f"blocks.{i}.hook_resid_post"] for i in layer_indices]
    return torch.cat(layers, dim=-1)

def extract_attn_layers(cache, layer_indices):
    layers = [cache[f"blocks.{i}.hook_attn_out"] for i in layer_indices]
    return torch.cat(layers, dim=-1)

# Define specific extractors for each model
activation_extractor1 = partial(extract_attn_layers, layer_indices=[22])
activation_extractor2 = partial(extract_attn_layers, layer_indices=[10])

# Common variables
story_base_dir = "gentle"
max_files = 4
stories = ["pieman", "milkywayoriginal", "black", "shapesphysical"]


def create_two_model_activations_iterators(stories: List[str]) -> TimeIntervalIteratorList:
    """
    Create a TimeIntervalIteratorList from activations of two different models.
    For each story, extract activations from both models and create a combined iterator.
    
    Returns:
        TimeIntervalIteratorList combining all iterators
    """
    
    combined_iterators = []
    
    for i, story_name in enumerate(stories):
        if i >= max_files:
            break
        
        try:
            # Create tokenized story for both models (they may tokenize differently)
            tokenized_story1 = TimeIntervalTokenizedStory(
                story_name=story_name,
                tokenizer=tokenizer1,
                base_dir=story_base_dir
            )
            
            tokenized_story2 = TimeIntervalTokenizedStory(
                story_name=story_name,
                tokenizer=tokenizer2,
                base_dir=story_base_dir
            )
            
            # Extract activations from model 1
            activations1 = TimeIntervalActivations(
                activations=None,
                activation_to_time_map=None,
                story_name=story_name,
                activation_name="model1_activations",
                activation_extractor=activation_extractor1,
                story=tokenized_story1,
                model=model1,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Extract activations from model 2
            activations2 = TimeIntervalActivations(
                activations=None,
                activation_to_time_map=None,
                story_name=story_name,
                activation_name="model2_activations",
                activation_extractor=activation_extractor2,
                story=tokenized_story2,
                model=model2,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Create combined iterator from both models' activations
            combined_iterator = CombinedIterator(activations1, activations2)
            combined_iterators.append(combined_iterator)
            
            print(f"Created combined iterator for '{story_name}' with both models")
        except Exception as e:
            print(f"Error processing story {story_name}: {e}")
            continue
    
    if combined_iterators:
        return TimeIntervalIteratorList(combined_iterators)
    else:
        raise ValueError("No valid iterator pairs found")

if __name__ == "__main__":
    # Create TimeIntervalIteratorList with all combined iterators
    iterator_list = create_two_model_activations_iterators(stories)
    print(f"Created TimeIntervalIteratorList with combined iterators")
    
    activations_data1 = []
    activations_data2 = []
    
    for t1, t2, x in iterator_list:
        a1, a2 = x
        activations_data1.append(a1)
        activations_data2.append(a2)
    
    activations_data1 = torch.stack(activations_data1, dim=0)
    activations_data2 = torch.stack(activations_data2, dim=0)
    
    print(f"Shape of activations_data1: {activations_data1.shape}")
    print(f"Shape of activations_data2: {activations_data2.shape}")
    
    # Apply PCA to reduce dimensionality
    d = 64
    activations1_pca = pca_pytorch(activations_data1, d)
    activations2_pca = pca_pytorch(activations_data2, d)
    
    print(f"Shape of activations1_pca: {activations1_pca.shape}")
    print(f"Shape of activations2_pca: {activations2_pca.shape}")
    
    # Compute similarity metrics
    cka = linear_cka_efficient(activations1_pca, activations2_pca)
    print(f"CKA: {cka}")
    
    _, _, procrustes_result = procrustes_pytorch(activations1_pca, activations2_pca)
    print(f"Procrustes: {procrustes_result}")
    
    rsa_result, _, _ = rsa_with_rdms(activations1_pca, activations2_pca)
    print(f"RSA: {rsa_result}")
    
    cca_result = cca_similarity(activations1_pca, activations2_pca)
    print(f"CCA: {cca_result}")
