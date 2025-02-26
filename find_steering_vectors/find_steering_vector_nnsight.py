#google/gemma-2-2b
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from nnsight import LanguageModel
import torch
from typing import Tuple, List, Union


def get_avg_and_last_token_activations_gemma(
    model: LanguageModel,
    prompts: Union[str, List[str]],
    tail_tokens: int = 2,
    hidden_dim: int = 2304,
    n_layers: int = 25,
    start: int = 0,
    batch_size: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate average and last token activations for Gemma model layers.

    Args:
        model (LanguageModel): The Gemma language model instance.
        prompts (Union[str, List[str]]): Input text prompt(s) to process.
        tail_tokens (int, optional): Number of tokens from the end to consider. Defaults to 2.
        hidden_dim (int, optional): Hidden dimension size of the model. Defaults to 2304.
        n_layers (int, optional): Number of layers in the model. Defaults to 25.
        start (int, optional): Starting position for averaging activations. Defaults to 0.
        batch_size (int, optional): Number of prompts to process in each batch. Defaults to 8.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - avg_activation: Average activations across all layers (shape: n_layers x hidden_dim)
            - key_activation: Average activations of tail tokens across all layers (shape: n_layers x hidden_dim)
    """
    # Convert single string prompt to list for consistent handling
    if isinstance(prompts, str):
        prompts = [prompts]
    
    avg_activation: torch.Tensor = torch.zeros((n_layers, hidden_dim))
    key_activation: torch.Tensor = torch.zeros((n_layers, hidden_dim))
    
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_avg_activation = torch.zeros((n_layers, hidden_dim))
        batch_key_activation = torch.zeros((n_layers, hidden_dim))
        
        with model.trace(batch) as tracer:
            for layer in range(n_layers):
                batch_avg_activation[layer] = model.model.layers[layer].output[0][:,start:].mean(dim=(0,1)).save()
                batch_key_activation[layer] = model.model.layers[layer].output[0][:,-tail_tokens:].mean(dim=(0,1)).save()
        
        # Accumulate batch results with appropriate weighting
        avg_activation += batch_avg_activation * (len(batch) / len(prompts))
        key_activation += batch_key_activation * (len(batch) / len(prompts))
                
    return avg_activation, key_activation

def get_avg_and_last_token_activations_gpt2(
    model: LanguageModel,
    prompts: Union[str, List[str]],
    tail_tokens: int = 2,
    hidden_dim: int = 1280,
    n_layers: int = 35,
    start: int = 0,
    batch_size: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate average and last token activations for GPT-2 model layers.

    Args:
        model (LanguageModel): The GPT-2 language model instance.
        prompts (Union[str, List[str]]): Input text prompt(s) to process.
        tail_tokens (int, optional): Number of tokens from the end to consider. Defaults to 2.
        hidden_dim (int, optional): Hidden dimension size of the model. Defaults to 1280.
        n_layers (int, optional): Number of layers in the model. Defaults to 35.
        start (int, optional): Starting position for averaging activations. Defaults to 0.
        batch_size (int, optional): Number of prompts to process in each batch. Defaults to 8.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - avg_activation: Average activations across all layers (shape: n_layers x hidden_dim)
            - key_activation: Average activations of tail tokens across all layers (shape: n_layers x hidden_dim)
    """
    # Convert single string prompt to list for consistent handling
    if isinstance(prompts, str):
        prompts = [prompts]
    
    avg_activation: torch.Tensor = torch.zeros((n_layers, hidden_dim))
    key_activation: torch.Tensor = torch.zeros((n_layers, hidden_dim))
    
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_avg_activation = torch.zeros((n_layers, hidden_dim))
        batch_key_activation = torch.zeros((n_layers, hidden_dim))
        
        with model.trace(batch) as tracer:
            for layer in range(n_layers):
                batch_avg_activation[layer] = model.transformer.h[layer].output[0][:,start:].mean(dim=(0,1)).save()
                batch_key_activation[layer] = model.transformer.h[layer].output[0][:,-tail_tokens:].mean(dim=(0,1)).save()
        
        # Accumulate batch results with appropriate weighting
        avg_activation += batch_avg_activation * (len(batch) / len(prompts))
        key_activation += batch_key_activation * (len(batch) / len(prompts))
    
    return avg_activation, key_activation

def get_positive_negative_steering_vectors_gpt2(
    model: LanguageModel,
    a_prompts: Union[str, List[str]],
    b_prompts: Union[str, List[str]],
    tail_tokens: int = 2,
    n_layers: int = 35,
    hidden_dim: int = 1280,
    batch_size: int = 8
) -> torch.Tensor:
    """Calculate steering vectors for GPT-2 by comparing activations between positive and negative prompts.

    This function computes the difference between activation patterns of two sets of prompts,
    typically used to find directional vectors in the model's activation space that guide
    generation towards desired outputs. Supports processing prompts in batches.

    Args:
        model (LanguageModel): The GPT-2 language model instance.
        a_prompts (Union[str, List[str]]): First set of prompts (typically positive examples).
        b_prompts (Union[str, List[str]]): Second set of prompts (typically negative examples).
        tail_tokens (int, optional): Number of tokens from the end to consider. Defaults to 2.
        n_layers (int, optional): Number of layers in the model. Defaults to 35.
        hidden_dim (int, optional): Hidden dimension size of the model. Defaults to 1280.
        batch_size (int, optional): Number of prompts to process in each batch. Defaults to 8.

    Returns:
        torch.Tensor: The steering vector computed as the difference between
            positive and negative prompt activations (shape: n_layers x hidden_dim).
    """
    # Convert single string prompts to lists for consistent handling
    if isinstance(a_prompts, str):
        a_prompts = [a_prompts]
    if isinstance(b_prompts, str):
        b_prompts = [b_prompts]
    
    
    good_prompt_vectors: torch.Tensor = torch.zeros((n_layers, hidden_dim))
    bad_prompt_vectors: torch.Tensor = torch.zeros((n_layers, hidden_dim))
    
    # Process a_prompts in batches
    for i in range(0, len(a_prompts), batch_size):
        batch_a = a_prompts[i:i+batch_size]
        batch_vectors = torch.zeros((n_layers, hidden_dim))
        
        with model.trace(batch_a) as tracer:
            for layer in range(n_layers):
                batch_vectors[layer] = model.transformer.h[layer].output[0][:,-tail_tokens:].mean(dim=(0,1)).save()
        
        # Accumulate batch results
        good_prompt_vectors += batch_vectors * (len(batch_a) / len(a_prompts))
    
    # Process b_prompts in batches
    for i in range(0, len(b_prompts), batch_size):
        batch_b = b_prompts[i:i+batch_size]
        batch_vectors = torch.zeros((n_layers, hidden_dim))
        
        with model.trace(batch_b) as tracer:
            for layer in range(n_layers):
                batch_vectors[layer] = model.transformer.h[layer].output[0][:,-tail_tokens:].mean(dim=(0,1)).save()
        
        # Accumulate batch results
        bad_prompt_vectors += batch_vectors * (len(batch_b) / len(b_prompts))
    
    return good_prompt_vectors - bad_prompt_vectors

def get_positive_negative_steering_vectors_gemma(
    model: LanguageModel,
    a_prompts: Union[str, List[str]],
    b_prompts: Union[str, List[str]],
    tail_tokens: int = 2,
    n_layers: int = None,
    hidden_dim: int = None,
    batch_size: int = 8
) -> torch.Tensor:
    """Calculate steering vectors for Gemma by comparing activations between positive and negative prompts.

    This function computes the difference between activation patterns of two sets of prompts,
    typically used to find directional vectors in the model's activation space that guide
    generation towards desired outputs. Supports processing prompts in batches.

    Args:
        model (LanguageModel): The Gemma language model instance.
        a_prompts (Union[str, List[str]]): First set of prompts (typically positive examples).
        b_prompts (Union[str, List[str]]): Second set of prompts (typically negative examples).
        tail_tokens (int, optional): Number of tokens from the end to consider. Defaults to 2.
        n_layers (int, optional): Number of layers in the model. Defaults to 25.
        hidden_dim (int, optional): Hidden dimension size of the model. Defaults to 2304.
        batch_size (int, optional): Number of prompts to process in each batch. Defaults to 8.

    Returns:
        torch.Tensor: The steering vector computed as the difference between
            positive and negative prompt activations (shape: n_layers x hidden_dim).
    """
    # Convert single string prompts to lists for consistent handling
    if isinstance(a_prompts, str):
        a_prompts = [a_prompts]
    if isinstance(b_prompts, str):
        b_prompts = [b_prompts]
    
    
    if n_layers is None:
        n_layers = model.config.num_hidden_layers
    if hidden_dim is None:
        hidden_dim = model.config.hidden_size
    good_prompt_vectors: torch.Tensor = torch.zeros((n_layers, hidden_dim))
    bad_prompt_vectors: torch.Tensor = torch.zeros((n_layers, hidden_dim))
    
    # Process a_prompts in batches
    for i in range(0, len(a_prompts), batch_size):
        batch_a = a_prompts[i:i+batch_size]
        batch_vectors = torch.zeros((n_layers, hidden_dim))
        
        with model.trace(batch_a) as tracer:
            for layer in range(n_layers):
                batch_vectors[layer] = model.model.layers[layer].output[0][:,-tail_tokens:].mean(dim=(0,1)).save()
        
        # Accumulate batch results
        good_prompt_vectors += batch_vectors * (len(batch_a) / len(a_prompts))
    
    # Process b_prompts in batches
    for i in range(0, len(b_prompts), batch_size):
        batch_b = b_prompts[i:i+batch_size]
        batch_vectors = torch.zeros((n_layers, hidden_dim))
        
        with model.trace(batch_b) as tracer:
            for layer in range(n_layers):
                batch_vectors[layer] = model.model.layers[layer].output[0][:,-tail_tokens:].mean(dim=(0,1)).save()
        
        # Accumulate batch results
        bad_prompt_vectors += batch_vectors * (len(batch_b) / len(b_prompts))
    
    return good_prompt_vectors - bad_prompt_vectors