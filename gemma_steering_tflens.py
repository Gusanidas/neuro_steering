from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
import torch.nn.functional as F
import torch
from typing import Dict, List
import random

from find_steering_vectors.find_steering_vector_tlens import (
    get_avg_and_last_token_activations,
    get_positive_negative_steering_vectors
)
from prompts.hexby.multiple_choice_prompts import (
    generate_concept_prompt,
    concept_prompts_dict
)
from prompts.hexby.explicit_concept_prompts import (
    concept_prompts_dict as explicit_concept_prompts_dict,
    anti_concept_prompts_dict as explicit_anti_concept_prompts_dict
)

def setup_model(device=None):
    """Setup and return the Gemma model and its configuration."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = "google/gemma-2-2b-it"
    
    print(f"Loading model {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device
    )
    return model

def generate_concept_prompts(concept: str, n_samples: int = 10) -> List[str]:
    """Generate multiple prompts for a given concept."""
    prompts = []
    for _ in range(n_samples):
        prompt = generate_concept_prompt(concept, None)
        prompts.append(prompt)
    return prompts

def calculate_concept_steering_vectors_positive_negative(
    model: HookedTransformer,
    concept: str,
    n_samples: int = 10,
    batch_size: int = 4
) -> torch.Tensor:
    """Calculate steering vectors for a specific concept using positive and negative examples."""
    positive_prompts = generate_concept_prompts(concept, n_samples)
    
    other_concepts = [c for c in concept_prompts_dict.keys() if c != concept]
    negative_concept = random.choice(other_concepts)
    negative_prompts = generate_concept_prompts(negative_concept, n_samples)
    
    steering_vector = get_positive_negative_steering_vectors(
        model=model,
        a_prompts=positive_prompts,
        b_prompts=negative_prompts,
        batch_size=batch_size,
        start = -2,
    )
    
    return steering_vector

def calculate_concept_steering_vectors_last_token(
    model: HookedTransformer,
    concept: str,
    n_samples: int = 10,
    batch_size: int = 4,
    tail_tokens: int = 2,
    tail_start: int = None
) -> torch.Tensor:
    """Calculate steering vectors for a specific concept using last token activations.
    
    This function generates concept prompts and extracts the activations from the last tokens,
    which are expected to be most relevant to the concept being learned.
    
    Args:
        model (HookedTransformer): The transformer model to analyze
        concept (str): The concept to generate steering vectors for
        n_samples (int, optional): Number of prompts to generate. Defaults to 10.
        batch_size (int, optional): Batch size for processing. Defaults to 4.
        tail_tokens (int, optional): Number of tokens from the end to consider. Defaults to 2.
        tail_start (int, optional): Starting position for tail token activations. If None, uses the last 'tail_tokens' tokens. Defaults to None.
        
    Returns:
        torch.Tensor: The steering vector computed from the last token activations
    """
    prompts = generate_concept_prompts(concept, n_samples)
    
    # Get both average and last token activations
    avg_activation, last_token_activation = get_avg_and_last_token_activations(
        model=model,
        prompts=prompts,
        tail_tokens=tail_tokens,
        batch_size=batch_size,
        start=-2,  # Consider only the last few tokens for average activations
        tail_start=tail_start  # Use the new parameter for tail token activations
    )
    
    # Use the last token activation as our steering vector
    return last_token_activation

def main():
    device = "mps" # For M1 Macs, use another device if not available
    model = setup_model(device)
    
    # Calculate positive-negative steering vectors
    concept_vectors = {}
    for concept in concept_prompts_dict.keys():
        print(f"\nCalculating positive-negative steering vector for concept: {concept}")
        steering_vector = calculate_concept_steering_vectors_positive_negative(model, concept)
        # Normalize the steering vector
        #steering_vector = F.normalize(steering_vector, p=2, dim=1)
        print(f"Steering vector shape: {steering_vector.shape}")
        concept_vectors[concept] = steering_vector
        
        torch.save(
            steering_vector,
            f"steering_vectors/multiple_choice/gemma_pn_{concept}_steering_vector.pt"
        )
        print(f"Saved normalized positive-negative steering vector for {concept}")
    
    print("\nDone! Calculated positive-negative steering vectors for all concepts.")
    
    # Calculate last-token steering vectors
    last_token_vectors = {}
    print("\nCalculating last-token steering vectors...")
    
    # First pass: calculate all vectors
    for concept in concept_prompts_dict.keys():
        print(f"\nCalculating last-token steering vector for concept: {concept}")
        steering_vector = calculate_concept_steering_vectors_last_token(
            model=model, 
            concept=concept,
            tail_start=None  # Use default behavior (last tail_tokens tokens)
        )
        print(f"Steering vector shape: {steering_vector.shape}")
        # Normalize before storing
        #steering_vector = F.normalize(steering_vector, p=2, dim=1)
        last_token_vectors[concept] = steering_vector
    
    # Calculate average vector across all concepts
    all_vectors = torch.stack(list(last_token_vectors.values()))
    avg_vector = torch.mean(all_vectors, dim=0)
    # Normalize the average vector
    #avg_vector = F.normalize(avg_vector, p=2, dim=0)
    
    # Second pass: subtract average and save
    for concept in concept_prompts_dict.keys():
        # Subtract average to get concept-specific activation
        centered_vector = last_token_vectors[concept] - avg_vector
        # Normalize the final centered vector
        #centered_vector = F.normalize(centered_vector, p=2, dim=1)
        print(f"Centered vector shape: {centered_vector.shape}")
        torch.save(
            centered_vector,
            f"steering_vectors/multiple_choice/gemma_lt_{concept}_steering_vector.pt"
        )
        print(f"Saved normalized and centered last-token steering vector for {concept}")
    
    print("\nDone! Calculated and centered last-token steering vectors for all concepts.")

    for concept in explicit_concept_prompts_dict.keys():
        positive_prompts = explicit_concept_prompts_dict[concept]
        negative_prompts = explicit_anti_concept_prompts_dict[concept]
        
        steering_vector = get_positive_negative_steering_vectors(
            model=model,
            a_prompts=positive_prompts,
            b_prompts=negative_prompts,
            batch_size=4,
            start=10,
        )
        print(f"Steering vector shape: {steering_vector.shape}")
        #steering_vector = F.normalize(steering_vector, p=2, dim=1)
        print(f"Saved normalized positive-negative steering vector for {concept}")
        torch.save(
            steering_vector,
            f"steering_vectors/explicit_concept/gemma_pn_{concept}_steering_vector.pt"
        )

if __name__ == "__main__":
    main()