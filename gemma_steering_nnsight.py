from nnsight import LanguageModel
import torch
import torch.nn.functional as F
import random
from typing import Dict, List, Union

from find_steering_vectors.find_steering_vector_nnsight import (
    get_avg_and_last_token_activations_gemma,
    get_positive_negative_steering_vectors_gemma
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
    """Setup and return the Gemma model using nnsight."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = "google/gemma-2-2b-it"
    
    print(f"Loading model {model_name} on {device}...")
    model = LanguageModel(model_name, device_map=device)
    return model

def generate_concept_prompts(concept: str, n_samples: int = 10) -> List[str]:
    """Generate multiple prompts for a given concept."""
    prompts = []
    for _ in range(n_samples):
        prompt = generate_concept_prompt(concept, None)
        prompts.append(prompt)
    return prompts

def calculate_concept_steering_vectors_positive_negative(
    model: LanguageModel,
    concept: str,
    n_samples: int = 10,
    batch_size: int = 4,
    tail_tokens: int = 2
) -> torch.Tensor:
    """Calculate steering vectors for a specific concept using positive and negative examples."""
    positive_prompts = generate_concept_prompts(concept, n_samples)
    
    other_concepts = [c for c in concept_prompts_dict.keys() if c != concept]
    negative_concept = random.choice(other_concepts)
    negative_prompts = generate_concept_prompts(negative_concept, n_samples)
    
    steering_vector = get_positive_negative_steering_vectors_gemma(
        model=model,
        a_prompts=positive_prompts,
        b_prompts=negative_prompts,
        tail_tokens=tail_tokens,
    )
    
    return steering_vector

def calculate_concept_steering_vectors_last_token(
    model: LanguageModel,
    concept: str,
    n_samples: int = 10,
    batch_size: int = 4,
    tail_tokens: int = 2
) -> torch.Tensor:
    """Calculate steering vectors for a specific concept using last token activations.
    
    This function generates concept prompts and extracts the activations from the last tokens,
    which are expected to be most relevant to the concept being learned.
    
    Args:
        model (LanguageModel): The transformer model to analyze
        concept (str): The concept to generate steering vectors for
        n_samples (int, optional): Number of prompts to generate. Defaults to 10.
        batch_size (int, optional): Batch size for processing. Defaults to 4.
        tail_tokens (int, optional): Number of tokens from the end to consider. Defaults to 2.
        
    Returns:
        torch.Tensor: The steering vector computed from the last token activations
    """
    prompts = generate_concept_prompts(concept, n_samples)
    
    # Get both average and last token activations
    avg_activation, last_token_activation = get_avg_and_last_token_activations_gemma(
        model=model,
        prompts=prompts,
        tail_tokens=tail_tokens,
    )
    
    # Use the last token activation as our steering vector
    return last_token_activation

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
            f"steering_vectors/nnsight_multiple_choice/gemma_pn_{concept}_steering_vector.pt"
        )
        print(f"Saved normalized positive-negative steering vector for {concept}")
    
    print("\nDone! Calculated positive-negative steering vectors for all concepts.")
    
    last_token_vectors = {}
    print("\nCalculating last-token steering vectors...")
    
    for concept in concept_prompts_dict.keys():
        print(f"\nCalculating last-token steering vector for concept: {concept}")
        steering_vector = calculate_concept_steering_vectors_last_token(model, concept)
        print(f"Steering vector shape: {steering_vector.shape}")
        last_token_vectors[concept] = steering_vector
    
    all_vectors = torch.stack(list(last_token_vectors.values()))
    avg_vector = torch.mean(all_vectors, dim=0)

    for concept in concept_prompts_dict.keys():
        print(f"\nCalculating centered last-token steering vector for concept: {concept}")
        centered_vector = last_token_vectors[concept] - avg_vector
        print(f"Centered vector shape: {centered_vector.shape}")
        torch.save(
            centered_vector,
            f"steering_vectors/nnsight_multiple_choice/gemma_lt_{concept}_steering_vector.pt"
        )
        print(f"Saved normalized and centered last-token steering vector for {concept}")
    
    print("\nDone! Calculated and centered last-token steering vectors for all concepts.")

    for concept in explicit_concept_prompts_dict.keys():
        positive_prompts = explicit_concept_prompts_dict[concept]
        negative_prompts = explicit_anti_concept_prompts_dict[concept]
        
        steering_vector = get_positive_negative_steering_vectors_gemma(
            model=model,
            a_prompts=positive_prompts,
            b_prompts=negative_prompts,
            tail_tokens=2
        )
        print(f"Steering vector shape: {steering_vector.shape}")
        print(f"Saved normalized positive-negative steering vector for {concept}")
        torch.save(
            steering_vector,
            f"steering_vectors/nnsight_explicit_concept/gemma_pn_{concept}_steering_vector.pt"
        )

if __name__ == "__main__":
    main()
