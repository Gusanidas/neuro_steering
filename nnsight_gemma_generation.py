from nnsight import LanguageModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List, Dict, Union


def setup_model(device=None):
    """Setup and return the Gemma model using nnsight."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = "google/gemma-2-2b-it"
    print(f"Loading model {model_name} on {device}...")
    model = LanguageModel(model_name, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_with_steering(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    steering_directions: List[torch.Tensor],
    layers: List[int],
    max_tokens: int = 60,
    alpha: float = 1.0,
    initial_prompt: str = "I went to see the movie, and I think it is",
    mode: str = "last"
) -> str:
    """Generate text using Gemma model with steering vectors applied during generation.

    Args:
        model (LanguageModel): The Gemma language model instance
        tokenizer (AutoTokenizer): The tokenizer for the model
        steering_directions (List[torch.Tensor]): List of steering vectors to apply during generation
        layers (List[int]): List of layer indices where to apply steering
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 60.
        alpha (float, optional): Scaling factor for steering vectors. Defaults to 1.0.
        initial_prompt (str, optional): Starting prompt for generation. Defaults to example prompt.
        mode (str, optional): Mode for applying steering. "last" applies to last token only. Defaults to "last".

    Returns:
        str: Generated text with steering applied
    """
    with model.generate(initial_prompt, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id) as tracer:
        for i, l in enumerate(layers):
            model.model.layers[l].all()
            if mode == "last":
                # Apply steering to the last token only
                model.model.layers[l].input[0, -1, :] += steering_directions[i] * alpha
            else:
                # Apply steering to all tokens
                model.model.layers[l].input[0, :, :] += steering_directions[i] * alpha
        out = model.generator.output.save()
    
    output = tokenizer.decode(out[0].cpu())
    return output


def generate_with_multiple_alphas(
    model: LanguageModel, 
    tokenizer: AutoTokenizer, 
    concept: str, 
    device: str,
    layers: List[int] = [21, 22, 23, 24],  # Default to last 4 layers
    alphas: List[float] = [1, 2, 4, 8, 16, 32, 64, 128],
    steering_vector: torch.Tensor = None,
    mode: str = "last",
    prompt: str = "I went to see the movie, and I think it is"
) -> Dict[float, str]:
    """Generate text with different steering strengths (alpha values) and specified layers.
    
    Args:
        model: The LanguageModel model
        tokenizer: The model's tokenizer
        concept: String name of the concept to steer towards
        device: Device to use (cuda, cpu, mps)
        layers: List of layer indices to apply steering
        alphas: List of steering strengths to try
        steering_vector: Pre-computed steering vector to use
        mode: Mode for applying steering ("last" or "all")
    
    Returns:
        Dict[float, str]: Dictionary mapping alpha values to generated text
    """
    # Create a list of steering vectors, one for each layer
    steering_vectors = [
        steering_vector[layer].to(device) for layer in layers
    ]
    

    
    # Generate text for each alpha value
    results = {}
    for alpha in alphas:
        generated_text = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            steering_directions=steering_vectors,
            layers=layers,
            max_tokens=60,
            alpha=alpha,
            initial_prompt=prompt,
            mode=mode,
        )
        results[alpha] = generated_text
        print(f"\nAlpha = {alpha}:")
        print("Generated text:", generated_text)
    
    return results


def main():
    # Set up device and model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = setup_model(device)
    concept = "cat"
    
    # Example: Using custom layers and alphas
    custom_layers = [21]
    custom_alphas = [0.5, 1, 2, 3, 5, 8, 12, 20, 30]

    messages = [
        {
            "role": "user",
            "content": """Which animal is the most annoying to have around?
            a) cat
            b) dog
            c) mouse
            d) bird
            e) fish
            f) spider
            g) snake
            Give first letter of the answer. Then a reasoning. Any other text is not allowed.
            """
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    #steering_vector = torch.load(
    #    f"steering_vectors/nnsight_explicit_concept/gemma_pn_{concept}_steering_vector.pt"
    #)
    steering_vector = torch.load(
        f"steering_vectors/nnsight_multiple_choice/gemma_lt_{concept}_steering_vector.pt"
    )
    
    # Generate texts with custom parameters
    results = generate_with_multiple_alphas(
        model, 
        tokenizer, 
        concept, 
        device,
        layers=custom_layers,
        alphas=custom_alphas,
        steering_vector=steering_vector,
        mode="last",
        prompt=prompt
    )


if __name__ == "__main__":
    main() 