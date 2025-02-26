from transformer_lens import HookedTransformer
import torch
import torch.nn.functional as F
from generate_with_steering.tf_lens import generate_with_steering
from transformers import AutoTokenizer


def setup_model(device=None):
    """Setup and return the Gemma model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "google/gemma-2-2b-it"
    print(f"Loading model {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_with_multiple_alphas(
    model, 
    tokenizer, 
    concept, 
    device, 
    layers=[-4, -3, -2, -1],  # Default to last 4 layers
    alphas=[1, 2, 4, 8, 16, 32, 64, 128],
    steering_vector=None
):
    """Generate text with different steering strengths (alpha values) and specified layers.
    
    Args:
        model: The HookedTransformer model
        tokenizer: The model's tokenizer
        concept: String name of the concept to steer towards
        device: torch device to use
        layers: List of layer indices to apply steering (negative indices allowed)
        alphas: List of steering strengths to try
    """

    
    # Convert negative indices to positive
    layers = [model.cfg.n_layers + l if l < 0 else l for l in layers]
    # Create a list of steering vectors, one for each layer
    steering_vectors = [
        steering_vector[layer].to(device) for layer in range(len(layers))
    ]
    
    messages = [
        {
            "role": "user",
            "content": """Choose one of the following options:
            a) cat
            b) bottle
            c) shoe
            d) house
            e) chair
            f) scissors
            g) face
            Give first letter of the answer. Then a reasoning. Any other text is not allowed.
            """
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = "I am going to go to the park. Then I will go to the store. Then"
    
    # Generate text for each alpha value
    results = {}
    for alpha in alphas:
        generated_text = generate_with_steering(
            model=model,
            steering_directions=steering_vectors,
            layers=layers,
            max_tokens=60,
            alpha=alpha,
            initial_prompt=prompt,
            mode="last",  # Apply steering to last token only
            normalize=False,
        )
        results[alpha] = generated_text
        print(f"\nAlpha = {alpha}:")
        print("Generated text:", generated_text)
    
    return results

def main():
    # Set up device and model
    device = torch.device("mps")  # For M1 Macs, change as needed
    model, tokenizer = setup_model(device)
    model.eval()
    model.reset_hooks()
    concept = "cat"
    
    # Example: Using custom layers and alphas
    custom_layers = [21]
    custom_alphas = [0.5, 1, 2,3,5, 8, 12, 20, 30]

    steering_vector = torch.load(
        f"steering_vectors/explicit_concept/gemma_pn_{concept}_steering_vector.pt"
    )
    
    # Generate texts with custom parameters
    results = generate_with_multiple_alphas(
        model, 
        tokenizer, 
        concept, 
        device,
        layers=custom_layers,
        alphas=custom_alphas,
        steering_vector=steering_vector
    )

if __name__ == "__main__":
    main()