from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
import torch
from typing import List
from transformers import AutoTokenizer


def generate_with_steering_gpt2(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    steering_directions: torch.Tensor,
    layers: List[int],
    max_tokens: int = 20,
    alpha: float = 0.5,
    initial_prompt: str = "I went to see the movie, and I think it is",
    mode: str = "last"
) -> str:
    """Generate text using GPT-2 model with steering vectors applied during generation.

    Args:
        model (LanguageModel): The GPT-2 language model instance
        tokenizer (AutoTokenizer): The tokenizer for the model
        steering_directions (torch.Tensor): Steering vectors to apply during generation
        layers (List[int]): List of layer indices where to apply steering
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 20.
        alpha (float, optional): Scaling factor for steering vectors. Defaults to 0.5.
        initial_prompt (str, optional): Starting prompt for generation. Defaults to example prompt.
        mode (str, optional): Where to apply steering vectors: "last" (default, last token only),
                            "first" (first token only), or "all" (all tokens).

    Returns:
        str: Generated text with steering applied
    """
    mode = mode.lower().strip()
    if mode not in ["last", "first", "all"]:
        raise ValueError("mode must be one of: 'last', 'first', 'all'")
        
    with model.generate(initial_prompt, max_new_tokens=max_tokens, pad_token_id=model.tokenizer.eos_token_id) as tracer:
        for i, l in enumerate(layers):
            model.transformer.h[l].all()
            if mode == "last":
                model.transformer.h[l].input[0, -1, :] += steering_directions[i] * alpha
            elif mode == "first":
                model.transformer.h[l].input[0, 0, :] += steering_directions[i] * alpha
            else:  # mode == "all"
                model.transformer.h[l].input[0, :, :] += steering_directions[i] * alpha
        out = model.generator.output.save()
    output = tokenizer.decode(out[0].cpu())
    return output

def generate_with_steering_gemma(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    steering_directions: torch.Tensor,
    layers: List[int],
    max_tokens: int = 20,
    alpha: float = 0.5,
    initial_prompt: str = "I went to see the movie, and I think it is",
    mode: str = "last"
) -> str:
    """Generate text using Gemma model with steering vectors applied during generation.

    Args:
        model (LanguageModel): The Gemma language model instance
        tokenizer (AutoTokenizer): The tokenizer for the model
        steering_directions (torch.Tensor): Steering vectors to apply during generation
        layers (List[int]): List of layer indices where to apply steering
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 20.
        alpha (float, optional): Scaling factor for steering vectors. Defaults to 0.5.
        initial_prompt (str, optional): Starting prompt for generation. Defaults to example prompt.
        mode (str, optional): Where to apply steering vectors: "last" (default, last token only),
                            "first" (first token only), or "all" (all tokens).

    Returns:
        str: Generated text with steering applied
    """
    mode = mode.lower().strip()
    if mode not in ["last", "first", "all"]:
        raise ValueError("mode must be one of: 'last', 'first', 'all'")
        
    with model.generate(initial_prompt, max_new_tokens=max_tokens, pad_token_id=model.tokenizer.eos_token_id) as tracer:
        for i, l in enumerate(layers):
            model.model.layers[l].all()
            if mode == "last":
                model.model.layers[l].input[0, -1, :] += steering_directions[i] * alpha
            elif mode == "first":
                model.model.layers[l].input[0, 0, :] += steering_directions[i] * alpha
            else:  # mode == "all"
                model.model.layers[l].input[0, :, :] += steering_directions[i] * alpha
        out = model.generator.output.save()
    output = tokenizer.decode(out[0].cpu())
    return output
