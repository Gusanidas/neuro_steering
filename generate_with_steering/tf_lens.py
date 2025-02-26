import torch
from typing import List, Union
from transformer_lens import HookedTransformer


def generate_with_steering(
    model: HookedTransformer,
    steering_directions: torch.Tensor,
    layers: List[int],
    max_tokens: int = 20,
    alpha: float = 0.5,
    initial_prompt: str = "I went to see the movie, and I think it is",
    mode: str = "last",
    normalize: bool = False
) -> str:
    """
    Generate text from a 'Gemma'-style HookedTransformer model with steering vectors
    applied to specific layers using TransformerLens hooks.

    This is logically the same as the GPT-2 version, but if your "Gemma" model
    differs in how you want to handle hooking or generation, you can tailor
    this function accordingly.

    Args:
        model (HookedTransformer): A HookedTransformer loaded with the Gemma weights.
        tokenizer (AutoTokenizer): A tokenizer to decode the generated tokens if needed.
        steering_directions (torch.Tensor): Steering vectors to apply at each chosen layer.
        layers (List[int]): Layer indices where we insert the steering vectors.
        max_tokens (int, optional): Number of tokens to generate. Defaults to 20.
        alpha (float, optional): Steering vector scale. Defaults to 0.5.
        initial_prompt (str, optional): Prompt string.
        mode (str, optional): Where to apply steering vectors: "last" (default, last token only),
                            "first" (first token only), or "all" (all tokens).
        normalize (bool, optional): Whether to normalize the activation after adding steering. Defaults to False.
    Returns:
        str: The generated text.
    """
    mode = mode.lower().strip()
    if mode not in ["last", "first", "all"]:
        raise ValueError("mode must be one of: 'last', 'first', 'all'")

    for i, layer_idx in enumerate(layers):
        direction_i = steering_directions[i]

        def steer_hook(activation, hook, direction=direction_i):
            """
            activation has shape [batch, seq_len, hidden_dim].
            We add 'direction' scaled by alpha based on the specified mode.
            """
            if mode == "last":
                original_norm = torch.norm(activation[:, -1, :], dim=-1, keepdim=True)
                activation[:, -1, :] += direction * alpha
                if normalize:
                    activation[:, -1, :] = activation[:, -1, :] * (original_norm / torch.norm(activation[:, -1, :], dim=-1, keepdim=True))
            elif mode == "first":
                original_norm = torch.norm(activation[:, 0, :], dim=-1, keepdim=True)
                activation[:, 0, :] += direction * alpha
                if normalize:
                    activation[:, 0, :] = activation[:, 0, :] * (original_norm / torch.norm(activation[:, 0, :], dim=-1, keepdim=True))
            else:  # mode == "all"
                original_norm = torch.norm(activation, dim=-1, keepdim=True)
                activation += direction * alpha
                if normalize:
                    activation = activation * (original_norm / torch.norm(activation, dim=-1, keepdim=True))
            return activation

        # Adjust the hook name to match your Gemma architecture if it differs
        model.add_hook(f"blocks.{layer_idx}.hook_resid_post", steer_hook)

    generated_text = model.generate(
        initial_prompt,
        max_new_tokens=max_tokens,
    )

    model.reset_hooks()

    return generated_text