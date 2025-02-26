from transformer_lens import HookedTransformer
import torch
from typing import Tuple, List, Union
import gc

def get_avg_and_last_token_activations(
    model: HookedTransformer,
    prompts: Union[str, List[str]],
    tail_tokens: int = 2,
    batch_size: int = 32,
    start: int = 0,
    tail_start: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate average and last token activations for a TransformerLens model.

    Args:
        model (HookedTransformer): The TransformerLens model instance.
        prompts (Union[str, List[str]]): Input text prompt(s) to process.
        tail_tokens (int, optional): Number of tokens from the end to consider. Defaults to 2.
        batch_size (int, optional): Batch size for processing. Defaults to 32.
        start (int, optional): Starting position for averaging activations. Defaults to 0.
        tail_start (int, optional): Starting position for tail token activations. If None, uses the last 'tail_tokens' tokens. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - avg_activation: Average activations across all layers (shape: n_layers x hidden_dim)
            - key_activation: Average activations of tail tokens across all layers (shape: n_layers x hidden_dim)
    """
    n_layers = model.cfg.n_layers
    hidden_dim = model.cfg.d_model
    device = model.cfg.device

    if isinstance(prompts, str):
        prompts = [prompts]

    # Pre-allocate on CPU to minimize GPU memory
    total_avg_activation = torch.zeros((n_layers, hidden_dim), device="cpu")
    total_key_activation = torch.zeros((n_layers, hidden_dim), device="cpu")
    total_examples = 0
    total_tail_tokens = 0

    def make_avg_hook(layer_idx):
        def hook_fn(activation, hook):
            # Only consider tokens from start onwards
            total_avg_activation[layer_idx] += activation[:, start:, :].detach().cpu().sum(dim=(0, 1))
        return hook_fn

    def make_key_hook(layer_idx):
        def hook_fn(activation, hook):
            nonlocal total_tail_tokens
            if tail_start is None:
                # Use the last tail_tokens tokens
                total_key_activation[layer_idx] += activation[:, -tail_tokens:, :].detach().cpu().sum(dim=(0, 1))
                total_tail_tokens = len(prompts) * tail_tokens
            else:
                # Use tokens from tail_start onwards
                total_key_activation[layer_idx] += activation[:, tail_start:, :].detach().cpu().sum(dim=(0, 1))
                # Calculate total tokens processed for tail activations
                batch_size_actual = activation.shape[0]
                seq_len = activation.shape[1]
                total_tail_tokens += batch_size_actual * max(0, seq_len - tail_start)
        return hook_fn

    avg_hooks = [
        (f"blocks.{layer_idx}.hook_resid_post", make_avg_hook(layer_idx))
        for layer_idx in range(n_layers)
    ]
    key_hooks = [
        (f"blocks.{layer_idx}.hook_resid_post", make_key_hook(layer_idx))
        for layer_idx in range(n_layers)
    ]
    all_hooks = avg_hooks + key_hooks

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_tokens = model.to_tokens(batch_prompts)
        batch_size_actual = batch_tokens.shape[0]
        seq_len = batch_tokens.shape[1]
        total_examples += batch_size_actual * (seq_len - start)  # Only count tokens from start onwards

        model.run_with_hooks(
            batch_tokens,
            fwd_hooks=all_hooks
        )

        model.reset_hooks()
        torch.cuda.empty_cache()
        gc.collect()

    avg_activation = total_avg_activation / total_examples
    
    # Use the appropriate denominator based on whether tail_start was provided
    if tail_start is None:
        key_activation = total_key_activation / (len(prompts) * tail_tokens)
    else:
        key_activation = total_key_activation / total_tail_tokens if total_tail_tokens > 0 else total_key_activation

    return avg_activation, key_activation

def get_positive_negative_steering_vectors(
    model: HookedTransformer,
    a_prompts: Union[str, List[str]],
    b_prompts: Union[str, List[str]],
    start: int = 0,
    batch_size: int = 32
) -> torch.Tensor:
    """Calculate steering vectors by comparing activations between positive and negative prompts.

    This function computes the difference between activation patterns of two sets of prompts,
    typically used to find directional vectors in the model's activation space that guide
    generation towards desired outputs.

    Args:
        model (HookedTransformer): The TransformerLens model instance.
        a_prompts (Union[str, List[str]]): First set of prompts (typically positive examples).
        b_prompts (Union[str, List[str]]): Second set of prompts (typically negative examples).
        start (int, optional): Starting position for averaging activations. Defaults to 0.
        batch_size (int, optional): Batch size for processing. Defaults to 32.

    Returns:
        torch.Tensor: The steering vector computed as the difference between
            positive and negative prompt activations (shape: n_layers x hidden_dim).
    """
    n_layers = model.cfg.n_layers
    hidden_dim = model.cfg.d_model

    if isinstance(a_prompts, str):
        a_prompts = [a_prompts]
    if isinstance(b_prompts, str):
        b_prompts = [b_prompts]

    # Pre-allocate on CPU
    good_prompt_vectors = torch.zeros((n_layers, hidden_dim), device="cpu")
    bad_prompt_vectors = torch.zeros((n_layers, hidden_dim), device="cpu")
    total_a_examples = 0
    total_b_examples = 0

    def make_hook(layer_idx, target_tensor):
        def hook_fn(activation, hook):
            target_tensor[layer_idx] += activation[:, start:, :].detach().cpu().sum(dim=(0, 1))
        return hook_fn

    # Process positive prompts
    hooks = [
        (f"blocks.{layer_idx}.hook_resid_post", make_hook(layer_idx, good_prompt_vectors))
        for layer_idx in range(n_layers)
    ]

    for i in range(0, len(a_prompts), batch_size):
        batch_prompts = a_prompts[i : i + batch_size]
        batch_tokens = model.to_tokens(batch_prompts)
        batch_size_actual = batch_tokens.shape[0]
        seq_len = batch_tokens.shape[1]
        total_a_examples += batch_size_actual * (seq_len - start)

        model.run_with_hooks(
            batch_tokens,
            fwd_hooks=hooks
        )

        model.reset_hooks()
        torch.cuda.empty_cache()
        gc.collect()

    # Process negative prompts
    hooks = [
        (f"blocks.{layer_idx}.hook_resid_post", make_hook(layer_idx, bad_prompt_vectors))
        for layer_idx in range(n_layers)
    ]

    for i in range(0, len(b_prompts), batch_size):
        batch_prompts = b_prompts[i : i + batch_size]
        batch_tokens = model.to_tokens(batch_prompts)
        batch_size_actual = batch_tokens.shape[0]
        seq_len = batch_tokens.shape[1]
        total_b_examples += batch_size_actual * (seq_len - start)

        model.run_with_hooks(
            batch_tokens,
            fwd_hooks=hooks
        )

        model.reset_hooks()
        torch.cuda.empty_cache()
        gc.collect()

    good_prompt_vectors /= total_a_examples
    bad_prompt_vectors /= total_b_examples

    return good_prompt_vectors - bad_prompt_vectors
