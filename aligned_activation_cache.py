import torch
import os
import csv
from typing import List, Tuple, Dict, Iterator, Union, Callable, Optional
from aligned_iterator import AlignedIterator
from aligned_tokenized_story import AlignedTokenizedStory
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens import ActivationCache

class AlignedActivationsCache(AlignedIterator):
    """
    Stores activations from a transformer model, aligned with time intervals
    derived from the original tokenization of a story.

    Inherits from AlignedIterator, where `data` holds the activation tensors
    and `element_to_time_map` maps activation indices (corresponding to original
    token positions with timing info) to time intervals.
    """
    def __init__(
        self,
        activations: torch.Tensor,
        activation_to_time_map: Dict[int, Tuple[float, float]],
        story_name: Optional[str] = None,
        activation_name: Optional[str] = None
    ):
        """
        Initialize an AlignedActivationsCache object.

        Args:
            activations (torch.Tensor): A tensor of activations, typically shape
                                        [num_aligned_elements, activation_dim].
                                        num_aligned_elements should match the number of
                                        entries in activation_to_time_map.
            activation_to_time_map (Dict[int, Tuple[float, float]]):
                                        A dictionary mapping activation indices (from 0 to
                                        num_aligned_elements-1) to time intervals (start_time, end_time).
            story_name (Optional[str]): Name of the source story.
            activation_name (Optional[str]): Name or description of the extracted activation
                                            (e.g., 'resid_post_layer_5').
        """
        super().__init__(data=activations, element_to_time_map=activation_to_time_map)

        # Store additional metadata
        self.story_name = story_name
        self.activation_name = activation_name

    def get_activation_for_seconds(self, seconds: float) -> Optional[int]:
        return self.get_element_for_time_seconds(seconds)

    def get_seconds_for_activation(self, activation_idx: int) -> Tuple[float, float]:
        return self.get_time_seconds_for_element(activation_idx)

    def get_activation_vector(self, activation_idx: int) -> torch.Tensor:
        if not 0 <= activation_idx < len(self.data):
             raise IndexError(f"Activation index {activation_idx} out of bounds for data length {len(self.data)}")
        return self.data[activation_idx]

    @classmethod
    def build_from_tokenized_story(cls, story: AlignedTokenizedStory, model: HookedTransformer, activation_extractor: Callable[[ActivationCache], torch.Tensor], activation_name: str, device: Optional[str] = None) -> 'AlignedActivationsCache':
        return build_aligned_activation_cache(story=story, model=model, activation_extractor=activation_extractor, activation_name=activation_name, device=device)


def build_aligned_activation_cache(
    story: AlignedTokenizedStory,
    model: HookedTransformer,
    activation_extractor: Callable[[ActivationCache], torch.Tensor],
    activation_name: Optional[str] = None,
    device: Optional[str] = None
) -> AlignedActivationsCache:
    """
    Creates an AlignedActivationsCache by running a model on a story's tokens,
    extracting specific activations, and aligning them with the story's timing info.

    Args:
        story (AlignedTokenizedStory): The tokenized story with time alignments.
        model (HookedTransformer): A TransformerLens model.
        activation_extractor (Callable[[ActivationCache], torch.Tensor]):
            A function that takes the model's ActivationCache dictionary and returns
            the desired activation tensor. The returned tensor should typically have shape
            [batch_size, sequence_length, activation_dim], where sequence_length
            matches the length of the input tokens sequence given to the model.
        activation_name (Optional[str]): A name for the extracted activation (for metadata).
        batch_size (int): Batch size for model inference (currently supports 1).
        device (Optional[str]): Device to run the model on ('cuda', 'cpu', etc.).

    Returns:
        AlignedActivationsCache: An object containing the extracted activations aligned with time.

    Raises:
        ValueError: If batch_size is not 1 (current limitation) or if shapes mismatch.
        KeyError: If token indices from the story's time map are out of bounds for the model's output activations.
    """

    model_input_tokens = story.data.unsqueeze(0)
    if device:
        model.to(device)
        model_input_tokens = model_input_tokens.to(device)

    with torch.no_grad(): 
        logits, cache = model.run_with_cache(model_input_tokens)

    # Expected shape: [1, seq_len, activation_dim]
    extracted_activations = activation_extractor(cache)

    if extracted_activations.ndim != 3 or extracted_activations.shape[0] != 1:
        raise ValueError(f"Activation extractor returned tensor with unexpected shape: "
                         f"{extracted_activations.shape}. Expected [1, seq_len, activation_dim].")

    activations_single = extracted_activations[0].cpu() 
    model_seq_len = activations_single.shape[0]

    original_timed_indices = sorted(story.element_to_time_map.keys())

    # Filter out indices that might be out of bounds for the model's output sequence length
    valid_timed_indices = [idx for idx in original_timed_indices if 0 <= idx < model_seq_len]
    if len(valid_timed_indices) != len(original_timed_indices):
        print(f"Warning: {len(original_timed_indices) - len(valid_timed_indices)} timed token indices "
              f"were out of bounds for the model output sequence length ({model_seq_len}).")
        if not valid_timed_indices:
             raise ValueError("No valid timed token indices found within model output sequence length.")


    # Shape: [num_valid_timed_tokens, activation_dim]
    try:
        relevant_activations = activations_single[valid_timed_indices]
    except IndexError as e:
         raise IndexError(f"Error indexing activations tensor (shape {activations_single.shape}) with indices "
                          f"{valid_timed_indices}. Original error: {e}")


    new_activation_to_time_map: Dict[int, Tuple[float, float]] = {
        new_idx: story.element_to_time_map[original_idx]
        for new_idx, original_idx in enumerate(valid_timed_indices)
    }

    # Instantiate and return the AlignedActivationsCache
    return AlignedActivationsCache(
        activations=relevant_activations,
        activation_to_time_map=new_activation_to_time_map,
        story_name=story.story_name,
        activation_name=activation_name
    )



if __name__ == '__main__':
    from transformer_lens import HookedTransformer
    from aligned_tokenized_story import AlignedTokenizedStory
    from transformers import GPT2Tokenizer
    import torch

    model = HookedTransformer.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    test_text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer.encode(test_text, return_tensors="pt")[0]
    
    token_to_time_map = {
        i: (i * 0.1, (i + 1) * 0.1) for i in range(len(tokens))
    }
    story = AlignedTokenizedStory(
        story_name="test_story",
        tokenizer=tokenizer,
        tokens=tokens,
        token_to_time_map=token_to_time_map
    )

    def extract_layers_4_5_6(cache):
        layer_4 = cache["blocks.4.hook_resid_post"]
        layer_5 = cache["blocks.5.hook_resid_post"]
        layer_6 = cache["blocks.6.hook_resid_post"]
        return torch.cat([layer_4, layer_5, layer_6], dim=-1)

    activation_cache = build_aligned_activation_cache(
        story=story,
        model=model,
        activation_extractor=extract_layers_4_5_6,
        activation_name="layers_4_5_6_resid_post"
    )

    print(f"Number of activations: {len(activation_cache.data)}")
    print(f"Activation shape: {activation_cache.data.shape}")
    
    test_time = 0.15
    activation_idx = activation_cache.get_activation_for_seconds(test_time)
    if activation_idx is not None:
        print(f"\nActivation at time {test_time}s:")
        print(f"Activation index: {activation_idx}")
        print(f"Activation vector shape: {activation_cache.get_activation_vector(activation_idx).shape}")
        print(f"Time interval: {activation_cache.get_seconds_for_activation(activation_idx)}")
