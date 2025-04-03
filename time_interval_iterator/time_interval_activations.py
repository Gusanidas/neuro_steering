import torch
import os
import csv
from typing import List, Tuple, Dict, Iterator, Union, Callable, Optional
from time_interval_iterator.time_interval_iterator import TimeIntervalIterator
from time_interval_iterator.time_interval_tokenized_story import TimeIntervalTokenizedStory
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens import ActivationCache

class TimeIntervalActivations(TimeIntervalIterator):
    """
    Stores activations from a transformer model, aligned with time intervals
    derived from the original tokenization of a story.

    Inherits from AlignedIterator, where `data` holds the activation tensors
    and `element_to_time_map` maps activation indices (corresponding to original
    token positions with timing info) to time intervals.
    """
    def __init__(
        self,
        activations: List,
        activation_to_time_map: Dict[int, Tuple[float, float]],
        story_name: str, 
        activation_name: str,
        activation_extractor: Callable[[ActivationCache], torch.Tensor],
        story: TimeIntervalTokenizedStory,
        model: HookedTransformer,
        device: Optional[str] = None
    ):
        """
        Initialize an TimeIntervalActivations object.

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

        self.story_name = story_name
        self.activation_name = activation_name
        self.activation_extractor = activation_extractor
        self.story = story
        self.model = model
        self.device = device
        self._initialized = False

        if activations is not None and activation_to_time_map is not None:
            super().__init__(data=activations, element_to_time_map=activation_to_time_map)
            self.initialized = True

    def _initialize(self):
        if self._initialized:
            return
        self.story.ensure_initialized()
        self._build_from_tokenized_story(story=self.story, model=self.model, activation_extractor=self.activation_extractor, activation_name=self.activation_name, device=self.device)
        self._initialized = True

    def _ensure_initialized(self):
        if not self._initialized:
            self._initialize()

    def __iter__(self):
        self._ensure_initialized()
        return super().__iter__()

    def __next__(self):
        self._ensure_initialized()
        return super().__next__()

    def get_activation_for_seconds(self, seconds: float) -> Optional[int]:
        self._ensure_initialized()
        return self.get_element_for_time_seconds(seconds)

    def get_seconds_for_activation(self, activation_idx: int) -> Tuple[float, float]:
        self._ensure_initialized()
        return self.get_time_seconds_for_element(activation_idx)

    def get_time_seconds_for_element(self, element_idx: int) -> Tuple[float, float]:
        self._ensure_initialized()
        return super().get_time_seconds_for_element(element_idx)

    def get_element_for_time_seconds(self, seconds: float) -> Optional[int]:
        self._ensure_initialized()
        return super().get_element_for_time_seconds(seconds)

    def get_activation_vector(self, activation_idx: int) -> torch.Tensor:
        self._ensure_initialized()
        if not 0 <= activation_idx < len(self.data):
             raise IndexError(f"Activation index {activation_idx} out of bounds for data length {len(self.data)}")
        return self.data[activation_idx]

    def _build_from_tokenized_story(self, story: TimeIntervalTokenizedStory, model: HookedTransformer, activation_extractor: Callable[[ActivationCache], torch.Tensor], activation_name: str, device: Optional[str] = None) -> 'TimeIntervalActivations':
        print(f"Building TimeIntervalActivations from tokenized story")

        model_input_tokens = story.data.unsqueeze(0)
        if device:
            model.to(device)
            model_input_tokens = model_input_tokens.to(device)

        with torch.no_grad(): 
            logits, cache = model.run_with_cache(model_input_tokens)

        # Expected shape: [1, seq_len, activation_dim]
        extracted_activations = activation_extractor(cache)

        if isinstance(extracted_activations, torch.Tensor):
            if extracted_activations.ndim != 3 or extracted_activations.shape[0] != 1:
                raise ValueError(f"Activation extractor returned tensor with unexpected shape: "
                                 f"{extracted_activations.shape}. Expected [1, seq_len, activation_dim].")
    
            activations_single = extracted_activations[0]

            model_seq_len = activations_single.shape[0]
    
            original_timed_indices = sorted(story.element_to_time_map.keys())
    
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
    

            super().__init__(data=relevant_activations, element_to_time_map=new_activation_to_time_map)
        elif isinstance(extracted_activations, list) and isinstance(extracted_activations[0], dict):
    
            super().__init__(data=extracted_activations, element_to_time_map=story.element_to_time_map)


if __name__ == '__main__':
    from transformer_lens import HookedTransformer
    from time_interval_iterator.time_interval_tokenized_story import TimeIntervalTokenizedStory
    from transformers import GPT2Tokenizer
    import torch

    model = HookedTransformer.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    test_text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer.encode(test_text, return_tensors="pt")[0]
    
    token_to_time_map = {
        i: (i * 0.1, (i + 1) * 0.1) for i in range(len(tokens))
    }
    story = TimeIntervalTokenizedStory(
        base_dir="",
        story_name="test_story",
        tokenizer=tokenizer,
        tokens=tokens,
        token_to_time_map=token_to_time_map
    )

    def extract_cache_list(cache, keys_to_return=None):
        # Get all keys and the sequence length from the first activation
        all_keys = list(cache.keys())
        if not all_keys:
            return []
        
        # If keys_to_return is provided, use only those keys that exist in the cache
        if keys_to_return is not None:
            keys = [key for key in keys_to_return if key in cache]
            if not keys:
                print(f"Warning: None of the requested keys {keys_to_return} found in cache. Available keys: {all_keys[:5]}...")
                return []
        else:
            keys = all_keys
        
        first_key = all_keys[0]  # Use any key to get shape info
        _, seq_len, dim = cache[first_key].shape
        
        # Create a list of dictionaries
        result = []
        for i in range(seq_len):
            entry = {}
            for key in keys:
                # Extract the i-th position from each activation
                entry[key] = cache[key][0, i]  # Shape: [dim]
            result.append(entry)
        
        return result

    def extract_layers_4_5_6(cache):
        layer_4 = cache["blocks.4.hook_resid_post"]
        layer_5 = cache["blocks.5.hook_resid_post"]
        layer_6 = cache["blocks.6.hook_resid_post"]
        return torch.cat([layer_4, layer_5, layer_6], dim=-1)

    activation_cache = build_aligned_activation_cache(
        story=story,
        model=model,
        activation_extractor=lambda cache: extract_cache_list(
            cache, 
            keys_to_return=["blocks.4.hook_resid_post", "blocks.5.hook_resid_post", "blocks.6.hook_resid_post"]
        ),
        #activation_extractor=extract_layers_4_5_6,
        activation_name="layers_4_5_6_resid_post"
    )

    print(f"Number of activations: {len(activation_cache.data)}")
    #print(f"Activation shape: {activation_cache.data.shape}")
    
    test_time = 0.15
    activation_idx = activation_cache.get_activation_for_seconds(test_time)
    if activation_idx is not None:
        print(f"\nActivation at time {test_time}s:")
        print(f"Activation index: {activation_idx}")
        #print(f"Activation vector shape: {activation_cache.get_activation_vector(activation_idx).shape}")
        print(f"Time interval: {activation_cache.get_seconds_for_activation(activation_idx)}")

    print("----")

    for i,x in enumerate(activation_cache):
        print(f"Activation {i}:")
        a, b, d = x
        print(f"a: {a}")
        print(f"b: {b}")
        for k,v in d.items():
            print(f"{k}: {v.shape}")
        if i>3:
            break
