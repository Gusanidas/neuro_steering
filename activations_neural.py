import os
import torch
from transformers import AutoTokenizer
from aligned_tokenized_story import AlignedTokenizedStory
from aligned_neural import AlignedNeuralData
from combined_iterator import CombinedIterator
from map_region import map_region_to_harvard_oxford
from transformer_lens import HookedTransformer
from aligned_activation_cache import AlignedActivationsCache
from centered_kernel_alignment import centered_kernel_alignment

# Configuration
STORY_NAME = "pieman"
STORY_BASE_DIR = "narratives/stimuli/gentle"
SUBJECT_ID = "sub-015"
BRAIN_REGION = "temporal"
NIFTI_FILE_PATH = "derivatives/afni-smooth/sub-015/func/sub-015_task-pieman_run-1_space-MNI152NLin2009cAsym_res-native_desc-sm6_bold.nii.gz"
NEURAL_BASE_DIR = "narratives"
TR_VALUE = 1.5
MODEL_NAME = "phi-1"

def extract_layers_4_5_6(cache):
    layer_4 = cache["blocks.4.hook_resid_post"]
    layer_5 = cache["blocks.5.hook_resid_post"]
    layer_6 = cache["blocks.6.hook_resid_post"]
    return torch.cat([layer_4, layer_5, layer_6], dim=-1)

def main():
    model = HookedTransformer.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
    
    aligned_story = AlignedTokenizedStory.build_from_tokenizer(
        story_name=STORY_NAME,
        tokenizer=tokenizer,
        base_dir=STORY_BASE_DIR
    )
    print(f"Story loaded: {len(aligned_story.data)} tokens, {len(aligned_story.element_to_time_map)} time-aligned")
    
    activation_cache = AlignedActivationsCache.build_from_tokenized_story(
        story=aligned_story,
        model=model,
        activation_extractor=extract_layers_4_5_6,
        activation_name="layers_4_5_6_resid_post"
    )
    print(f"Activation cache created: {activation_cache.data.shape}")
    
    aligned_neural = AlignedNeuralData.build_from_file(
        story_name=STORY_NAME,
        subject=SUBJECT_ID,
        region=BRAIN_REGION,
        file_path=NIFTI_FILE_PATH,
        region_to_label_func=map_region_to_harvard_oxford,
        TR=TR_VALUE,
        base_dir=NEURAL_BASE_DIR
    )
    print(f"Neural data loaded: {aligned_neural.data.shape} (Timepoints x Voxels)")
    
    # Create and iterate through combined data (neural data and model activations)
    combined_iterator = CombinedIterator(activation_cache, aligned_neural)
    
    activations = []
    neural_data = []
    # Show first 10 segments
    for i, (start_time, end_time, activation_data, neural_data_segment) in enumerate(combined_iterator):
        activations.append(activation_data)
        neural_data.append(neural_data_segment)
            
        if i < 10:
            print(f"\nSegment {i+1}: [{start_time:.3f}s - {end_time:.3f}s]")
            print(f"Activation: Shape={activation_data.shape}")
            print(f"Neural Data: Shape={neural_data_segment.shape}")

    activations = torch.stack(activations)
    neural_data = torch.stack(neural_data)

    cka = centered_kernel_alignment(activations, neural_data)
    print(f"CKA: {cka}")

if __name__ == "__main__":
    main()