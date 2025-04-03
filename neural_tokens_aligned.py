import os
import torch
from transformers import GPT2TokenizerFast
import time
from aligned_iterator import AlignedIterator
from aligned_tokenized_story import AlignedTokenizedStory
from aligned_neural import AlignedNeuralData
from combined_iterator import CombinedIterator
from map_region import map_region_to_harvard_oxford

# Configuration
STORY_NAME = "pieman"
STORY_BASE_DIR = "narratives/stimuli/gentle"
SUBJECT_ID = "sub-015"
BRAIN_REGION = "temporal"
NIFTI_FILE_PATH = "derivatives/afni-smooth/sub-015/func/sub-015_task-pieman_run-1_space-MNI152NLin2009cAsym_res-native_desc-sm6_bold.nii.gz"
NEURAL_BASE_DIR = "narratives"
TR_VALUE = 1.5
TOKENIZER_NAME = "gpt2"

def main():
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)
    
    aligned_story = AlignedTokenizedStory.build_from_tokenizer(
        story_name=STORY_NAME,
        tokenizer=tokenizer,
        base_dir=STORY_BASE_DIR
    )
    print(f"Story loaded: {len(aligned_story.data)} tokens, {len(aligned_story.element_to_time_map)} time-aligned")
    
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
    
    combined_iterator = CombinedIterator(aligned_story, aligned_neural)
    
    for i, (start_time, end_time, token_data, neural_data_segment) in enumerate(combined_iterator):
        if i >= 10:
            break
            
        token_id = token_data.item()
        token_text = tokenizer.decode([token_id])
        
        print(f"\nSegment {i+1}: [{start_time:.3f}s - {end_time:.3f}s]")
        print(f"Token: ID={token_id}, Text='{token_text}'")
        print(f"Neural Data: Shape={neural_data_segment.shape}")

if __name__ == "__main__":
    main()