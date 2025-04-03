import re
import os

story_name_maps = {
    "milkyway": "milkywayoriginal",
    "notthefalllongscram": "notthefallintact",
}

def get_subject_and_story_from_filename(filename, default_subject_id="sub-999", default_story_name="unknown"):
    """
    Extracts subject ID and story name from a NIFTI filename.
    Example: sub-041_task-lucy_space_space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz
    Should return: ('041', 'lucy_space')
    
    If filename contains "averaged", returns "averaged" as the subject_id and 
    the first word up to "_" in the base filename as the story_name.
    """
    # Extract the base filename without path and extension
    base_filename = os.path.basename(filename)
    base_filename = os.path.splitext(base_filename)[0]
    # Remove .nii if there's a double extension like .nii.gz
    base_filename = os.path.splitext(base_filename)[0] if base_filename.endswith('.nii') else base_filename
    
    # Check if filename contains "averaged"
    if "averaged" in base_filename:
        subject_id = "averaged"
        # Extract first word up to "_" as story name
        story_match = re.search(r'^([^_]+)', base_filename)
        story_name = story_match.group(1) if story_match else default_story_name
    else:
        subject_match = re.search(r'sub-(\w+)_', filename)
        story_match = re.search(r'_task-([^_]+(?:_[^_]+)*?)_(space|run)-', filename)
        subject_id = subject_match.group(1) if subject_match else None
        story_name = story_match.group(1) if story_match else None
        if not subject_id:
            print(f"Warning: Could not extract subject ID from {filename}")
            subject_id = default_subject_id
        if not story_name:
            story_match_fallback = re.search(r'_task-([^_]+)_', filename)
            story_name = story_match_fallback.group(1) if story_match_fallback else None
            if not story_name:
                print(f"Warning: Could not extract story name from {filename}")
                story_name = default_story_name
    if story_name in story_name_maps:
        story_name = story_name_maps[story_name]
    return subject_id, story_name