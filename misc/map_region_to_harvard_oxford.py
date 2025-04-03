import nibabel as nib
import numpy as np
from nilearn import datasets, image
from typing import Dict, List, Union



def map_region_to_harvard_oxford(region: str, atlas) -> Dict[str, int]:
    """
    Maps a brain region name to Harvard-Oxford atlas labels, including specific language-related areas.

    Args:
        region (str): Name of the brain region (e.g., 'temporal', 'frontal', 'language')
        atlas: Harvard-Oxford atlas object from nilearn

    Returns:
        Dict[str, int]: Dictionary mapping label names to indices
    """
    # Get available labels from the atlas
    labels = atlas['labels']

    # Dictionary to map regions to specific areas in the Harvard-Oxford atlas
    region_mapping = {
        'temporal': [
            'Temporal Pole', 'Superior Temporal Gyrus', 'Middle Temporal Gyrus',
            'Inferior Temporal Gyrus', 'Temporal Occipital Fusiform Cortex'
        ],
        'frontal': [
            'Frontal Pole', 'Superior Frontal Gyrus', 'Middle Frontal Gyrus',
            'Inferior Frontal Gyrus', 'Frontal Orbital Cortex', 'Frontal Operculum Cortex'
        ],
        'parietal': [
            'Postcentral Gyrus', 'Superior Parietal Lobule', 'Supramarginal Gyrus',
            'Angular Gyrus', 'Precuneous Cortex'
        ],
        'occipital': [
            'Occipital Pole', 'Lateral Occipital Cortex', 'Lingual Gyrus',
            'Cuneal Cortex', 'Occipital Fusiform Gyrus'
        ],
        'language': [  # Specific language-related areas
            'Inferior Frontal Gyrus, pars opercularis',  # Broca's area (BA44)
            'Inferior Frontal Gyrus, pars triangularis',  # Broca's area (BA45)
            'Superior Temporal Gyrus, posterior division',  # Wernicke's area
            'Angular Gyrus',
            'Supramarginal Gyrus, posterior division',
            'Middle Temporal Gyrus, posterior division',
            'Planum Temporale',
            'Temporal Pole',
            'Heschl\'s Gyrus (includes H1 and H2)',
            'Frontal Operculum Cortex'
        ],
        'broca' : [
            'Inferior Frontal Gyrus, pars opercularis',  # Broca's area (BA44)
            'Inferior Frontal Gyrus, pars triangularis',  # Broca's area (BA45)
        ], 
        'wernicke' : [
            'Superior Temporal Gyrus, posterior division',  # Wernicke's area
        ],
        'auditory': [
            'Planum Polare',
            'Primary Auditory Cortex (Heschl\'s Gyrus)'
        ],
        'memory': [
            'Hippocampus',
            'Parahippocampal Cortex',
            'Entorhinal Cortex'
        ],
        'semantic': [
            'Anterior Temporal Lobes (ATL)',
            'Inferior Parietal Lobule (IPL)'
        ],
        'subcortical': [
         'Thalamus',
         'Caudate',             # Part of Basal Ganglia / Striatum
         'Putamen',            # Part of Basal Ganglia / Striatum
         'Pallidum',           # Part of Basal Ganglia
         'Hippocampus',        # Already in 'memory' but useful standalone or in a broader 'limbic' group
         'Amygdala',           # Key for emotion processing
         'Accumbens'           # Part of reward circuitry
     ],
     # Or more specific basal ganglia grouping:
     'basal_ganglia': [
         'Caudate',
         'Putamen',
        'Pallidum',
         'Accumbens' # Nucleus Accumbens is often grouped here
        ],
        'limbic': [
         'Hippocampus',
         'Amygdala',
         'Parahippocampal Gyrus, anterior division', # Often includes Entorhinal cortex functionally
         'Parahippocampal Gyrus, posterior division',
         'Cingulate Gyrus, anterior division',
         'Cingulate Gyrus, posterior division',
         'Temporal Pole', # Connects closely with limbic structures
         # 'Fornix' is anatomically part of it, but often not a standard HO label
     ],
    }

    # Default to empty if region not found
    target_areas = region_mapping.get(region.lower(), [])

    # Find indices for the target areas
    label_indices = {}
    for i, label in enumerate(labels):
        if label in target_areas:
            label_indices[label] = i

    return label_indices


if __name__ == "__main__":
    # Load the Harvard-Oxford atlas
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr50-2mm')
    
    # Map a region to its label indices
    region = "temporal"
    print(map_region_to_harvard_oxford(region, atlas))