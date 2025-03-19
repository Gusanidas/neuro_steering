import os

import csv
import torch
from typing import Union, Tuple, List, Dict, Optional

class TokenizedStory:
    def __init__(
        self, 
        story_name: str, 
        tokenizer, 
        tokens: Union[torch.Tensor, List[int]], 
        token_to_time_map: Dict[int, Tuple[float, float]]
    ):
        """
        Initialize a TokenizedStory object.
        
        Args:
            story_name (str): Name of the story
            tokenizer: The tokenizer used to tokenize the story
            tokens: A tensor or list of token IDs
            token_to_time_map: A dictionary mapping token indices to time intervals (start_time, end_time)
        """
        self.story_name = story_name
        self.tokenizer = tokenizer
        self.tokens = tokens if isinstance(tokens, torch.Tensor) else torch.tensor(tokens)
        self.token_to_time_map = token_to_time_map
        
        # Create a sorted list of (time_interval, token_index) tuples for binary search
        # Each tuple is ((start_time, end_time), token_index)
        self.time_intervals = sorted([
            ((start_time, end_time), token_idx) 
            for token_idx, (start_time, end_time) in token_to_time_map.items()
        ], key=lambda x: x[0][0])  # Sort by start_time
        self.base_dir = "narratives/stimuli/gentle/"
    
    def get_token_for_seconds(self, seconds: float) -> int:
        """
        Get the token index for a given time in seconds using binary search.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            int: The index of the token at that time
        """
        # Edge cases
        if not self.time_intervals:
            raise ValueError("No time intervals available")
        
        # If time is before the first interval or after the last one
        if seconds < self.time_intervals[0][0][0]:
            return self.time_intervals[0][1]
        if seconds > self.time_intervals[-1][0][1]:
            return self.time_intervals[-1][1]
        
        # Binary search to find the interval that contains the time
        left, right = 0, len(self.time_intervals) - 1
        
        while left <= right:
            mid = (left + right) // 2
            (start_time, end_time), token_idx = self.time_intervals[mid]
            
            # Check if the time is within this interval
            if start_time <= seconds <= end_time:
                return token_idx
            # If the time is before this interval
            elif seconds < start_time:
                right = mid - 1
            # If the time is after this interval
            else:
                left = mid + 1
        
        # If no exact interval is found, return the token with the closest start time
        closest_idx = min(range(len(self.time_intervals)), 
                          key=lambda i: abs(self.time_intervals[i][0][0] - seconds))
        return self.time_intervals[closest_idx][1]
    
    def get_seconds_for_token(self, token_idx: int) -> Tuple[float, float]:
        """
        Get the time interval for a given token index.
        
        Args:
            token_idx (int): Index of the token
            
        Returns:
            Tuple[float, float]: The (start_time, end_time) interval for the token
        """
        if token_idx not in self.token_to_time_map:
            raise ValueError(f"Token index {token_idx} not found in the mapping")
        
        return self.token_to_time_map[token_idx]
    
    @classmethod
    def build_from_tokenizer(cls, story_name: str, tokenizer, base_dir: str = ""):
        """
        Build a TokenizedStory object from a story name and tokenizer.
        
        Args:
            story_name (str): Name of the story
            tokenizer: A tokenizer object (from HuggingFace)
            base_dir (str, optional): Base directory where the story files are located
            
        Returns:
            TokenizedStory: A new TokenizedStory object
        """
        # Construct file paths
        story_dir = os.path.join(base_dir, story_name)
        align_file = os.path.join(story_dir, "align.csv")
        transcript_file = os.path.join(story_dir, "transcript.txt")
        
        # Read the transcript
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        # Tokenize the transcript
        tokenized = tokenizer(transcript, return_offsets_mapping=True, return_tensors="pt")
        tokens = tokenized.input_ids[0]
        offset_mapping = tokenized.offset_mapping[0].numpy().tolist()
        
        # Read the alignment file
        word_timings = []
        with open(align_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                timing_data = {
                    'word': row[0] if len(row) > 0 else "",
                    'index': i
                }
                
                # Handle missing timing data
                if len(row) >= 4 and row[2] and row[3]:
                    try:
                        timing_data['start_time'] = float(row[2])
                        timing_data['end_time'] = float(row[3])
                    except (ValueError, TypeError):
                        timing_data['start_time'] = None
                        timing_data['end_time'] = None
                else:
                    # Use None as placeholder for missing data
                    timing_data['start_time'] = None
                    timing_data['end_time'] = None
                
                print(f"Word: {timing_data['word']}, start_time: {timing_data['start_time']}, end_time: {timing_data['end_time']}, iteration: {i}")
                word_timings.append(timing_data)
        
        # Process missing timing data
        processed_word_timings = []
        missing_blocks = []
        current_missing_block = []
        
        # First pass: identify missing data blocks
        for i, timing in enumerate(word_timings):
            if timing['start_time'] is None or timing['end_time'] is None:
                current_missing_block.append(i)
            else:
                if current_missing_block:
                    missing_blocks.append(current_missing_block)
                    current_missing_block = []
                processed_word_timings.append(timing)
        
        # Add the last missing block if exists
        if current_missing_block:
            missing_blocks.append(current_missing_block)
        
        # Second pass: fill in missing data
        for block in missing_blocks:
            # Skip if it's the first or last word
            if block[0] == 0 or block[-1] == len(word_timings) - 1:
                continue
                
            # Get timing of surrounding words
            prev_word = word_timings[block[0] - 1]
            next_word = word_timings[block[-1] + 1]
            
            if prev_word['start_time'] is not None and next_word['end_time'] is not None:
                # If only one word with missing data, interpolate between prev and next
                if len(block) == 1:
                    word_idx = block[0]
                    word_timings[word_idx]['start_time'] = prev_word['end_time']
                    word_timings[word_idx]['end_time'] = next_word['start_time']
                    processed_word_timings.append(word_timings[word_idx])
                # If multiple words, divide the time evenly
                else:
                    total_duration = next_word['start_time'] - prev_word['end_time']
                    segment_duration = total_duration / len(block)
                    
                    for i, word_idx in enumerate(block):
                        start = prev_word['end_time'] + i * segment_duration
                        end = start + segment_duration
                        word_timings[word_idx]['start_time'] = start
                        word_timings[word_idx]['end_time'] = end
                        processed_word_timings.append(word_timings[word_idx])
        
        # Sort the processed timings by original index
        processed_word_timings.sort(key=lambda x: x['index'])
        
        # Create a character-to-time mapping
        char_to_time = {}
        
        # Split transcript into words and track their positions
        words = []
        word_starts = []
        
        position = 0
        # Skip initial whitespace
        while position < len(transcript) and transcript[position].isspace():
            position += 1
        
        # Find all words and their positions
        word_start = position
        for i in range(position, len(transcript)):
            if i < len(transcript) and transcript[i].isspace():
                if i > word_start:  # Only add non-empty words
                    words.append(transcript[word_start:i])
                    word_starts.append(word_start)
                
                # Skip whitespace to find next word
                while i < len(transcript) and transcript[i].isspace():
                    i += 1
                
                word_start = i
        
        # Add the last word if there is one
        if word_start < len(transcript):
            words.append(transcript[word_start:])
            word_starts.append(word_start)
        
        # Match words to alignment timings
        for i, (word, word_start) in enumerate(zip(words, word_starts)):
            if i < len(processed_word_timings):
                timing = processed_word_timings[i]
                word_end = word_start + len(word)
                
                # Only map if valid timing data exists
                if timing['start_time'] is not None and timing['end_time'] is not None:
                    # Map each character in the word to its time range
                    for char_idx in range(word_start, word_end):
                        char_to_time[char_idx] = (timing['start_time'], timing['end_time'])
        
        # Map tokens to time intervals
        token_to_time_map = {}
        
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            # Skip special tokens
            if start_char == 0 and end_char == 0:
                continue
            
            # Find all character time mappings for this token
            times = []
            for char_idx in range(start_char, end_char):
                if char_idx in char_to_time:
                    times.append(char_to_time[char_idx])
            
            if times:
                # Use the earliest start time and latest end time
                start_time = min(t[0] for t in times)
                end_time = max(t[1] for t in times)
                token_to_time_map[token_idx] = (start_time, end_time)
        
        return cls(
            story_name=story_name,
            tokenizer=tokenizer,
            tokens=tokens,
            token_to_time_map=token_to_time_map
        )


if __name__ == "__main__":
    try:
        from transformers import GPT2TokenizerFast
        import time
        
        print("Loading GPT-2 fast tokenizer...")
        try:
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        except Exception as e:
            print(f"Error loading fast tokenizer: {e}")
            print("Fast tokenizer is required for offset mapping functionality.")
            print("Make sure 'tokenizers' is installed. Try: pip install tokenizers")
            exit(1)
        
        # Set the base directory where "black" story is located
        base_dir = "narratives/stimuli/gentle"
        story_name = "black"
        
        print(f"Building TokenizedStory for '{story_name}'...")
        start_time = time.time()
        
        try:
            story = TokenizedStory.build_from_tokenizer(
                story_name=story_name,
                tokenizer=tokenizer,
                base_dir=base_dir
            )
            print(f"TokenizedStory built in {time.time() - start_time:.2f} seconds")
            
            # Show some basic information
            print(f"Total tokens: {len(story.tokens)}")
            print(f"Time-aligned tokens: {len(story.token_to_time_map)}")
            
            # Demonstrate get_token_for_seconds
            test_times = [10.0, 30.0, 60.0, 90.0]
            print("\nTesting get_token_for_seconds:")
            for test_time in test_times:
                token_idx = story.get_token_for_seconds(test_time)
                token_text = story.tokenizer.decode([story.tokens[token_idx]])
                print(f"Time {test_time}s -> Token index {token_idx}, text: '{token_text}'")
            
            # Demonstrate get_seconds_for_token
            test_tokens = [10, 50, 100, 200]
            print("\nTesting get_seconds_for_token:")
            for token_idx in test_tokens:
                if token_idx in story.token_to_time_map:
                    start, end = story.get_seconds_for_token(token_idx)
                    token_text = story.tokenizer.decode([story.tokens[token_idx]])
                    print(f"Token {token_idx} ('{token_text}') -> Time range: {start:.2f}s to {end:.2f}s")
                else:
                    print(f"Token {token_idx} does not have timing information")
            
            # Demonstrate looking up a specific word
            print("\nFinding a specific word ('once'):")
            word_to_find = "once"
            for i, token_id in enumerate(story.tokens):
                token_text = story.tokenizer.decode([token_id])
                if word_to_find in token_text:
                    if i in story.token_to_time_map:
                        start, end = story.get_seconds_for_token(i)
                        print(f"Found '{token_text}' at token {i}, time: {start:.2f}s to {end:.2f}s")
                    else:
                        print(f"Found '{token_text}' at token {i}, but no timing information available")
                    break
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Make sure the story '{story_name}' exists in '{base_dir}' with all required files.")
        
    except ImportError:
        print("Error: This demo requires transformers with fast tokenizer support.")
        print("Install with: pip install transformers[tokenizers]")