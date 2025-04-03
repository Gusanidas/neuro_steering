import os
import csv
import torch
from typing import Union, Tuple, List, Dict, Optional
from transformers import PreTrainedTokenizerFast
from time_interval_iterator.time_interval_iterator import TimeIntervalIterator

class TimeIntervalTokenizedStory(TimeIntervalIterator):
    def __init__(
        self, 
        story_name: str, 
        tokenizer: PreTrainedTokenizerFast,
        base_dir: str = "",
        tokens: Optional[Union[torch.Tensor, List[int]]] = None, 
        token_to_time_map: Optional[Dict[int, Tuple[float, float]]] = None
    ):
        """
        Initialize an TimeIntervalTokenizedStory object.
        
        Args:
            story_name (str): Name of the story
            tokenizer: The tokenizer used to tokenize the story
            tokens: A tensor or list of token IDs
            token_to_time_map: A dictionary mapping token indices to time intervals (start_time, end_time)
        """
        self.story_name = story_name
        self.tokenizer = tokenizer
        self.base_dir = base_dir
        self._initialized = False

        if tokens is not None and token_to_time_map is not None:
            super().__init__(data=tokens, element_to_time_map=token_to_time_map)
            self.initialized = True

    def get_token_for_seconds(self, seconds: float) -> int:
        """Alias for get_element_for_time_seconds for backward compatibility"""
        self.ensure_initialized()
        return self.get_element_for_time_seconds(seconds)
    
    def get_seconds_for_token(self, token_idx: int) -> Tuple[float, float]:
        """Alias for get_time_seconds_for_element for backward compatibility"""
        self.ensure_initialized()
        return self.get_time_seconds_for_element(token_idx)

    def _initialize(self):
        if self._initialized:
            return
        self._build_from_tokenizer(self.story_name, self.tokenizer, self.base_dir)
        self._initialized = True
        
    def ensure_initialized(self):
        if not self._initialized:
            self._initialize()

    def __iter__(self):
        self.ensure_initialized()
        return super().__iter__()
    
    def _build_from_tokenizer(self, story_name: str, tokenizer, base_dir: str = ""):
        """
        Build an TimeIntervalTokenizedStory object from a story name and tokenizer.
        
        Args:
            story_name (str): Name of the story
            tokenizer: A tokenizer object (from HuggingFace)
            base_dir (str, optional): Base directory where the story files are located
            
        Returns:
            TimeIntervalTokenizedStory: A new TimeIntervalTokenizedStory object
        """
        story_dir = os.path.join(base_dir, story_name)
        align_file = os.path.join(story_dir, "align.csv")
        transcript_file = os.path.join(story_dir, "transcript.txt")
        
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        tokenized = tokenizer(transcript, return_offsets_mapping=True, return_tensors="pt")
        tokens = tokenized.input_ids[0]
        offset_mapping = tokenized.offset_mapping[0].numpy().tolist()
        
        word_timings = []
        with open(align_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                timing_data = {
                    'word': row[0] if len(row) > 0 else "",
                    'index': i
                }
                
                if len(row) >= 4 and row[2] and row[3]:
                    try:
                        timing_data['start_time'] = float(row[2])
                        timing_data['end_time'] = float(row[3])
                    except (ValueError, TypeError):
                        timing_data['start_time'] = None
                        timing_data['end_time'] = None
                else:
                    timing_data['start_time'] = None
                    timing_data['end_time'] = None
                
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
            if block[0] == 0 or block[-1] == len(word_timings) - 1:
                continue
                
            prev_word = word_timings[block[0] - 1]
            next_word = word_timings[block[-1] + 1]
            
            if prev_word['start_time'] is not None and next_word['end_time'] is not None:
                if len(block) == 1:
                    word_idx = block[0]
                    word_timings[word_idx]['start_time'] = prev_word['end_time']
                    word_timings[word_idx]['end_time'] = next_word['start_time']
                    processed_word_timings.append(word_timings[word_idx])
                else:
                    total_duration = next_word['start_time'] - prev_word['end_time']
                    segment_duration = total_duration / len(block)
                    
                    for i, word_idx in enumerate(block):
                        start = prev_word['end_time'] + i * segment_duration
                        end = start + segment_duration
                        word_timings[word_idx]['start_time'] = start
                        word_timings[word_idx]['end_time'] = end
                        processed_word_timings.append(word_timings[word_idx])
        
        processed_word_timings.sort(key=lambda x: x['index'])
        char_to_time = {}
        
        words = []
        word_starts = []
        
        position = 0
        while position < len(transcript) and transcript[position].isspace():
            position += 1
        
        word_start = position
        for i in range(position, len(transcript)):
            if i < len(transcript) and transcript not in "abcdefghijklmnopqrstuvwxyz":
                if i > word_start:  # Only add non-empty words
                    words.append(transcript[word_start:i])
                    word_starts.append(word_start)
                
                # Skip whitespace to find next word
                while i < len(transcript) and transcript[i].isspace():
                    i += 1
                
                word_start = i
        
        if word_start < len(transcript):
            words.append(transcript[word_start:])
            word_starts.append(word_start)
        
        for i, (word, word_start) in enumerate(zip(words, word_starts)):
            if i < len(processed_word_timings):
                timing = processed_word_timings[i]
                word_end = word_start + len(word)
                
                if timing['start_time'] is not None and timing['end_time'] is not None:
                    for char_idx in range(word_start, word_end):
                        char_to_time[char_idx] = (timing['start_time'], timing['end_time'])
        
        token_to_time_map = {}
        
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            if start_char == 0 and end_char == 0:
                continue
            
            times = []
            for char_idx in range(start_char, end_char):
                if char_idx in char_to_time:
                    times.append(char_to_time[char_idx])
            
            if times:
                start_time = min(t[0] for t in times)
                end_time = max(t[1] for t in times)
                token_to_time_map[token_idx] = (start_time, end_time)
        
        super().__init__(data=tokens, element_to_time_map=token_to_time_map)

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
        
        base_dir = "narratives/stimuli/gentle"
        story_name = "black"
        
        print(f"\n=== Testing AlignedTokenizedStory.build_from_tokenizer ===")
        print(f"Building AlignedTokenizedStory for '{story_name}'...")
        
        try:
            start_time = time.time()
            story = TimeIntervalTokenizedStory(
                story_name=story_name,
                tokenizer=tokenizer,
                base_dir=base_dir
            )
            print(f"AlignedTokenizedStory built in {time.time() - start_time:.2f} seconds")
            for token_idx, (start_time, end_time) in story:

                if token_idx>1800:  
                    token_text = story.tokenizer.decode([story.data[token_idx]])
                    print(f"Token {token_idx}: {start_time:.2f}s - {end_time:.2f}s, text: '{token_text}'")
            
            print(f"Total tokens: {len(story.data)}")
            print(f"Time-aligned tokens: {len(story.element_to_time_map)}")
            
            test_times = [10.0, 30.0, 60.0, 90.0]
            print("\nTesting get_token_for_seconds:")
            for test_time in test_times:
                token_idx = story.get_token_for_seconds(test_time)
                token_text = story.tokenizer.decode([story.data[token_idx]])
                print(f"Time {test_time}s -> Token index {token_idx}, text: '{token_text}'")
            
            test_tokens = [10, 50, 100, 200]
            print("\nTesting get_seconds_for_token:")
            for token_idx in test_tokens:
                if token_idx in story.element_to_time_map:
                    start, end = story.get_seconds_for_token(token_idx)
                    token_text = story.tokenizer.decode([story.data[token_idx]])
                    print(f"Token {token_idx} ('{token_text}') -> Time range: {start:.2f}s to {end:.2f}s")
                else:
                    print(f"Token {token_idx} does not have timing information")
            
            print("\nIterating through the first 5 tokens with timing:")
            count = 0
            for start_time, end_time, token in story:
                token_text = story.tokenizer.decode([token])
                print(f"Time range: {start_time:.2f}s - {end_time:.2f}s, Token: '{token_text}'")
                count += 1
                if count >= 5:
                    break
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Make sure the story '{story_name}' exists in '{base_dir}' with all required files.")
            
            print("\n=== Demonstrating with simple example data ===")
            
            tokens = torch.tensor([10, 20, 30, 40, 50])
            
            token_to_time_map = {
                0: (0.0, 2.5),    # Token 0 spans from 0s to 2.5s
                1: (2.5, 4.0),    # Token 1 spans from 2.5s to 4.0s
                2: (4.0, 7.5),    # Token 2 spans from 4.0s to 7.5s
                3: (7.5, 10.0),   # Token 3 spans from 7.5s to 10.0s
                4: (10.0, 15.0)   # Token 4 spans from 10.0s to 15.0s
            }
            
            example_story = TimeIntervalTokenizedStory(
                story_name="example",
                tokenizer=tokenizer,
                tokens=tokens,
                token_to_time_map=token_to_time_map
            )
            for start_time, end_time, token in example_story:
                token_text = example_story.tokenizer.decode([token])
                print(f"Time range: {start_time:.1f}s - {end_time:.1f}s, Token value: {token.item()}, Token text: '{token_text}'")
            
            print("\nTokens at specific times:")
            test_times = [1.0, 3.0, 5.0, 8.0, 12.0, 20.0]
            for time in test_times:
                token_idx = example_story.get_token_for_seconds(time)
                token_value = example_story.data[token_idx].item()
                print(f"At time {time:.1f}s: Token index {token_idx}, Value {token_value}")
            
            print("\nIterating through aligned data:")
            for start_time, end_time, token in example_story:
                print(f"Time range: {start_time:.1f}s - {end_time:.1f}s, Token value: {token.item()}")
            
            print("\nTime ranges for specific tokens:")
            for idx in range(len(tokens)):
                start_time, end_time = example_story.get_seconds_for_token(idx)
                print(f"Token {idx} (value {tokens[idx].item()}) is at time range: {start_time:.1f}s - {end_time:.1f}s")
        
    except ImportError:
        print("Error: This demo requires transformers with fast tokenizer support.")
        print("Install with: pip install transformers[tokenizers]")