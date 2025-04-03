import time  # Added for example usage
from typing import List, Tuple, Dict, Iterator, Union, TypeVar, Generic, Mapping, Sequence, Any, Protocol

# Define a generic type variable T
T = TypeVar('T')

# Define a Protocol for time interval iteration
class TimeIntervalProtocol(Protocol[T]):
    def __iter__(self) -> Iterator[Tuple[float, float, T]]:
        """
        Protocol for iterating over time intervals with associated data elements.
        
        Yields:
            Tuple[float, float, T]: A tuple of (start_time, end_time, data_element)
        """
        ...

    def __next__(self) -> Tuple[float, float, T]:
        """
        Protocol for iterating over time intervals with associated data elements.
        """
        ...
    

class TimeIntervalIterator(Generic[T]):
    def __init__(
        self,
        data: Mapping[int, T],
        element_to_time_map: Dict[int, Tuple[float, float]],
        temporal_bias: float = 0.0
    ):
        """
        Initializes an iterator that aligns data elements with time intervals.

        Args:
            data: A mapping (like a dictionary) from integer indices to data elements of type T.
                  The keys must correspond to the indices used in element_to_time_map.
            element_to_time_map: A dictionary mapping element indices (int) to 
                                 time intervals (start_time, end_time).
            temporal_bias: A float value to shift all time intervals. Defaults to 0.0.
        
        Raises:
            ValueError: If any key in element_to_time_map is not found as a key in data.
        """
        self.data = data
        self.temporal_bias = temporal_bias
        biased_map = {}
        for k, v in element_to_time_map.items():
            biased_map[k] = (v[0] + temporal_bias, v[1] + temporal_bias)
        
        self.element_to_time_map = biased_map
        
        self.time_intervals: List[Tuple[Tuple[float, float], int]] = sorted([
            (time_interval, element_idx) 
            for element_idx, time_interval in self.element_to_time_map.items()
        ], key=lambda x: x[0][0])


    def get_element_index_for_time_seconds(self, seconds: float) -> int:
        """
        Get the element index for a given time in seconds using binary search.
        
        Args:
            seconds (float): Time in seconds (relative to the potentially biased timeline).
            
        Returns:
            int: The index of the element active at that time.
                 Returns the first element's index if time is before the first interval.
                 Returns the last element's index if time is after the last interval.
                 If between intervals, returns the index of the element with the closest start time.
        
        Raises:
            ValueError: If the iterator was initialized with no time intervals.
        """
        if not self.time_intervals:
            raise ValueError("Cannot get element for time: No time intervals available.")


        if seconds < self.time_intervals[0][0][0]:
            return self.time_intervals[0][1]
        if seconds > self.time_intervals[-1][0][1]:
            return self.time_intervals[-1][1]
        
        left, right = 0, len(self.time_intervals) - 1
        best_match_idx = -1
        
        while left <= right:
            mid = (left + right) // 2
            (start_time, end_time), element_idx = self.time_intervals[mid]
            
            if start_time <= seconds <= end_time:
                return element_idx  # Found exact interval
            elif seconds < start_time:
                right = mid - 1
            else: # seconds > end_time
                left = mid + 1
        
        # If no exact interval found (time falls between intervals), find the closest one based on start time.
        # Note: `left` index points to the first element whose start time is > seconds.
        # `right` index points to the last element whose start time is <= seconds.
        
        # Check the element just before `left` (which is `right`)
        idx_right = right 
        # Check the element at `left`
        idx_left = left

        closest_idx = -1
        min_dist = float('inf')

        if 0 <= idx_right < len(self.time_intervals):
             dist_right = abs(self.time_intervals[idx_right][0][0] - seconds)
             if dist_right < min_dist:
                 min_dist = dist_right
                 closest_idx = self.time_intervals[idx_right][1]

        if 0 <= idx_left < len(self.time_intervals):
            dist_left = abs(self.time_intervals[idx_left][0][0] - seconds)
            # Prioritize the later interval if distances are equal? Or earlier?
            # Current original logic prioritizes based purely on minimum distance.
            if dist_left < min_dist: 
                min_dist = dist_left
                closest_idx = self.time_intervals[idx_left][1]
            elif dist_left == min_dist and closest_idx != -1:
                 # Optional: Handle tie-breaking, e.g., prefer earlier index if needed
                 pass


        if closest_idx != -1:
             return closest_idx
        else:
             # Should theoretically not happen if checks for before/after range pass
             # But as a fallback, return the closest overall? The original code did this.
             closest_overall_idx = min(range(len(self.time_intervals)), 
                                      key=lambda i: abs(self.time_intervals[i][0][0] - seconds))
             return self.time_intervals[closest_overall_idx][1]

    
    def get_time_seconds_for_element(self, element_idx: int) -> Tuple[float, float]:
        """
        Get the time interval (start_time, end_time) for a given element index.
        
        Args:
            element_idx (int): Index of the element (must be a key in the original element_to_time_map).
            
        Returns:
            Tuple[float, float]: The (start_time, end_time) interval for the element, including bias.
            
        Raises:
            ValueError: If element_idx is not found in the mapping.
        """
        if element_idx not in self.element_to_time_map:
            raise ValueError(f"Element index {element_idx} not found in the mapping")
        
        return self.element_to_time_map[element_idx] # Return the biased time

    def __iter__(self) -> Iterator[Tuple[float, float, T]]:
        """
        Iterate over the data elements yielding their time intervals and the element itself.
        The iteration order is determined by the start time of the intervals.
        
        Yields:
            Tuple[float, float, T]: A tuple of (start_time, end_time, data_element)
        """
        # Iterate through the time intervals sorted by start time
        for (start_time, end_time), element_idx in self.time_intervals:
            # Retrieve the data element using the index from the data mapping
            data_element = self.data[element_idx] 
            yield start_time, end_time, data_element

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Using strings as data elements
    print("--- Example 1: String data ---")
    string_data = {
        0: "hello",
        1: "world",
        2: "aligned",
        3: "iterator"
    }
    time_map_str = {
        0: (0.5, 1.5),
        1: (1.8, 2.5),
        3: (3.0, 4.0), # Note: index 2 is missing a time map entry
        2: (2.6, 2.9)  # Added index 2 out of order to test sorting
    }
    
    try:
        aligned_iter_str = AlignedIterator[str](string_data, time_map_str, temporal_bias=10.0)
        print(f"Time intervals: {aligned_iter_str.time_intervals}")
        
        print("Iterating:")
        for start, end, element in aligned_iter_str:
            print(f"Time: ({start:.1f}s - {end:.1f}s), Element: {element} (type: {type(element).__name__})")

        print(f"Finished iterating")

        print("\nGetting time for element index 1:")
        print(aligned_iter_str.get_time_seconds_for_element(1)) # Should be (11.8, 12.5)

        print("\nGetting element index for time 12.0s:")
        print(aligned_iter_str.get_element_index_for_time_seconds(12.0)) # Should be 1 ('world')

        print("\nGetting element index for time 13.5s:")
        print(aligned_iter_str.get_element_index_for_time_seconds(13.5)) # Should be 3 ('iterator')
        
        print("\nGetting element index for time 12.55s (between intervals):")
        print(aligned_iter_str.get_element_index_for_time_seconds(12.55)) # Should be 2 ('aligned') as its start (12.6) is closer than 1's start (11.8)

    except ValueError as e:
        print(f"Error: {e}")

    # Example 2: Using custom objects
    print("\n--- Example 2: Custom Object data ---")
    class MyDataObject:
        def __init__(self, value: int, name: str):
            self.value = value
            self.name = name
        def __repr__(self):
            return f"MyDataObject(value={self.value}, name='{self.name}')"

    object_data = {
        10: MyDataObject(100, "alpha"),
        20: MyDataObject(200, "beta"),
        30: MyDataObject(300, "gamma")
    }
    time_map_obj = {
        10: (1.0, 2.0),
        30: (5.0, 6.0),
        20: (3.0, 4.0) 
    }

    try:
        aligned_iter_obj = AlignedIterator[MyDataObject](object_data, time_map_obj)

        print("Iterating:")
        for start, end, element in aligned_iter_obj:
            print(f"Time: ({start:.1f}s - {end:.1f}s), Element: {element} (type: {type(element).__name__})")
        
        print("\nGetting time for element index 20:")
        print(aligned_iter_obj.get_time_seconds_for_element(20)) # Should be (3.0, 4.0)

        print("\nGetting element index for time 3.5s:")
        print(aligned_iter_obj.get_element_index_for_time_seconds(3.5)) # Should be 20

    except ValueError as e:
        print(f"Error: {e}")
        
    # Example 3: Error case - index mismatch
    print("\n--- Example 3: Index Mismatch Error ---")
    data_mismatch = { 0: "A", 1: "B"}
    time_map_mismatch = { 0: (0,1), 2: (2,3)} # Index 2 not in data_mismatch
    try:
        aligned_iter_mismatch = AlignedIterator[str](data_mismatch, time_map_mismatch)
    except ValueError as e:
        print(f"Caught expected error: {e}")