import torch
from typing import List, Tuple, Dict, Iterator, Union

class AlignedIterator:
    def __init__(
        self,
        data: Union[torch.Tensor, List],
        element_to_time_map: Dict[int, Tuple[float, float]]
    ):
        """
        Initialize an AlignedIterator object.
        
        Args:
            data: A tensor or list of data elements (will be converted to a tensor if not already one)
            element_to_time_map: A dictionary mapping element indices to time intervals (start_time, end_time)
        """
        self.data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
        self.element_to_time_map = element_to_time_map
        
        # Create a sorted list of (time_interval, element_index) tuples for binary search
        # Each tuple is ((start_time, end_time), element_index)
        self.time_intervals = sorted([
            ((start_time, end_time), element_idx) 
            for element_idx, (start_time, end_time) in element_to_time_map.items()
        ], key=lambda x: x[0][0])  # Sort by start_time
    
    def get_element_for_time_seconds(self, seconds: float) -> int:
        """
        Get the element index for a given time in seconds using binary search.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            int: The index of the element at that time
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
            (start_time, end_time), element_idx = self.time_intervals[mid]
            
            # Check if the time is within this interval
            if start_time <= seconds <= end_time:
                return element_idx
            # If the time is before this interval
            elif seconds < start_time:
                right = mid - 1
            # If the time is after this interval
            else:
                left = mid + 1
        
        # If no exact interval is found, return the element with the closest start time
        closest_idx = min(range(len(self.time_intervals)), 
                          key=lambda i: abs(self.time_intervals[i][0][0] - seconds))
        return self.time_intervals[closest_idx][1]
    
    def get_time_seconds_for_element(self, element_idx: int) -> Tuple[float, float]:
        """
        Get the time interval for a given element index.
        
        Args:
            element_idx (int): Index of the element
            
        Returns:
            Tuple[float, float]: The (start_time, end_time) interval for the element
        """
        if element_idx not in self.element_to_time_map:
            raise ValueError(f"Element index {element_idx} not found in the mapping")
        
        return self.element_to_time_map[element_idx]
    
    def __iter__(self) -> Iterator[Tuple[float, float, torch.Tensor]]:
        """
        Iterate over the data elements with their time intervals.
        
        Yields:
            Tuple[float, float, torch.Tensor]: A tuple of (start_time, end_time, data_element)
        """
        # Iterate through all elements that have time intervals
        sorted_indices = sorted(self.element_to_time_map.keys())
        for element_idx in sorted_indices:
            start_time, end_time = self.element_to_time_map[element_idx]
            data_element = self.data[element_idx]
            yield start_time, end_time, data_element

if __name__ == "__main__":
    # Create some sample data
    data = torch.tensor([10, 20, 30, 40, 50])
    
    # Create a mapping of element indices to time intervals
    element_to_time_map = {
        0: (0.0, 2.5),    # Element 0 spans from 0s to 2.5s
        1: (2.5, 4.0),    # Element 1 spans from 2.5s to 4.0s
        2: (4.0, 7.5),    # Element 2 spans from 4.0s to 7.5s
        3: (7.5, 10.0),   # Element 3 spans from 7.5s to 10.0s
        4: (10.0, 15.0)   # Element 4 spans from 10.0s to 15.0s
    }
    
    # Create the aligned iterator
    aligned_iter = AlignedIterator(data, element_to_time_map)
    
    # Demonstrate getting elements at specific times
    print("Elements at specific times:")
    test_times = [1.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    for time in test_times:
        element_idx = aligned_iter.get_element_for_time_seconds(time)
        element_value = data[element_idx].item()
        print(f"At time {time:.1f}s: Element index {element_idx}, Value {element_value}")
    
    # Demonstrate iterating through the aligned data
    print("\nIterating through aligned data:")
    for start_time, end_time, value in aligned_iter:
        print(f"Time range: {start_time:.1f}s - {end_time:.1f}s, Value: {value.item()}")
    
    # Demonstrate getting time for a specific element
    print("\nTime ranges for specific elements:")
    for idx in range(len(data)):
        start_time, end_time = aligned_iter.get_time_seconds_for_element(idx)
        print(f"Element {idx} (value {data[idx].item()}) is at time range: {start_time:.1f}s - {end_time:.1f}s")


