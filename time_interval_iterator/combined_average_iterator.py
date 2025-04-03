from time_interval_iterator.time_interval_iterator import TimeIntervalIterator
from typing import List, Tuple, Dict, Iterator, Union, Optional, TypeVar, Generic, Mapping, Any, Callable
from statistics import mean
import torch

T1 = TypeVar('T1')
T2 = TypeVar('T2')

class CombinedAverageIterator(Generic[T1, T2]):
    """
    Combines two TimeIntervalIterators, yielding aligned elements from both
    during periods where their time intervals overlap. If multiple elements from one
    iterator overlap with a single element from the other, they are averaged before yielding.
    """
    def __init__(
        self, 
        iter1: TimeIntervalIterator[T1], 
        iter2: TimeIntervalIterator[T2],
        avg_func1: Callable[[List[T1]], T1] = None,
        avg_func2: Callable[[List[T2]], T2] = None
    ):
        """
        Initialize the CombinedAverageIterator.

        Args:
            iter1: The first TimeIntervalIterator yielding elements of type T1.
            iter2: The second TimeIntervalIterator yielding elements of type T2.
            avg_func1: Optional function to average multiple T1 elements. If None, uses a default 
                       approach that assumes T1 objects can be averaged with the builtin mean function.
            avg_func2: Optional function to average multiple T2 elements. If None, uses a default 
                       approach that assumes T2 objects can be averaged with the builtin mean function.
        """
        self.iter1: TimeIntervalIterator[T1] = iter1
        self.iter2: TimeIntervalIterator[T2] = iter2
        
        # Default averaging functions
        self.avg_func1 = avg_func1 if avg_func1 is not None else self._default_average
        self.avg_func2 = avg_func2 if avg_func2 is not None else self._default_average
        
        # Internal state
        self._iter1_buffer: List[Tuple[float, float, T1]] = []
        self._iter2_buffer: List[Tuple[float, float, T2]] = []
        self._iter1_exhausted: bool = False
        self._iter2_exhausted: bool = False
        self._current_interval: Optional[Tuple[float, float]] = None

    def _default_average(self, items):
        """Default averaging function that tries to use the mean function."""
        if not items:
            return None
        if len(items) == 1:
            return items[0]
        if isinstance(items[0], torch.Tensor):
            return torch.mean(torch.stack(items), dim=0)
        try:
            return mean(items)
        except TypeError:
            # If items can't be directly averaged, return the first one
            # Users should provide a custom avg_func for complex types
            return items[0]

    def __iter__(self) -> 'CombinedAverageIterator[T1, T2]':
        """Return self as the iterator."""
        # Reset the internal state
        self._iter1_buffer = []
        self._iter2_buffer = []
        self._iter1_exhausted = False
        self._iter2_exhausted = False
        self._current_interval = None
        return self

    def _fill_buffers(self) -> None:
        """
        Fills the buffers with items from both iterators that could potentially
        contribute to the next overlapping interval.
        """
        # Determine the next possible interval boundary
        if not self._iter1_buffer and not self._iter2_buffer:
            # Both buffers are empty, need to get at least one item from each
            try:
                s1, e1, d1 = next(iter(self.iter1))
                self._iter1_buffer.append((s1, e1, d1))
            except StopIteration:
                self._iter1_exhausted = True
                
            try:
                s2, e2, d2 = next(iter(self.iter2))
                self._iter2_buffer.append((s2, e2, d2))
            except StopIteration:
                self._iter2_exhausted = True
                
            if self._iter1_exhausted or self._iter2_exhausted:
                return
        
        # Find the maximum start time and minimum end time from current buffers
        if self._iter1_buffer and self._iter2_buffer:
            s1_min = min(s for s, _, _ in self._iter1_buffer)
            e1_max = max(e for _, e, _ in self._iter1_buffer)
            s2_min = min(s for s, _, _ in self._iter2_buffer)
            e2_max = max(e for _, e, _ in self._iter2_buffer)
            
            overlap_start = max(s1_min, s2_min)
            overlap_end = min(e1_max, e2_max)
            
            if overlap_start < overlap_end:
                self._current_interval = (overlap_start, overlap_end)
                return
        
        # Keep loading from the iterator with the earlier ending time
        # until we find an overlap or exhaust an iterator
        while not self._iter1_exhausted and not self._iter2_exhausted:
            if not self._iter1_buffer:
                try:
                    s1, e1, d1 = next(iter(self.iter1))
                    self._iter1_buffer.append((s1, e1, d1))
                except StopIteration:
                    self._iter1_exhausted = True
                    break
            
            if not self._iter2_buffer:
                try:
                    s2, e2, d2 = next(iter(self.iter2))
                    self._iter2_buffer.append((s2, e2, d2))
                except StopIteration:
                    self._iter2_exhausted = True
                    break
            
            # Get the current buffer boundaries
            s1_min = min(s for s, _, _ in self._iter1_buffer)
            e1_max = max(e for _, e, _ in self._iter1_buffer)
            s2_min = min(s for s, _, _ in self._iter2_buffer)
            e2_max = max(e for _, e, _ in self._iter2_buffer)
            
            # Find potential overlap
            overlap_start = max(s1_min, s2_min)
            overlap_end = min(e1_max, e2_max)
            
            if overlap_start < overlap_end:
                self._current_interval = (overlap_start, overlap_end)
                return
            
            # No overlap - advance the earlier ending iterator
            if e1_max <= e2_max:
                try:
                    s1, e1, d1 = next(iter(self.iter1))
                    self._iter1_buffer.append((s1, e1, d1))
                except StopIteration:
                    self._iter1_exhausted = True
            else:
                try:
                    s2, e2, d2 = next(iter(self.iter2))
                    self._iter2_buffer.append((s2, e2, d2))
                except StopIteration:
                    self._iter2_exhausted = True

    def _get_overlapping_items(self) -> Tuple[List[T1], List[T2]]:
        """
        Returns lists of items from both iterators that overlap with the current interval.
        Also removes items from the buffers that are completely before the interval.
        """
        if self._current_interval is None:
            return [], []
        
        overlap_start, overlap_end = self._current_interval
        
        # Clean up buffers by removing items that end before overlap_start
        self._iter1_buffer = [
            (s, e, d) for s, e, d in self._iter1_buffer if e > overlap_start
        ]
        self._iter2_buffer = [
            (s, e, d) for s, e, d in self._iter2_buffer if e > overlap_start
        ]
        
        # Get overlapping items
        items1 = [
            d for s, e, d in self._iter1_buffer 
            if s < overlap_end and e > overlap_start
        ]
        items2 = [
            d for s, e, d in self._iter2_buffer 
            if s < overlap_end and e > overlap_start
        ]
        
        # Add more items if needed to complete the overlap
        while not self._iter1_exhausted:
            try:
                s, e, d = next(iter(self.iter1))
                if s >= overlap_end:
                    # This item is after our current interval, save it for later
                    self._iter1_buffer.append((s, e, d))
                    break
                self._iter1_buffer.append((s, e, d))
                if s < overlap_end and e > overlap_start:
                    items1.append(d)
            except StopIteration:
                self._iter1_exhausted = True
                break
        
        while not self._iter2_exhausted:
            try:
                s, e, d = next(iter(self.iter2))
                if s >= overlap_end:
                    # This item is after our current interval, save it for later
                    self._iter2_buffer.append((s, e, d))
                    break
                self._iter2_buffer.append((s, e, d))
                if s < overlap_end and e > overlap_start:
                    items2.append(d)
            except StopIteration:
                self._iter2_exhausted = True
                break
                
        return items1, items2

    def __next__(self) -> Tuple[float, float, Tuple[T1, T2]]:
        """
        Yields the next aligned time interval and corresponding averaged data elements
        as a tuple (element1, element2).

        Raises:
            StopIteration: When both input iterators are exhausted or a valid
                           overlap cannot be found.

        Returns:
            Tuple[float, float, Tuple[T1, T2]]:
                (start_time, end_time, (avg_data_element1, avg_data_element2))
                corresponding to the overlapping time interval.
        """
        # Process buffers to find next overlap interval
        self._fill_buffers()
        
        if self._current_interval is None:
            raise StopIteration("No more overlapping intervals")
        
        overlap_start, overlap_end = self._current_interval
        
        # Get all items from both iterators that overlap with this interval
        items1, items2 = self._get_overlapping_items()
        
        if not items1 or not items2:
            # This shouldn't happen with proper _fill_buffers implementation
            raise StopIteration("No overlapping items found")
        
        # Average the items
        avg_item1 = self.avg_func1(items1)
        avg_item2 = self.avg_func2(items2)
        
        # Clear current interval to prepare for next iteration
        self._current_interval = None
        
        return (overlap_start, overlap_end, (avg_item1, avg_item2))