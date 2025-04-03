from typing import List, Tuple, Iterator, TypeVar, Protocol, Generic, Optional
from time_interval_iterator.time_interval_iterator import TimeIntervalIterator, TimeIntervalProtocol

T = TypeVar('T')



class TimeIntervalIteratorList(Generic[T]):
    """
    A class that combines multiple TimeIntervalProtocol iterators into a single seamless iterator.
    The time intervals from one iterator flow into the next, with times adjusted to ensure continuity.
    """
    
    def __init__(self, iterators: List[TimeIntervalProtocol[T]]):
        """
        Initialize with a list of TimeIntervalProtocol iterators.
        
        Args:
            iterators: List of objects implementing the TimeIntervalProtocol
        """
        if not iterators:
            raise ValueError("Cannot initialize with an empty list of iterators")
        
        self.iterators = iterators
        self.current_iterator_index = 0
        self.current_iterator = None
        self.accumulated_time = 0.0
        self._initialized = False
    
    def __iter__(self) -> Iterator[Tuple[float, float, T]]:
        """
        Return self as iterator.
        """
        self.current_iterator_index = 0
        self.accumulated_time = 0.0
        self._initialized = False
        return self
    
    def __next__(self) -> Tuple[float, float, T]:
        """
        Get the next time interval and data element, adjusting time to ensure continuity
        between different iterators.
        
        Returns:
            Tuple[float, float, T]: A tuple of (adjusted_start_time, adjusted_end_time, data_element)
            
        Raises:
            StopIteration: When all iterators are exhausted
        """
        if not self._initialized:
            if self.current_iterator_index >= len(self.iterators):
                raise StopIteration
            
            self.current_iterator = iter(self.iterators[self.current_iterator_index])
            self._initialized = True
        
        try:
            start_time, end_time, data_element = next(self.current_iterator)
            
            adjusted_start_time = start_time + self.accumulated_time
            adjusted_end_time = end_time + self.accumulated_time
            
            return adjusted_start_time, adjusted_end_time, data_element
            
        except StopIteration:
            self.current_iterator_index += 1
            
            if self.current_iterator_index >= len(self.iterators):
                raise StopIteration
            
            self.current_iterator = iter(self.iterators[self.current_iterator_index])
            
            start_time, end_time, data_element = next(self.current_iterator)
            
            previous_iterator = self.iterators[self.current_iterator_index - 1]
            last_time = 0.0
            
            for prev_start, prev_end, _ in previous_iterator:
                if prev_end > last_time:
                    last_time = prev_end
            
            self.accumulated_time += last_time
            
            adjusted_start_time = start_time + self.accumulated_time
            adjusted_end_time = end_time + self.accumulated_time
            
            return adjusted_start_time, adjusted_end_time, data_element
            
    def reset(self):
        """
        Reset the iterator to the beginning.
        """
        self.current_iterator_index = 0
        self.accumulated_time = 0.0
        self._initialized = False