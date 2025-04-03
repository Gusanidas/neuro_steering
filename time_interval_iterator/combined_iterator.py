from time_interval_iterator.time_interval_iterator import TimeIntervalIterator
from typing import List, Tuple, Dict, Iterator, Union, Optional, TypeVar, Generic, Mapping, Any

T1 = TypeVar('T1')
T2 = TypeVar('T2')

class CombinedIterator(Generic[T1, T2]):
    """
    Combines two AlignedIterators, yielding aligned elements from both
    during periods where their time intervals overlap.
    """
    def __init__(self, iter1: TimeIntervalIterator[T1], iter2: TimeIntervalIterator[T2]):
        """
        Initialize the CombinedIterator.

        Args:
            iter1: The first AlignedIterator yielding elements of type T1.
            iter2: The second AlignedIterator yielding elements of type T2.
        """
        self.aligned_iter1: TimeIntervalIterator[T1] = iter1
        self.aligned_iter2: TimeIntervalIterator[T2] = iter2
        
        self.iter1_iter: Optional[Iterator[Tuple[float, float, T1]]] = None
        self.iter2_iter: Optional[Iterator[Tuple[float, float, T2]]] = None
        self.current1: Optional[Tuple[float, float, T1]] = None
        self.current2: Optional[Tuple[float, float, T2]] = None

    def __iter__(self) -> 'CombinedIterator[T1, T2]':
        """
        Initialize the internal iterators and fetch the first elements.
        Returns self as the iterator object.
        """
        self.iter1_iter = iter(self.aligned_iter1)
        self.iter2_iter = iter(self.aligned_iter2)

        try:
            if self.iter1_iter is not None:
                self.current1 = next(self.iter1_iter)
            else:
                 self.current1 = None 
        except StopIteration:
            self.current1 = None 
        
        try:
            if self.iter2_iter is not None:
                 self.current2 = next(self.iter2_iter)
            else:
                 self.current2 = None 
        except StopIteration:
            self.current2 = None 

        return self

    def __next__(self) -> Tuple[float, float, Tuple[T1, T2]]:
        """
        Yields the next aligned time interval and corresponding data elements
        as a tuple (element1, element2).

        Raises:
            StopIteration: When both input iterators are exhausted or a valid
                           overlap cannot be found.

        Returns:
            Tuple[float, float, Tuple[T1, T2]]:
                (start_time, end_time, (data_element1, data_element2))
                corresponding to the overlapping time interval.
        """
        if self.iter1_iter is None or self.iter2_iter is None:
             raise StopIteration("Iterators not properly initialized.")

        while True: 
            if self.current1 is None or self.current2 is None:
                raise StopIteration("One or both iterators are exhausted")

            s1, e1, d1 = self.current1
            s2, e2, d2 = self.current2

            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)

            if overlap_start < overlap_end:
                result: Tuple[float, float, Tuple[T1, T2]] = (overlap_start, overlap_end, (d1, d2))

                advance1 = (e1 == overlap_end)
                advance2 = (e2 == overlap_end)

                if advance1:
                    try:
                        self.current1 = next(self.iter1_iter)
                    except StopIteration:
                        self.current1 = None 
                if advance2:
                    try:
                        self.current2 = next(self.iter2_iter)
                    except StopIteration:
                        self.current2 = None 
                
                return result 

            else:
                advance1 = (e1 <= e2) 
                advance2 = (e2 <= e1) 

                try: 
                    if advance1:
                        if self.current1 is not None:
                            self.current1 = next(self.iter1_iter)

                    if advance2 and not (advance1 and self.current1 is None):
                         if self.current2 is not None:
                            self.current2 = next(self.iter2_iter)

                except StopIteration:
                     if advance1 and self.current1 is not None:
                         self.current1 = None
                     if advance2 and self.current2 is not None:
                         self.current2 = None
                     continue
                



if __name__ == "__ain__":
    print("--- Example 1 ---")
    data1 = [5.5, 6.5]
    map1 = {
        0: (9.0, 10.0),
        1: (10.0, 25.0)
    }
    aligned_iter1 = TimeIntervalIterator(data1, map1)

    data2 = [100, 200, 300, 400, 500]
    map2 = {
        0: (0.0, 3.0),
        1: (3.0, 8.0),
        2: (8.0, 15.0),
        3: (15.0, 17.0),
        4: (17.0, 25.0)
    }
    aligned_iter2 = TimeIntervalIterator(data2, map2)

    print("--- Aligned Iterator 1 Output ---")
    for s, e, d in aligned_iter1:
         print(f"({s:.1f}, {e:.1f}, {d})")


    print("\n--- Aligned Iterator 2 Output ---")
    for s, e, d in aligned_iter2:
         print(f"({s:.1f}, {e:.1f}, {d})")

    print("\n--- Combined Iterator Output ---")
    combined_iter = CombinedIterator(aligned_iter1, aligned_iter2)
    try:
        for start, end, elem1, elem2 in combined_iter:
            print(f"({start:.1f}, {end:.1f}, {elem1}, {elem2})")
    except StopIteration as e:
        print(f"Iteration finished: {e}")

    print("\n--- Example with slight non-alignment at start ---")
    data3 = [5.5]
    map3 = {0: (1.0, 5.0)}
    aligned_iter3 = TimeIntervalIterator(data3, map3)

    data4 = [99]
    map4 = {0: (0.0, 5.0)}
    aligned_iter4 = TimeIntervalIterator(data4, map4)

    combined_iter_2 = CombinedIterator(aligned_iter3, aligned_iter4)
    try:
        for start, end, elem1, elem2 in combined_iter_2:
            print(f"({start:.1f}, {end:.1f}, {elem1}, {elem2})")
    except StopIteration as e:
         print(f"Iteration finished: {e}")

    print("\n--- Example where one iterator is shorter ---")
    data5 = [5.5, 6.5]
    map5 = {0: (0.0, 10.0), 1: (10.0, 20.0)}
    aligned_iter5 = TimeIntervalIterator(data5, map5)

    data6 = [-30]
    map6 = {0: (0.0, 5.0)}
    aligned_iter6 = TimeIntervalIterator(data6, map6)

    combined_iter_3 = CombinedIterator(aligned_iter5, aligned_iter6)
    try:
        for start, end, elem1, elem2 in combined_iter_3:
            print(f"({start:.1f}, {end:.1f}, {elem1}, {elem2})")
    except StopIteration as e:
         print(f"Iteration finished: {e}") # Expected: Iteration stops after (0.0, 5.0)