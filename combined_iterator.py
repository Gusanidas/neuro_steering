import torch
from typing import List, Tuple, Dict, Iterator, Union, Optional
from aligned_iterator import AlignedIterator
AlignedElement = Tuple[float, float, torch.Tensor]

class CombinedIterator:
    def __init__(self, iter1: AlignedIterator, iter2: AlignedIterator):
        """
        Initialize the CombinedIterator.

        Args:
            iter1: The first AlignedIterator.
            iter2: The second AlignedIterator.
        """
        self.aligned_iter1 = iter1
        self.aligned_iter2 = iter2
        self.iter1_iter: Optional[Iterator[AlignedElement]] = None
        self.iter2_iter: Optional[Iterator[AlignedElement]] = None
        self.current1: Optional[AlignedElement] = None
        self.current2: Optional[AlignedElement] = None

    def __iter__(self) -> 'CombinedIterator':
        """
        Initialize the internal iterators and fetch the first elements.
        Returns self as the iterator object.
        """
        self.iter1_iter = iter(self.aligned_iter1)
        self.iter2_iter = iter(self.aligned_iter2)

        # Fetch the first element from each iterator, handle empty iterators
        try:
            self.current1 = next(self.iter1_iter)
        except StopIteration:
            self.current1 = None # Mark as exhausted
        try:
            self.current2 = next(self.iter2_iter)
        except StopIteration:
            self.current2 = None # Mark as exhausted

        return self

    def __next__(self) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        """
        Yields the next aligned time interval and corresponding data elements.

        Raises:
            StopIteration: When both input iterators are exhausted.

        Returns:
            Tuple[float, float, torch.Tensor, torch.Tensor]:
                (start_time, end_time, data_element1, data_element2)
        """
        while True: # Loop until a yieldable segment is found or iterators end
            # Check if either iterator is exhausted
            if self.current1 is None or self.current2 is None:
                raise StopIteration("One or both iterators are exhausted")

            # Unpack current elements
            s1, e1, d1 = self.current1
            s2, e2, d2 = self.current2

            # Determine the start and end of the overlapping interval
            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)

            # --- Yielding Logic ---
            # Only yield if there is a valid time duration (start < end)
            if overlap_start < overlap_end:
                # Prepare the result for this interval
                result = (overlap_start, overlap_end, d1, d2)

                # --- Advancing Logic ---
                # Advance the iterator(s) whose current interval ends at overlap_end
                # Use temporary variables to track if next() should be called
                advance1 = (e1 == overlap_end)
                advance2 = (e2 == overlap_end)

                if advance1:
                    try:
                        self.current1 = next(self.iter1_iter)
                    except StopIteration:
                        self.current1 = None # Mark as exhausted
                if advance2:
                    try:
                        self.current2 = next(self.iter2_iter)
                    except StopIteration:
                        self.current2 = None # Mark as exhausted

                return result # Yield the aligned data for the calculated interval

            # --- Non-Overlapping or Zero-Duration Case ---
            # If overlap_start >= overlap_end, there's no overlap duration *in this step*
            # We need to advance the iterator that finishes *earlier* to find the next potential overlap.
            else:
                 # Advance the iterator that finishes first (or both if they finish at the same time)
                advance1 = (e1 <= e2) # Advance iter1 if it ends sooner or at the same time
                advance2 = (e2 <= e1) # Advance iter2 if it ends sooner or at the same time

                if advance1:
                    try:
                        self.current1 = next(self.iter1_iter)
                    except StopIteration:
                        self.current1 = None
                        raise StopIteration("Iterator 1 exhausted while resolving non-overlap") # Raise here as we can't proceed

                if advance2:
                     # Avoid double advance if e1 == e2 and iter1 already advanced
                    if self.current1 is None and e1 == e2:
                         raise StopIteration("Both iterators exhausted simultaneously")

                    try:
                        # Only advance iter2 if it wasn't already handled by e1==e2 and iter1 advancing
                        if not (advance1 and e1 == e2):
                           self.current2 = next(self.iter2_iter)
                    except StopIteration:
                        self.current2 = None
                        raise StopIteration("Iterator 2 exhausted while resolving non-overlap") # Raise here as we can't proceed

                # After advancing, loop again (`continue` is implicit) to re-evaluate with the new elements


if __name__ == "__main__":
    data1 = [5.5, 6.5]
    map1 = {
        0: (0.0, 10.0),
        1: (10.0, 25.0)
    }
    aligned_iter1 = AlignedIterator(data1, map1)

    data2 = [100, 200, 300, 400, 500]
    map2 = {
        0: (0.0, 3.0),
        1: (3.0, 8.0),
        2: (8.0, 15.0),
        3: (15.0, 17.0),
        4: (17.0, 25.0)
    }
    aligned_iter2 = AlignedIterator(data2, map2)

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
            print(f"({start:.1f}, {end:.1f}, {elem1.item()}, {elem2.item()})")
    except StopIteration as e:
        print(f"Iteration finished: {e}")

    print("\n--- Example with slight non-alignment at start ---")
    data3 = [5.5]
    map3 = {0: (1.0, 5.0)}
    aligned_iter3 = AlignedIterator(data3, map3)

    data4 = [99]
    map4 = {0: (0.0, 5.0)}
    aligned_iter4 = AlignedIterator(data4, map4)

    combined_iter_2 = CombinedIterator(aligned_iter3, aligned_iter4)
    try:
        for start, end, elem1, elem2 in combined_iter_2:
            print(f"({start:.1f}, {end:.1f}, {elem1.item()}, {elem2.item()})")
    except StopIteration as e:
         print(f"Iteration finished: {e}")

    print("\n--- Example where one iterator is shorter ---")
    data5 = [5.5, 6.5]
    map5 = {0: (0.0, 10.0), 1: (10.0, 20.0)}
    aligned_iter5 = AlignedIterator(data5, map5)

    data6 = [-30]
    map6 = {0: (0.0, 5.0)}
    aligned_iter6 = AlignedIterator(data6, map6)

    combined_iter_3 = CombinedIterator(aligned_iter5, aligned_iter6)
    try:
        for start, end, elem1, elem2 in combined_iter_3:
            print(f"({start:.1f}, {end:.1f}, {elem1.item()}, {elem2.item()})")
    except StopIteration as e:
         print(f"Iteration finished: {e}") # Expected: Iteration stops after (0.0, 5.0)