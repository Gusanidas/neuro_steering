from time_interval_iterator.time_interval_iterator import TimeIntervalIterator
from time_interval_iterator.combined_average_iterator import CombinedAverageIterator

if __name__ == "__main__":
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
    combined_iter = CombinedAverageIterator(aligned_iter1, aligned_iter2)
    try:
        for start, end, (elem1, elem2) in combined_iter:
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

    combined_iter_2 = CombinedAverageIterator(aligned_iter3, aligned_iter4)
    try:
        for start, end, (elem1, elem2) in combined_iter_2:
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

    combined_iter_3 = CombinedAverageIterator(aligned_iter5, aligned_iter6)
    try:
        for start, end, (elem1, elem2) in combined_iter_3:
            print(f"({start:.1f}, {end:.1f}, {elem1}, {elem2})")
    except StopIteration as e:
         print(f"Iteration finished: {e}") # Expected: Iteration stops after (0.0, 5.0)