import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Tuple

class MultiBucketCycleSampler:
    """
    Build a global 'deck' of all rays across all buckets (one bucket = one view of one frame).
    A cycle = one pass through the whole deck without replacement.
    Each call returns one batch as a mapping: {bucket_id: np.ndarray[sel_indices]}.
    That batch typically mixes several frame ids and several camera views.
    """
    def __init__(self, bucket_lengths: List[int], batch_size: int, 
                 shuffle_across_buckets: bool = False, shuffle_within_bucket: bool = False):
        self.bucket_lengths = list(bucket_lengths)   # len B
        self.batch_size     = int(batch_size)
        self.shuffle_across_buckets = shuffle_across_buckets
        self.shuffle_within_bucket = shuffle_within_bucket
        self._build_new_deck()

    def _build_new_deck(self):
        # every bucket (b) means every (frame_id, view) pair
        deck: List[Tuple[int, int]] = []
        for b, L in enumerate(self.bucket_lengths): # b: frame-view id; L: num of rays in that bucket
            if L <= 0:
                continue
            if self.shuffle_within_bucket:
                perm = np.random.permutation(L)
                deck.extend((b, int(i)) for i in perm)
            else:
                deck.extend((b, int(i)) for i in range(L))
        if self.shuffle_across_buckets:
            random.shuffle(deck) 
        self.deck = deck
        self.ptr  = 0

    def __call__(self) -> Tuple[List[int], List[np.ndarray]]:
        """
        Return one batch:
          - bucket_ids: list[int]
          - index_lists: list[np.ndarray] (same length as bucket_ids)
        If the remaining deck cannot fill a whole batch, start a new cycle.
        (So no cross-cycle mixing; no duplicates within cycle.)
        """
        remain = len(self.deck) - self.ptr
        if remain < self.batch_size:
            # start new cycle
            self._build_new_deck()

        start = self.ptr
        end   = self.ptr + self.batch_size
        batch_pairs = self.deck[start:end]
        self.ptr = end

        # Group by bucket
        by_bucket: Dict[int, List[int]] = defaultdict(list)
        for b, i in batch_pairs:
            by_bucket[b].append(i)

        bucket_ids   = list(by_bucket.keys())
        index_lists  = [np.asarray(by_bucket[b], dtype=np.int64) for b in bucket_ids]
        return bucket_ids, index_lists
