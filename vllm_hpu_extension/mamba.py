###############################################################################
# Copyright (C) 2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Mamba/FLA (Flash Linear Attention) cache management utilities for HPU.

These utilities manage the mapping between sequence IDs and mamba cache
slot indices, keeping cache usage minimal by tracking only active sequences.
"""

from typing import Dict, List


def compute_mamba_cache_bs(max_num_seqs: int) -> int:
    """Compute the mamba cache batch size.

    Uses the running queue size to determine the required cache size rather
    than over-allocating based on prefill and decode capacities separately.

    Args:
        max_num_seqs: Maximum number of sequences that can run concurrently.

    Returns:
        The mamba cache batch size.
    """
    return max(8, max_num_seqs) + 2


def FindMambaIndexForPrefill(
    mamba_dict: Dict[int, int],
    seq_id: int,
    max_concurrency: int,
) -> int:
    """Find and assign a mamba cache slot index for a new prefill sequence.

    Scans available indices sequentially and assigns the first free slot to
    the given sequence ID.

    Args:
        mamba_dict: Mapping from sequence ID to mamba cache slot index.
            Modified in-place to record the new assignment.
        seq_id: The sequence ID to assign a cache slot to.
        max_concurrency: Total number of available mamba cache slots.

    Returns:
        The assigned mamba cache slot index.

    Raises:
        RuntimeError: If no free slot is available within max_concurrency.
    """
    used_values = set(mamba_dict.values())
    for idx in range(max_concurrency):
        if idx not in used_values:
            mamba_dict[seq_id] = idx
            return idx

    raise RuntimeError(
        f"No available mamba index for seq_id={seq_id}, "
        f"max_concurrency={max_concurrency}, mamba_dict={mamba_dict}"
    )


def FindMambaIndexForDecode(
    mamba_dict: Dict[int, int],
    seq_list: List[int],
    running_queue_list: List[int],
) -> List[int]:
    """Update the mamba cache for a decode step and return active slot indices.

    Removes cache entries for sequences that are no longer in the scheduler's
    running queue, then returns the slot indices for all remaining sequences.

    Using ``running_queue_list`` (all sequences currently scheduled by the
    scheduler) rather than just the decode-batch ``seq_list`` ensures that
    prefill sequences whose entries should be preserved are not evicted
    prematurely.

    Args:
        mamba_dict: Mapping from sequence ID to mamba cache slot index.
            Modified in-place to remove stale entries.
        seq_list: Sequence IDs present in the current decode batch.
        running_queue_list: All sequence IDs currently in the scheduler's
            running queue (prefill + decode).

    Returns:
        List of mamba cache slot indices for the sequences that remain
        after eviction.
    """
    running_set = set(running_queue_list)
    invalid_keys = [
        key for key in mamba_dict
        if key not in running_set
    ]
    for key in invalid_keys:
        mamba_dict.pop(key)
    return list(mamba_dict.values())
