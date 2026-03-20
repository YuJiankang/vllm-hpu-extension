###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Tests for vllm_hpu_extension.mamba utilities.

Covers the mamba cache management functions introduced to support reduced
mamba cache sizes for FLA (Flash Linear Attention) models such as Qwen3.
"""

import pytest

from vllm_hpu_extension.mamba import (
    compute_mamba_cache_bs,
    FindMambaIndexForPrefill,
    FindMambaIndexForDecode,
)


# ---------------------------------------------------------------------------
# compute_mamba_cache_bs
# ---------------------------------------------------------------------------

class TestComputeMambaCacheBS:
    def test_small_max_num_seqs_uses_minimum_of_8(self):
        # When max_num_seqs < 8 the minimum of 8 should be used
        assert compute_mamba_cache_bs(1) == 8 + 2
        assert compute_mamba_cache_bs(4) == 8 + 2
        assert compute_mamba_cache_bs(8) == 8 + 2

    def test_large_max_num_seqs_uses_actual_value(self):
        assert compute_mamba_cache_bs(16) == 16 + 2
        assert compute_mamba_cache_bs(64) == 64 + 2
        assert compute_mamba_cache_bs(256) == 256 + 2

    def test_result_always_exceeds_max_num_seqs(self):
        for n in [1, 8, 16, 64, 128]:
            assert compute_mamba_cache_bs(n) > n


# ---------------------------------------------------------------------------
# FindMambaIndexForPrefill
# ---------------------------------------------------------------------------

class TestFindMambaIndexForPrefill:
    def test_empty_dict_assigns_index_zero(self):
        mamba_dict = {}
        idx = FindMambaIndexForPrefill(mamba_dict, seq_id=42, max_concurrency=10)
        assert idx == 0
        assert mamba_dict[42] == 0

    def test_assigns_first_free_index(self):
        # Slots 0 and 1 already taken; expect slot 2
        mamba_dict = {10: 0, 11: 1}
        idx = FindMambaIndexForPrefill(mamba_dict, seq_id=99, max_concurrency=10)
        assert idx == 2
        assert mamba_dict[99] == 2

    def test_fills_gap_in_assigned_indices(self):
        # Slots 0 and 2 taken; slot 1 is free
        mamba_dict = {10: 0, 11: 2}
        idx = FindMambaIndexForPrefill(mamba_dict, seq_id=99, max_concurrency=10)
        assert idx == 1
        assert mamba_dict[99] == 1

    def test_multiple_prefills_use_distinct_indices(self):
        mamba_dict = {}
        max_concurrency = 5
        assigned = []
        for seq_id in range(max_concurrency):
            idx = FindMambaIndexForPrefill(mamba_dict, seq_id, max_concurrency)
            assigned.append(idx)
        assert sorted(assigned) == list(range(max_concurrency))

    def test_raises_when_all_slots_full(self):
        max_concurrency = 3
        mamba_dict = {0: 0, 1: 1, 2: 2}
        with pytest.raises(RuntimeError, match="No available mamba index"):
            FindMambaIndexForPrefill(mamba_dict, seq_id=99, max_concurrency=max_concurrency)

    def test_updates_dict_in_place(self):
        mamba_dict = {5: 0}
        original_id = id(mamba_dict)
        FindMambaIndexForPrefill(mamba_dict, seq_id=6, max_concurrency=10)
        assert id(mamba_dict) == original_id
        assert 6 in mamba_dict


# ---------------------------------------------------------------------------
# FindMambaIndexForDecode
# ---------------------------------------------------------------------------

class TestFindMambaIndexForDecode:
    def test_removes_sequences_not_in_running_queue(self):
        mamba_dict = {1: 0, 2: 1, 3: 2}
        running_queue_list = [1, 3]
        result = FindMambaIndexForDecode(mamba_dict, seq_list=[1, 3],
                                         running_queue_list=running_queue_list)
        assert 2 not in mamba_dict
        assert set(result) == {0, 2}

    def test_keeps_all_sequences_when_all_running(self):
        mamba_dict = {1: 0, 2: 1, 3: 2}
        running_queue_list = [1, 2, 3]
        result = FindMambaIndexForDecode(mamba_dict, seq_list=[1, 2, 3],
                                         running_queue_list=running_queue_list)
        assert set(result) == {0, 1, 2}
        assert len(mamba_dict) == 3

    def test_removes_all_when_running_queue_empty(self):
        mamba_dict = {1: 0, 2: 1}
        result = FindMambaIndexForDecode(mamba_dict, seq_list=[],
                                         running_queue_list=[])
        assert result == []
        assert mamba_dict == {}

    def test_preserves_prefill_seqs_present_in_running_queue(self):
        # seq 10 is a prefill (not in seq_list) but still in running_queue
        mamba_dict = {10: 0, 20: 1, 30: 2}
        seq_list = [20, 30]            # decode-only batch
        running_queue_list = [10, 20, 30]  # scheduler running queue (includes prefill)
        result = FindMambaIndexForDecode(mamba_dict, seq_list=seq_list,
                                         running_queue_list=running_queue_list)
        # seq 10 must NOT be evicted because it is in running_queue_list
        assert 10 in mamba_dict
        assert set(result) == {0, 1, 2}

    def test_evicts_seqs_absent_from_running_queue_even_if_in_seq_list(self):
        # Edge case: seq in seq_list but not in running_queue → should be evicted
        mamba_dict = {1: 0, 2: 1}
        result = FindMambaIndexForDecode(mamba_dict, seq_list=[1, 2],
                                         running_queue_list=[1])
        assert 2 not in mamba_dict
        assert result == [0]

    def test_modifies_dict_in_place(self):
        mamba_dict = {1: 0, 2: 1}
        original_id = id(mamba_dict)
        FindMambaIndexForDecode(mamba_dict, seq_list=[1], running_queue_list=[1])
        assert id(mamba_dict) == original_id

    def test_returns_list(self):
        mamba_dict = {1: 0}
        result = FindMambaIndexForDecode(mamba_dict, seq_list=[1],
                                         running_queue_list=[1])
        assert isinstance(result, list)
