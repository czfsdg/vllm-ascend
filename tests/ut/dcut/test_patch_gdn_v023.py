# SPDX-License-Identifier: Apache-2.0

import torch

from dcut.patch_gdn_v023 import _compact_spec_state_indices


def test_compact_spec_state_indices_for_variable_query_lengths():
    state_indices = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    num_accepted_tokens = torch.tensor([4, 2], dtype=torch.int32)

    compact, clamped = _compact_spec_state_indices(
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        num_spec_decodes=2,
    )

    assert compact.tolist() == [10, 11, 20, 21, 22]
    assert clamped.tolist() == [2, 2]


def test_compact_spec_state_indices_ignores_padded_requests():
    state_indices = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [-1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor([0, 4, 5, 5], dtype=torch.int32)
    num_accepted_tokens = torch.tensor([4, 3, 1], dtype=torch.int32)

    compact, clamped = _compact_spec_state_indices(
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        num_spec_decodes=2,
    )

    assert compact.tolist() == [10, 11, 12, 13, 20]
    assert clamped.tolist() == [4, 1]
