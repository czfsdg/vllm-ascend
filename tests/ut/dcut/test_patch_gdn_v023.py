# SPDX-License-Identifier: Apache-2.0

import torch

from dcut.patch_gdn_v023 import (
    _compact_spec_state_indices,
    _rebase_spec_gdn_states,
    _run_padded_spec_causal_conv1d,
)


def test_compact_spec_state_indices_for_variable_query_lengths():
    state_indices = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    num_accepted_tokens = torch.tensor([2, 2], dtype=torch.int32)

    compact, accepted = _compact_spec_state_indices(
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        num_spec_decodes=2,
    )

    assert compact.tolist() == [10, 11, 20, 21, 22]
    assert accepted.tolist() == [2, 2]


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
    num_accepted_tokens = torch.tensor([4, 1, 1], dtype=torch.int32)

    compact, accepted = _compact_spec_state_indices(
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        num_spec_decodes=2,
    )

    assert compact.tolist() == [10, 11, 12, 13, 20]
    assert accepted.tolist() == [4, 1]


def test_compact_spec_state_indices_clamps_selector_to_live_row():
    state_indices = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    num_accepted_tokens = torch.tensor([4], dtype=torch.int32)

    _, accepted = _compact_spec_state_indices(
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        num_spec_decodes=1,
    )

    assert accepted.tolist() == [2]


def test_rebase_spec_gdn_states_commits_accepted_slot():
    conv_state = torch.arange(6 * 5 * 2, dtype=torch.float32).reshape(6, 5, 2)
    recurrent_state = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3)
    original_conv_state = conv_state.clone()
    original_recurrent_state = recurrent_state.clone()
    state_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 3, 2],
        ],
        dtype=torch.int32,
    )
    num_accepted_tokens = torch.tensor([3, 2], dtype=torch.int32)

    normalized = _rebase_spec_gdn_states(
        conv_state,
        recurrent_state,
        state_indices,
        num_accepted_tokens,
        num_spec_decodes=2,
    )

    assert normalized.tolist() == [1, 1]
    assert torch.equal(recurrent_state[0], original_recurrent_state[2])
    assert torch.equal(recurrent_state[4], original_recurrent_state[5])
    assert torch.equal(conv_state[0, :3], original_conv_state[0, 2:])
    assert torch.equal(conv_state[4, :4], original_conv_state[4, 1:])


def test_variable_spec_conv1d_is_padded_and_gathered():
    calls = []

    def fake_run(
        output,
        x,
        weight,
        conv_state,
        bias,
        query_start_loc,
        cache_indices,
        num_accepted_tokens,
        activation_mode,
    ):
        calls.append((x.clone(), query_start_loc.clone()))
        output.copy_(x + 100)

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    output = torch.empty_like(x)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    cache_indices = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        dtype=torch.int32,
    )

    _run_padded_spec_causal_conv1d(
        fake_run,
        output,
        x,
        torch.empty(0),
        torch.empty(0),
        None,
        query_start_loc,
        cache_indices,
        torch.ones(2, dtype=torch.int32),
        1,
    )

    padded_x, padded_query_start_loc = calls[0]
    assert padded_x.squeeze(1).tolist() == [1, 2, 0, 0, 3, 4, 5, 0]
    assert padded_query_start_loc.tolist() == [0, 4, 8]
    assert output.squeeze(1).tolist() == [101, 102, 103, 104, 105]
