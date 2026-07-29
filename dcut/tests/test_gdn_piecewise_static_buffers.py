# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from dcut.gdn_buffers import (
    _dcut_get_gdn_piecewise_spec_bufs,
    _dcut_prepare_gdn_piecewise_replay,
)
from dcut.globals import _dcut_gdn_static
from dcut.verify_adaptive_controller import VerifyAdaptiveController


class _GDNMetadata:
    pass


def _make_metadata(
    query_start_loc: list[int],
    state_indices: list[list[int]],
    accepted_tokens: list[int],
) -> _GDNMetadata:
    meta = _GDNMetadata()
    meta.num_spec_decodes = len(state_indices)
    meta.num_prefills = 0
    meta.num_decodes = 0
    meta.spec_sequence_masks = torch.ones(
        len(state_indices), dtype=torch.bool
    )
    meta.spec_state_indices_tensor = torch.tensor(
        state_indices, dtype=torch.int32
    )
    conv_meta = SimpleNamespace(
        query_start_loc=torch.tensor(
            query_start_loc, dtype=torch.int32
        ),
        num_accepted_tokens=torch.tensor(
            accepted_tokens, dtype=torch.int64
        ),
    )
    meta.spec_decode_metadata = SimpleNamespace(
        spec_causal_conv1d=conv_meta
    )
    return meta


def test_piecewise_spec_buffers_keep_addresses_and_refresh_values() -> None:
    _dcut_gdn_static.clear()
    meta = _make_metadata(
        [0, 2, 3],
        [[10, 11, 12], [20, 21, 22]],
        [2, 3],
    )
    context = SimpleNamespace(
        model_instance=object(),
        attn_metadata={"layers.0.mixer": meta},
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )
    bufs = _dcut_get_gdn_piecewise_spec_bufs(
        context, "layers.0.mixer", 8
    )
    pointers = {
        name: tensor.data_ptr()
        for name, tensor in bufs.items()
        if name != "token_index"
    }

    assert bufs["qsl"].tolist() == [0, 2, 3, 3, 3]
    assert bufs["asl"].tolist() == [0, 2, 1, 0, 0]
    assert bufs["nat"].tolist() == [2, 1, 0, 0]
    assert bufs["ssi"].tolist() == [
        [10, 11, 12],
        [20, 21, 22],
        [-1, -1, -1],
        [-1, -1, -1],
    ]
    assert bufs["token_mask"].tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]

    meta.spec_decode_metadata.spec_causal_conv1d.query_start_loc = (
        torch.tensor([0, 1, 3], dtype=torch.int32)
    )
    meta.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens = (
        torch.tensor([7, 1], dtype=torch.int64)
    )
    meta.spec_state_indices_tensor = torch.tensor(
        [[30, 31, 32], [40, 41, 42]], dtype=torch.int32
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )
    assert all(
        bufs[name].data_ptr() == pointer
        for name, pointer in pointers.items()
    )
    assert bufs["qsl"].tolist() == [0, 1, 3, 3, 3]
    assert bufs["asl"].tolist() == [0, 1, 2, 0, 0]
    assert bufs["nat"].tolist() == [1, 1, 0, 0]
    assert bufs["ssi"][:2].tolist() == [
        [30, 31, 32],
        [40, 41, 42],
    ]


def test_piecewise_replay_rejects_non_spec_and_mixed_batches() -> None:
    _dcut_gdn_static.clear()
    meta = _make_metadata([0, 1], [[10, 11, 12]], [1])
    context = SimpleNamespace(
        model_instance=object(),
        attn_metadata={"layers.0.mixer": meta},
    )

    meta.spec_sequence_masks = None
    assert not _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )

    meta.spec_sequence_masks = torch.ones(1, dtype=torch.bool)
    meta.num_prefills = 1
    assert not _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )


def test_empty_cost_table_keeps_full_length_without_index_error(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_DCUT_RANDOM_CUT", raising=False)
    controller = object.__new__(VerifyAdaptiveController)
    controller.config = SimpleNamespace(enabled=True)
    controller._sorted_bs = []
    controller._adaptive_draft_lens = {}

    controller.process_draft_output(
        selected_probs=torch.ones((1, 2), dtype=torch.float32),
        req_ids=["request-0"],
        active_draft_req_ids={"request-0"},
        batch_size=1,
    )

    assert controller._adaptive_draft_lens == {}
