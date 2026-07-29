# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

import dcut.gdn_buffers as gdn_buffers
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


def test_piecewise_spec_buffers_are_filled_once_per_metadata_group(
    monkeypatch,
) -> None:
    _dcut_gdn_static.clear()
    shared_meta = _make_metadata(
        [0, 2, 4],
        [[10, 11, 12], [20, 21, 22]],
        [2, 1],
    )
    prefixes = (
        "layers.0.mixer",
        "layers.1.mixer",
        "layers.2.mixer",
    )
    context = SimpleNamespace(
        model_instance=object(),
        attn_metadata={
            prefix: shared_meta
            for prefix in prefixes
        },
    )
    fill_calls = []
    original_fill = (
        gdn_buffers._dcut_fill_gdn_piecewise_spec_bufs
    )

    def _counting_fill(
        forward_context,
        group_prefixes,
        num_tokens,
        meta,
        max_num_seqs,
    ):
        fill_calls.append(tuple(group_prefixes))
        return original_fill(
            forward_context,
            group_prefixes,
            num_tokens,
            meta,
            max_num_seqs,
        )

    monkeypatch.setattr(
        gdn_buffers,
        "_dcut_fill_gdn_piecewise_spec_bufs",
        _counting_fill,
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )
    assert fill_calls == [prefixes]

    buffers = [
        _dcut_get_gdn_piecewise_spec_bufs(
            context, prefix, 8
        )
        for prefix in prefixes
    ]
    assert all(bufs is buffers[0] for bufs in buffers[1:])
    assert buffers[0]["ssi"][:2].tolist() == [
        [10, 11, 12],
        [20, 21, 22],
    ]


def test_piecewise_spec_buffers_keep_metadata_groups_isolated() -> None:
    _dcut_gdn_static.clear()
    first_meta = _make_metadata(
        [0, 2],
        [[10, 11, 12]],
        [2],
    )
    second_meta = _make_metadata(
        [0, 3],
        [[20, 21, 22]],
        [1],
    )
    context = SimpleNamespace(
        model_instance=object(),
        attn_metadata={
            "layers.0.mixer": first_meta,
            "layers.8.mixer": second_meta,
        },
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )
    first_bufs = _dcut_get_gdn_piecewise_spec_bufs(
        context, "layers.0.mixer", 8
    )
    second_bufs = _dcut_get_gdn_piecewise_spec_bufs(
        context, "layers.8.mixer", 8
    )

    assert first_bufs is not second_bufs
    assert first_bufs["qsl"].tolist() == [0, 2, 2, 2, 2]
    assert second_bufs["qsl"].tolist() == [0, 3, 3, 3, 3]
    assert first_bufs["ssi"][0].tolist() == [10, 11, 12]
    assert second_bufs["ssi"][0].tolist() == [20, 21, 22]


def test_piecewise_spec_group_reuses_addresses_with_new_metadata() -> None:
    _dcut_gdn_static.clear()
    prefixes = ("layers.0.mixer", "layers.1.mixer")
    initial_meta = _make_metadata(
        [0, 2],
        [[10, 11, 12]],
        [2],
    )
    context = SimpleNamespace(
        model_instance=object(),
        attn_metadata={
            prefix: initial_meta
            for prefix in prefixes
        },
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )
    bufs = _dcut_get_gdn_piecewise_spec_bufs(
        context, prefixes[0], 8
    )
    pointers = {
        name: tensor.data_ptr()
        for name, tensor in bufs.items()
    }

    refreshed_meta = _make_metadata(
        [0, 1],
        [[30, 31, 32]],
        [1],
    )
    context.attn_metadata = {
        prefix: refreshed_meta
        for prefix in prefixes
    }

    assert _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )
    refreshed = _dcut_get_gdn_piecewise_spec_bufs(
        context, prefixes[1], 8
    )
    assert refreshed is bufs
    assert {
        name: tensor.data_ptr()
        for name, tensor in refreshed.items()
    } == pointers
    assert refreshed["qsl"].tolist() == [0, 1, 1, 1, 1]
    assert refreshed["ssi"][0].tolist() == [30, 31, 32]


def test_piecewise_replay_rejects_metadata_group_topology_change() -> None:
    _dcut_gdn_static.clear()
    prefixes = ("layers.0.mixer", "layers.1.mixer")
    shared_meta = _make_metadata(
        [0, 1],
        [[10, 11, 12]],
        [1],
    )
    context = SimpleNamespace(
        model_instance=object(),
        attn_metadata={
            prefix: shared_meta
            for prefix in prefixes
        },
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )

    context.attn_metadata = {
        prefixes[0]: _make_metadata(
            [0, 1], [[20, 21, 22]], [1]
        ),
        prefixes[1]: _make_metadata(
            [0, 1], [[30, 31, 32]], [1]
        ),
    }
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
