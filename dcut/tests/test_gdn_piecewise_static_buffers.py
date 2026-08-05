# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from dcut.gdn_buffers import (  # noqa: E402
    _dcut_get_gdn_piecewise_spec_bufs,
    _dcut_prepare_gdn_eager_state,
    _dcut_prepare_gdn_piecewise_replay,
)
from dcut.globals import _dcut_gdn_static  # noqa: E402


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
        cache_indices=torch.arange(
            len(state_indices), dtype=torch.int32
        ),
    )
    meta.spec_decode_metadata = SimpleNamespace(
        spec_causal_conv1d=conv_meta,
        actual_seq_lengths=torch.tensor(
            [0]
            + [
                query_start_loc[index + 1] - query_start_loc[index]
                for index in range(len(state_indices))
            ],
            dtype=torch.int32,
        ),
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
    # NAT addresses the state selected by the previous verifier step.  The
    # second request accepted three tokens previously even though this step's
    # segment contains only one token, so it must not be clamped to ASL.
    assert bufs["nat"].tolist() == [2, 3, 0, 0]
    assert bufs["conv_state_offsets"].tolist() == [1, 2, 0, 0]
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
        torch.tensor([3, 1], dtype=torch.int64)
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
    assert bufs["nat"].tolist() == [3, 1, 0, 0]
    assert bufs["conv_state_offsets"].tolist() == [2, 0, 0, 0]
    assert bufs["ssi"][:2].tolist() == [
        [30, 31, 32],
        [40, 41, 42],
    ]


def test_piecewise_batch_buffers_are_shared_across_gdn_layers() -> None:
    _dcut_gdn_static.clear()
    first = _make_metadata(
        [0, 2, 3],
        [[10, 11, 12], [20, 21, 22]],
        [2, 3],
    )
    second = _make_metadata(
        [0, 2, 3],
        [[30, 31, 32], [40, 41, 42]],
        [2, 3],
    )
    context = SimpleNamespace(
        model_instance=object(),
        attn_metadata={
            "layers.0.mixer": first,
            "layers.1.mixer": second,
        },
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )
    first_bufs = _dcut_get_gdn_piecewise_spec_bufs(
        context, "layers.0.mixer", 8
    )
    second_bufs = _dcut_get_gdn_piecewise_spec_bufs(
        context, "layers.1.mixer", 8
    )

    for name in (
        "qsl",
        "asl",
        "nat",
        "token_index",
        "token_mask",
        "conv_state_offsets",
    ):
        assert first_bufs[name].data_ptr() == second_bufs[name].data_ptr()
    assert first_bufs["ssi"].data_ptr() != second_bufs["ssi"].data_ptr()
    assert first_bufs["ssi"][:2].tolist() == [
        [10, 11, 12],
        [20, 21, 22],
    ]
    assert second_bufs["ssi"][:2].tolist() == [
        [30, 31, 32],
        [40, 41, 42],
    ]

def test_eager_spec_state_is_prepared_once_per_forward() -> None:
    first = _make_metadata(
        [0, 2, 3],
        [[10, 11, 12], [20, 21, 22]],
        [2, 0],
    )
    second = _make_metadata(
        [0, 2, 3],
        [[30, 31, 32], [40, 41, 42]],
        [2, 0],
    )
    context = SimpleNamespace(
        attn_metadata={
            "layers.0.mixer": first,
            "layers.1.mixer": second,
        }
    )

    assert _dcut_prepare_gdn_eager_state(context, _GDNMetadata)
    state = context._dcut_gdn_eager_spec_state
    assert state["num_accepted_tokens"].dtype == torch.int32
    assert state["num_accepted_tokens"].tolist() == [2, 0]
    assert state["conv_state_offsets"].tolist() == [1, 0]
    assert (
        state["actual_seq_lengths"]
        is first.spec_decode_metadata.actual_seq_lengths
    )

    first.spec_sequence_masks = None
    second.spec_sequence_masks = None
    assert not _dcut_prepare_gdn_eager_state(
        context, _GDNMetadata
    )
    assert context._dcut_gdn_eager_spec_state is None


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

    meta.num_prefills = 0
    meta.num_decodes = 1
    assert not _dcut_prepare_gdn_piecewise_replay(
        context, 8, _GDNMetadata, 4
    )


def test_piecewise_buffers_do_not_alias_model_instances() -> None:
    _dcut_gdn_static.clear()
    meta = _make_metadata([0, 1], [[10, 11]], [1])
    first = SimpleNamespace(
        model_instance=object(),
        attn_metadata={"layers.0.mixer": meta},
    )
    second = SimpleNamespace(
        model_instance=object(),
        attn_metadata={"layers.0.mixer": meta},
    )

    assert _dcut_prepare_gdn_piecewise_replay(
        first, 4, _GDNMetadata, 2
    )
    assert _dcut_prepare_gdn_piecewise_replay(
        second, 4, _GDNMetadata, 2
    )
    first_bufs = _dcut_get_gdn_piecewise_spec_bufs(
        first, "layers.0.mixer", 4
    )
    second_bufs = _dcut_get_gdn_piecewise_spec_bufs(
        second, "layers.0.mixer", 4
    )

    assert first_bufs["qsl"].data_ptr() != second_bufs["qsl"].data_ptr()
