# SPDX-License-Identifier: Apache-2.0
"""Target-NPU validation for the graph-safe D-Cut GDN operators.

This script deliberately tests two different properties:

1. Eager parity with the original vLLM Ascend operators, including the
   in-place convolution and recurrent states.
2. Repeated ACL Graph replay with changed tensor contents, including cases
   where the previous accepted position is longer than the current sequence.

Run this file directly on an Ascend host after building and installing the
D-Cut AscendC package and Torch registration library.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

BF16_RTOL = 1e-2
BF16_ATOL = 1e-2
PAD_SLOT_ID = -1
CONV_WIDTH = 4
QUERY_LENGTH_VARIANTS = (
    (4, 2, 3),
    (3, 4, 2),
    (2, 3, 4),
)


@dataclass
class CausalCase:
    x: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    state: torch.Tensor
    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    state_offsets: torch.Tensor


@dataclass
class RecurrentCase:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    beta: torch.Tensor
    state: torch.Tensor
    actual_seq_lengths: torch.Tensor
    original_state_indices: torch.Tensor
    dcut_state_indices: torch.Tensor
    num_accepted_tokens: torch.Tensor
    g: torch.Tensor
    scale: float


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _to_device(
    tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return tensor.to(device=device)


def _query_start_loc(query_lengths: tuple[int, ...]) -> torch.Tensor:
    lengths = torch.tensor((0, *query_lengths), dtype=torch.int32)
    return torch.cumsum(lengths, dim=0)


def _make_causal_case(
    seed: int,
    device: torch.device,
    dim: int,
    *,
    previous_round_longer: bool,
) -> CausalCase:
    generator = _generator(seed)
    variant = QUERY_LENGTH_VARIANTS[seed % len(QUERY_LENGTH_VARIANTS)]
    num_requests = len(variant)
    total_tokens = sum(variant)
    state_len = 8

    x = torch.randn(
        (total_tokens, dim),
        generator=generator,
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        (CONV_WIDTH, dim),
        generator=generator,
        dtype=torch.bfloat16,
    )
    bias = torch.randn(
        (dim,),
        generator=generator,
        dtype=torch.bfloat16,
    )
    state = torch.randn(
        (num_requests + 2, state_len, dim),
        generator=generator,
        dtype=torch.bfloat16,
    )
    cache_indices = torch.randperm(
        num_requests + 2,
        generator=generator,
        dtype=torch.int32,
    )[:num_requests]

    if previous_round_longer:
        max_offset = state_len - (CONV_WIDTH - 1)
        offsets = (
            max_offset,
            max_offset - 1,
            max_offset - 2,
        )
    else:
        # offset + 1 must fit the current sequence for the original operator.
        offsets = tuple(max(0, length - 2) for length in variant)

    return CausalCase(
        x=_to_device(x, device),
        weight=_to_device(weight, device),
        bias=_to_device(bias, device),
        state=_to_device(state, device),
        query_start_loc=_to_device(_query_start_loc(variant), device),
        cache_indices=_to_device(cache_indices, device),
        state_offsets=_to_device(
            torch.tensor(offsets, dtype=torch.int32),
            device,
        ),
    )


def _make_recurrent_case(
    seed: int,
    device: torch.device,
    state_dtype: torch.dtype,
    *,
    previous_round_longer: bool,
) -> RecurrentCase:
    generator = _generator(seed)
    query_lengths = QUERY_LENGTH_VARIANTS[
        seed % len(QUERY_LENGTH_VARIANTS)
    ]
    batch_size = len(query_lengths)
    state_index_stride = max(max(query_lengths), 4)
    total_tokens = sum(query_lengths)
    num_key_heads = 2
    num_value_heads = 4
    key_dim = 128
    value_dim = 128
    num_state_slots = batch_size * state_index_stride

    query = torch.nn.functional.normalize(
        torch.rand(
            (total_tokens, num_key_heads, key_dim),
            generator=generator,
        ),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    key = torch.nn.functional.normalize(
        torch.rand(
            (total_tokens, num_key_heads, key_dim),
            generator=generator,
        ),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    value = torch.rand(
        (total_tokens, num_value_heads, value_dim),
        generator=generator,
        dtype=torch.bfloat16,
    )
    beta = torch.rand(
        (total_tokens, num_value_heads),
        generator=generator,
        dtype=torch.bfloat16,
    )
    g = (
        torch.rand(
            (total_tokens, num_value_heads),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.2
        - 0.1
    )
    state = torch.rand(
        (
            num_state_slots,
            num_value_heads,
            value_dim,
            key_dim,
        ),
        generator=generator,
        dtype=torch.float32,
    ).to(state_dtype)

    fixed_rows = torch.randperm(
        num_state_slots,
        generator=generator,
        dtype=torch.int32,
    ).reshape(batch_size, state_index_stride)
    flattened_valid = torch.cat(
        [
            fixed_rows[batch_index, :query_length]
            for batch_index, query_length in enumerate(query_lengths)
        ]
    )
    if previous_round_longer:
        accepted = torch.tensor(
            (
                state_index_stride,
                state_index_stride - 1,
                state_index_stride,
            ),
            dtype=torch.int32,
        )
    else:
        accepted = torch.tensor(
            tuple(max(1, length - 1) for length in query_lengths),
            dtype=torch.int32,
        )

    actual_seq_lengths = torch.tensor(
        (0, *query_lengths),
        dtype=torch.int32,
    )
    return RecurrentCase(
        query=_to_device(query, device),
        key=_to_device(key, device),
        value=_to_device(value, device),
        beta=_to_device(beta, device),
        state=_to_device(state, device),
        actual_seq_lengths=_to_device(actual_seq_lengths, device),
        original_state_indices=_to_device(flattened_valid, device),
        dcut_state_indices=_to_device(fixed_rows, device),
        num_accepted_tokens=_to_device(accepted, device),
        g=_to_device(g, device),
        scale=key_dim**-0.5,
    )


def _metric(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    results: list[dict[str, Any]],
) -> None:
    actual_cpu = actual.detach().to(torch.float32).cpu()
    expected_cpu = expected.detach().to(torch.float32).cpu()
    if not torch.isfinite(actual_cpu).all():
        raise AssertionError(f"{name}: actual contains non-finite values")
    if not torch.isfinite(expected_cpu).all():
        raise AssertionError(f"{name}: expected contains non-finite values")
    delta = (actual_cpu - expected_cpu).abs()
    max_abs = float(delta.max().item())
    max_rel = float(
        (delta / expected_cpu.abs().clamp_min(1e-6)).max().item()
    )
    torch.testing.assert_close(
        actual_cpu,
        expected_cpu,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
        equal_nan=True,
    )
    result = {
        "name": name,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "rtol": BF16_RTOL,
        "atol": BF16_ATOL,
    }
    results.append(result)
    print(
        f"PASS {name}: max_abs={max_abs:.6e} "
        f"max_rel={max_rel:.6e}"
    )


def _run_original_causal(
    case: CausalCase,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(case.x)
    state = case.state.clone()
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        output=output,
        x=case.x,
        weight=case.weight,
        conv_state=state,
        bias_opt=case.bias,
        query_start_loc_opt=case.query_start_loc,
        cache_indices_opt=case.cache_indices,
        initial_state_mode_opt=None,
        num_accepted_tokens_opt=case.state_offsets + 1,
        activation_mode=1,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=1,
    )
    return output, state


def _run_dcut_causal(
    case: CausalCase,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(case.x)
    state = case.state.clone()
    torch.ops._C_ascend.npu_dcut_causal_conv1d(
        output=output,
        x=case.x,
        weight=case.weight,
        conv_state=state,
        bias=case.bias,
        query_start_loc=case.query_start_loc,
        cache_indices=case.cache_indices,
        state_offsets=case.state_offsets,
        activation_mode=1,
        pad_slot_id=PAD_SLOT_ID,
    )
    return output, state


def _run_original_recurrent(
    case: RecurrentCase,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = case.state.clone()
    output = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=case.query,
        key=case.key,
        value=case.value,
        state=state,
        beta=case.beta,
        scale=case.scale,
        actual_seq_lengths=case.actual_seq_lengths,
        ssm_state_indices=case.original_state_indices,
        num_accepted_tokens=case.num_accepted_tokens,
        g=case.g,
        gk=None,
    )
    return output, state


def _run_dcut_recurrent(
    case: RecurrentCase,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = case.state.clone()
    output = torch.ops._C_ascend.npu_dcut_recurrent_gated_delta_rule(
        query=case.query,
        key=case.key,
        value=case.value,
        state=state,
        beta=case.beta,
        scale=case.scale,
        actual_seq_lengths=case.actual_seq_lengths,
        ssm_state_indices=case.dcut_state_indices,
        num_accepted_tokens=case.num_accepted_tokens,
        g=case.g,
        gk=None,
    )
    return output, state


def _validate_eager_parity(
    device: torch.device,
    causal_dim: int,
    results: list[dict[str, Any]],
) -> None:
    print("== eager parity: original operators vs D-Cut operators ==")
    for seed in (17, 29, 41):
        case = _make_causal_case(
            seed,
            device,
            causal_dim,
            previous_round_longer=False,
        )
        original_output, original_state = _run_original_causal(case)
        dcut_output, dcut_state = _run_dcut_causal(case)
        torch.npu.synchronize()
        _metric(
            f"eager/causal/output/seed={seed}",
            dcut_output,
            original_output,
            results,
        )
        _metric(
            f"eager/causal/state/seed={seed}",
            dcut_state,
            original_state,
            results,
        )

    for state_dtype in (torch.bfloat16, torch.float32):
        dtype_name = str(state_dtype).removeprefix("torch.")
        for seed in (53, 67, 79):
            case = _make_recurrent_case(
                seed,
                device,
                state_dtype,
                previous_round_longer=False,
            )
            original_output, original_state = _run_original_recurrent(case)
            dcut_output, dcut_state = _run_dcut_recurrent(case)
            torch.npu.synchronize()
            _metric(
                f"eager/recurrent/{dtype_name}/output/seed={seed}",
                dcut_output,
                original_output,
                results,
            )
            _metric(
                f"eager/recurrent/{dtype_name}/state/seed={seed}",
                dcut_state,
                original_state,
                results,
            )


def _copy_causal_case(
    destination: CausalCase,
    source: CausalCase,
) -> None:
    destination.x.copy_(source.x)
    destination.weight.copy_(source.weight)
    destination.bias.copy_(source.bias)
    destination.state.copy_(source.state)
    destination.query_start_loc.copy_(source.query_start_loc)
    destination.cache_indices.copy_(source.cache_indices)
    destination.state_offsets.copy_(source.state_offsets)


def _copy_recurrent_case(
    destination: RecurrentCase,
    source: RecurrentCase,
) -> None:
    destination.query.copy_(source.query)
    destination.key.copy_(source.key)
    destination.value.copy_(source.value)
    destination.beta.copy_(source.beta)
    destination.state.copy_(source.state)
    destination.actual_seq_lengths.copy_(source.actual_seq_lengths)
    destination.original_state_indices.copy_(source.original_state_indices)
    destination.dcut_state_indices.copy_(source.dcut_state_indices)
    destination.num_accepted_tokens.copy_(source.num_accepted_tokens)
    destination.g.copy_(source.g)


def _validate_causal_graph(
    device: torch.device,
    causal_dim: int,
    replays: int,
    results: list[dict[str, Any]],
) -> None:
    static_case = _make_causal_case(
        101,
        device,
        causal_dim,
        previous_round_longer=True,
    )
    _run_dcut_causal(static_case)
    torch.npu.synchronize()
    graph_output = torch.empty_like(static_case.x)
    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(
        graph,
        capture_error_mode="thread_local",
        auto_dispatch_capture=True,
    ):
        torch.ops._C_ascend.npu_dcut_causal_conv1d(
            output=graph_output,
            x=static_case.x,
            weight=static_case.weight,
            conv_state=static_case.state,
            bias=static_case.bias,
            query_start_loc=static_case.query_start_loc,
            cache_indices=static_case.cache_indices,
            state_offsets=static_case.state_offsets,
            activation_mode=1,
            pad_slot_id=PAD_SLOT_ID,
        )
    torch.npu.synchronize()

    for replay_index in range(replays):
        case = _make_causal_case(
            131 + replay_index * 17,
            device,
            causal_dim,
            previous_round_longer=True,
        )
        expected_output, expected_state = _run_dcut_causal(case)
        _copy_causal_case(static_case, case)
        graph_output.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        _metric(
            f"graph/causal/output/replay={replay_index}",
            graph_output,
            expected_output,
            results,
        )
        _metric(
            f"graph/causal/state/replay={replay_index}",
            static_case.state,
            expected_state,
            results,
        )


def _validate_recurrent_graph(
    device: torch.device,
    replays: int,
    results: list[dict[str, Any]],
) -> None:
    static_case = _make_recurrent_case(
        211,
        device,
        torch.float32,
        previous_round_longer=True,
    )
    _run_dcut_recurrent(static_case)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(
        graph,
        capture_error_mode="thread_local",
        auto_dispatch_capture=True,
    ):
        graph_output = (
            torch.ops._C_ascend.npu_dcut_recurrent_gated_delta_rule(
                query=static_case.query,
                key=static_case.key,
                value=static_case.value,
                state=static_case.state,
                beta=static_case.beta,
                scale=static_case.scale,
                actual_seq_lengths=static_case.actual_seq_lengths,
                ssm_state_indices=static_case.dcut_state_indices,
                num_accepted_tokens=static_case.num_accepted_tokens,
                g=static_case.g,
                gk=None,
            )
        )
    torch.npu.synchronize()

    for replay_index in range(replays):
        case = _make_recurrent_case(
            239 + replay_index * 19,
            device,
            torch.float32,
            previous_round_longer=True,
        )
        expected_output, expected_state = _run_dcut_recurrent(case)
        _copy_recurrent_case(static_case, case)
        graph_output.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        _metric(
            f"graph/recurrent/output/replay={replay_index}",
            graph_output,
            expected_output,
            results,
        )
        _metric(
            f"graph/recurrent/state/replay={replay_index}",
            static_case.state,
            expected_state,
            results,
        )


def _validate_graph_replay(
    device: torch.device,
    causal_dim: int,
    replays: int,
    results: list[dict[str, Any]],
) -> None:
    print("== ACL Graph replay: changed inputs and previous-round offsets ==")
    _validate_causal_graph(device, causal_dim, replays, results)
    _validate_recurrent_graph(device, replays, results)


def _default_library() -> Path:
    configured = os.getenv("VLLM_DCUT_TORCH_OP_LIBRARY")
    if configured:
        return Path(configured)
    return (
        Path(__file__).resolve().parents[1]
        / "kernel"
        / "build"
        / "torch_extension"
        / "dcut_torch_ops.so"
    )


def _prepare_runtime(
    device_name: str,
    library: Path,
) -> torch.device:
    import torch_npu
    from vllm_ascend.utils import enable_custom_op

    if not library.is_file():
        raise FileNotFoundError(
            f"D-Cut Torch registration library does not exist: {library}"
        )
    custom_opp_path = os.getenv("ASCEND_CUSTOM_OPP_PATH", "")
    if "dcut_transformer" not in custom_opp_path:
        raise RuntimeError(
            "ASCEND_CUSTOM_OPP_PATH must contain the installed "
            "dcut_transformer vendor root"
        )

    torch_npu.npu.set_compile_mode(jit_compile=False)
    device = torch.device(device_name)
    torch.npu.set_device(device)
    enable_custom_op()
    torch.ops.load_library(str(library.resolve()))

    required_ops = (
        "npu_causal_conv1d_custom",
        "npu_recurrent_gated_delta_rule",
        "npu_dcut_causal_conv1d",
        "npu_dcut_recurrent_gated_delta_rule",
    )
    missing = [
        name
        for name in required_ops
        if not hasattr(torch.ops._C_ascend, name)
    ]
    if missing:
        raise RuntimeError(f"Missing Torch operators: {missing}")
    print(f"device={device}")
    print(f"D-Cut Torch library={library.resolve()}")
    return device


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--library", type=Path, default=_default_library())
    parser.add_argument(
        "--mode",
        choices=("all", "eager", "graph"),
        default="all",
    )
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--causal-dim", type=int, default=256)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.replays < 2:
        parser.error("--replays must be at least 2")
    if args.causal_dim <= 0 or args.causal_dim % 32:
        parser.error("--causal-dim must be a positive multiple of 32")
    return args


def main() -> None:
    args = _parse_args()
    device = _prepare_runtime(args.device, args.library)
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        if args.mode in ("all", "eager"):
            _validate_eager_parity(device, args.causal_dim, results)
        if args.mode in ("all", "graph"):
            _validate_graph_replay(
                device,
                args.causal_dim,
                args.replays,
                results,
            )

    report = {
        "status": "pass",
        "device": str(device),
        "mode": args.mode,
        "replays": args.replays,
        "metrics": results,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"JSON report={args.json_out.resolve()}")
    print(f"DCUT_NPU_VALIDATION_PASS metrics={len(results)}")


if __name__ == "__main__":
    main()
