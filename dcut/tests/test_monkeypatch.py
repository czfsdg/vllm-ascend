# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


def _install_fake_vllm_logger():
    vllm_module = types.ModuleType("vllm")
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: _Logger()
    sys.modules.setdefault("vllm", vllm_module)
    sys.modules["vllm.logger"] = logger_module


def test_proposer_patch_does_not_replace_logits_processor_child_module(monkeypatch):
    _install_fake_vllm_logger()
    dcut_monkeypatch = importlib.import_module("dcut.monkeypatch")

    class FakeDraftTokenIds:
        def reshape(self, *args):
            return self

    class FakeLogitsProcessor:
        def __init__(self):
            self.called = False

        def forward(self, *args, **kwargs):
            self.called = True
            return "logits"

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class FakeModel:
        def __init__(self):
            self.logits_processor = FakeLogitsProcessor()
            self.lm_head = object()

        def __setattr__(self, name, value):
            if name == "logits_processor" and hasattr(self, "logits_processor"):
                msg = "torch.nn.Module child replacement should not happen"
                raise TypeError(msg)
            super().__setattr__(name, value)

    class FakeProposer:
        def __init__(self):
            self.runner = object()
            self.model = FakeModel()

        def _run_merged_draft(self):
            self.model.logits_processor(self.model.lm_head, "hidden_states")
            return FakeDraftTokenIds()

    module = types.SimpleNamespace(AscendSpecDecodeBaseProposer=FakeProposer)
    assert dcut_monkeypatch._patch_proposer_module(module)

    recorded = {}
    monkeypatch.setattr(dcut_monkeypatch, "_ensure_runner_state", lambda runner: True)
    monkeypatch.setattr(
        dcut_monkeypatch,
        "_record_selected_token_probs",
        lambda proposer, logits, draft_token_ids: recorded.setdefault("logits", logits),
    )

    proposer = FakeProposer()
    proposer._run_merged_draft()

    assert proposer.model.logits_processor.called
    assert recorded["logits"] == "logits"
