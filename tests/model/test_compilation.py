"""Compilation regression tests."""

import copy

import pytest
import torch
import torch._dynamo.config as dynamo_config

from morph.model.lightning._setup import SetupMixin


def test_compilation_preserves_both_live_models_and_checkpoint_keys(monkeypatch):
    class Harness(SetupMixin, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(3, 2)
            self.model_ema = copy.deepcopy(self.model).requires_grad_(False).eval()
            self.use_ema_for_eval = True
            self.is_compiled = False
            self.log_grad_norms_every_n_steps = 0

    monkeypatch.delenv("MORPH_COMPILE", raising=False)
    real_compile = torch.compile
    compiled_sources = []

    def compile_eager(fn, **kwargs):
        compiled_sources.append(fn.__self__)
        return real_compile(fn, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", compile_eager)
    module = Harness()
    current, ema = module.model, module.model_ema
    keys = set(module.state_dict())
    parameters = list(module.parameters())
    module.compile()
    current_call = current._compiled_call_impl
    ema_call = ema._compiled_call_impl
    module.compile()
    assert module.is_compiled
    assert module.model is current
    assert module.model_ema is ema
    assert current._compiled_call_impl is current_call
    assert ema._compiled_call_impl is ema_call
    assert compiled_sources == [current, ema]
    assert module._get_model() is ema
    assert set(module.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, module.parameters(), strict=True))
    x = torch.randn(4, 3)
    current(x).square().sum().backward()
    assert current.weight.grad is not None
    assert ema.weight.grad is None
    with torch.inference_mode():
        torch.testing.assert_close(ema(x), ema.forward(x))
        ema.weight.add_(2.0)
        torch.testing.assert_close(ema(x), ema.forward(x))
        assert not torch.allclose(ema(x), current(x))
    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    resumed = Harness()
    resumed.compile()
    resumed.on_load_checkpoint(checkpoint)
    resumed.load_state_dict(checkpoint["state_dict"], strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(resumed.model_ema(x), ema(x))
    module.use_ema_for_eval = False
    assert module._get_model() is current


@pytest.mark.parametrize("setting, enabled", [(None, True), ("1", True), ("0", False)])
def test_one_compilation_setting_controls_current_and_ema(
    monkeypatch, setting, enabled
):
    class Harness(SetupMixin, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(2, 2)
            self.model_ema = copy.deepcopy(self.model)
            self.is_compiled = False
            self.log_grad_norms_every_n_steps = 0

    if setting is None:
        monkeypatch.delenv("MORPH_COMPILE", raising=False)
    else:
        monkeypatch.setenv("MORPH_COMPILE", setting)
    calls = []

    def compile_spy(fn, **kwargs):
        calls.append(fn.__self__)
        return fn

    monkeypatch.setattr(torch, "compile", compile_spy)
    module = Harness()
    module.compile()
    assert module.is_compiled is enabled
    assert calls == ([module.model, module.model_ema] if enabled else [])
    assert (module.model._compiled_call_impl is not None) is enabled
    assert (module.model_ema._compiled_call_impl is not None) is enabled


def test_compiler_failure_falls_back_without_changing_gradients(monkeypatch):
    class Harness(SetupMixin, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(3, 2)
            self.model_ema = copy.deepcopy(self.model).requires_grad_(False).eval()
            self.is_compiled = False
            self.log_grad_norms_every_n_steps = 0

    monkeypatch.setenv("MORPH_COMPILE", "1")
    monkeypatch.setenv("MORPH_COMPILE_DDP_OPTIMIZER", "0")
    monkeypatch.setenv("MORPH_COMPILE_FALLBACK", "1")
    # Ensure the production switch overrides defaults, then restore global state.
    monkeypatch.setattr(dynamo_config, "optimize_ddp", True)
    monkeypatch.setattr(dynamo_config, "suppress_errors", False)
    real_compile = torch.compile

    def broken_backend(gm, inputs):
        raise RuntimeError("simulated backend device metadata failure")

    def compile_broken(fn, **kwargs):
        return real_compile(fn, backend=broken_backend, **kwargs)

    monkeypatch.setattr(torch, "compile", compile_broken)
    module = Harness()
    reference = copy.deepcopy(module.model)
    module.compile()
    assert dynamo_config.optimize_ddp is False
    assert dynamo_config.suppress_errors is True
    for n in (3, 7, 2):
        x = torch.randn(n, 3)
        module.model.zero_grad()
        reference.zero_grad()
        actual, expected = module.model(x), reference(x)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for p, q in zip(module.model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad)
        with torch.inference_mode():
            torch.testing.assert_close(module.model_ema(x), reference(x))


@pytest.mark.parametrize("accumulation", [1, 2, 4])
def test_resume_rebases_logging_counter_when_accumulation_changes(accumulation):
    class Harness(SetupMixin):
        automatic_optimization = False
        manual_gradient_accumulation_steps = accumulation

    checkpoint = {
        "global_step": 11515,
        "loops": {
            "fit_loop": {"epoch_loop.state_dict": {"_batches_that_stepped": 46060}}
        },
        "optimizer_states": [{"untouched": True}],
    }
    Harness().on_load_checkpoint(checkpoint)
    counter = checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"][
        "_batches_that_stepped"
    ]
    assert counter == 11515 * accumulation
    assert checkpoint["global_step"] == 11515
    assert checkpoint["optimizer_states"] == [{"untouched": True}]
