# SPDX-License-Identifier: Apache-2.0
"""Reject prefill caps that cannot advance Ascend's block-aligned hybrid cache."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.patch.platform import patch_mamba_config


def make_config(threshold=0, budget=8192, mode="align", block_size=128, store=False):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            block_size=block_size,
            mamba_page_size_padded=None,
            mamba_cache_mode=mode,
            enable_prefix_caching=not store,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            architecture="Qwen3_5ForConditionalGeneration",
            use_mla=False,
            get_num_kv_heads=lambda _: 4,
            get_head_size=lambda: 256,
            max_model_len=262144,
        ),
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            long_prefill_token_threshold=threshold,
            max_num_batched_tokens=budget,
        ),
        kv_transfer_config=SimpleNamespace(kv_connector="AscendStoreConnector") if store else None,
        speculative_config=None,
    )


@pytest.fixture
def verify(monkeypatch):
    # Qwen3.5-9B TP1 shapes resolve to a 1024-token attention block on Ascend.
    model = SimpleNamespace(
        get_mamba_state_shape_from_config=lambda _: ((32, 128, 128), (3, 4096)),
        get_mamba_state_dtype_from_config=lambda _: (torch.float32, torch.bfloat16),
    )
    monkeypatch.setattr(patch_mamba_config.ModelRegistry, "resolve_model_cls", lambda *a, **kw: (model, None))
    monkeypatch.setattr(patch_mamba_config.MambaModelConfig, "verify_and_update_config", Mock())
    return lambda config: patch_mamba_config.verify_and_update_config.__func__(None, config)


@pytest.mark.parametrize("threshold", [1, 512, 1023])
def test_rejects_sub_block_prefill_cap(verify, threshold):
    with pytest.raises(ValueError, match=r"long_prefill_token_threshold.*resolved block_size \(1024\)"):
        verify(make_config(threshold=threshold))


@pytest.mark.parametrize("threshold", [0, 1024, 1536, 2048, 4096])
def test_accepts_uncapped_or_progressing_prefill(verify, threshold):
    config = make_config(threshold=threshold)
    verify(config)
    assert config.cache_config.block_size == 1024
    assert config.scheduler_config.long_prefill_token_threshold == threshold


def test_checks_user_requested_larger_block(verify):
    with pytest.raises(ValueError, match=r"resolved block_size \(2048\)"):
        verify(make_config(threshold=1024, block_size=2048))


def test_rejects_insufficient_total_budget(verify):
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        verify(make_config(budget=512))


def test_allows_one_block_budget(verify):
    verify(make_config(threshold=1024, budget=1024))


def test_does_not_restrict_unaligned_mode(verify):
    verify(make_config(threshold=512, budget=512, mode="none"))


def test_allows_context_shorter_than_one_block(verify):
    config = make_config(threshold=256, budget=512)
    config.model_config.max_model_len = 512
    verify(config)


def test_checks_mode_enabled_by_store_connector(verify):
    with pytest.raises(ValueError, match="long_prefill_token_threshold"):
        verify(make_config(threshold=512, mode="none", store=True))


def test_scheduler_cannot_advance_sub_block_chunk():
    scheduler = SimpleNamespace(cache_config=SimpleNamespace(block_size=1024), use_eagle=False)
    request = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=4096, num_tokens=4096)
    assert Scheduler._mamba_block_aligned_split(scheduler, request, 512) == 0
    assert Scheduler._mamba_block_aligned_split(scheduler, request, 1024) == 1024
