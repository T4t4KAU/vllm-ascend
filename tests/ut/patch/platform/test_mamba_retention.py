# SPDX-License-Identifier: Apache-2.0
"""Exercise real 0.22.1 allocation, lookup and eviction with sparse states."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec

from vllm_ascend.core.mamba_retention import MambaRetentionConfig, MambaRetentionState
from vllm_ascend.patch.platform import patch_mamba_manager


@pytest.fixture
def configure(monkeypatch):
    original_init = KVCacheManager.__init__

    def apply(interval, diagnostics=True, prefer_reuse_boundaries=False):
        config = SimpleNamespace(
            additional_config={
                "mamba_cache_retention": {
                    "interval": interval,
                    "diagnostics": diagnostics,
                    "prefer_reuse_boundaries": prefer_reuse_boundaries,
                }
            },
            speculative_config=None,
            kv_transfer_config=None,
            model_config=SimpleNamespace(architecture="Qwen3_5ForConditionalGeneration"),
        )

        def initialize(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            patch_mamba_manager.configure_mamba_retention(config, self)

        monkeypatch.setattr(KVCacheManager, "__init__", initialize)
        return config

    return apply


def make_manager(num_blocks=256):
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attn"], FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
            ),
            KVCacheGroupSpec(
                ["gdn"], MambaSpec(block_size=4, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align")
            ),
        ],
    )
    return KVCacheManager(config, max_model_len=256, hash_block_size=4, max_num_batched_tokens=32)


def request(name, tokens=35, hashes=None):
    return SimpleNamespace(
        request_id=name,
        num_prompt_tokens=tokens,
        num_tokens=tokens,
        num_computed_tokens=0,
        num_preemptions=0,
        skip_reading_prefix_cache=False,
        block_hashes=hashes or [i.to_bytes(8, "big") for i in range(tokens // 4)],
    )


def prefill(manager, req, cap=4, finish=True):
    scheduler = SimpleNamespace(cache_config=SimpleNamespace(block_size=4), use_eagle=False, kv_cache_manager=manager)
    hits, computed = manager.get_computed_blocks(req)
    chunks = []
    while computed < req.num_tokens:
        for group in manager.coordinator.single_type_managers:
            group.new_step_starts()
        req.num_computed_tokens = computed
        count = Scheduler._mamba_block_aligned_split(scheduler, req, min(cap, req.num_tokens - computed))
        assert count > 0
        # Use the real admission/allocation path, including cached state refs.
        if not chunks:
            req.num_computed_tokens = 0
        allocated = manager.allocate_slots(
            req, count, num_new_computed_tokens=0 if chunks else computed, new_computed_blocks=None if chunks else hits
        )
        assert allocated is not None
        computed += count
        req.num_computed_tokens = computed
        chunks.append(computed)
    if finish:
        manager.free(req)
    return chunks


@pytest.mark.parametrize("interval", [-1, True, 1.5, "8192"])
def test_invalid_interval(interval):
    with pytest.raises(ValueError, match="interval"):
        MambaRetentionConfig(interval)


def test_resolved_alignment_required(configure):
    configure(6)
    with pytest.raises(ValueError, match="multiple"):
        make_manager()


@pytest.mark.parametrize("field", ["speculative_config", "kv_transfer_config"])
def test_rejects_unsupported_paths(configure, field):
    config = configure(16)
    setattr(config, field, object())
    with pytest.raises(ValueError, match="without speculation"):
        make_manager()


@pytest.mark.parametrize("interval, expected", [(None, list(range(4, 36, 4))), (16, [16, 32]), (0, [32])])
def test_sparse_states_keep_full_attention(configure, interval, expected):
    configure(interval)
    manager = make_manager()
    req = request("parent")
    prefill(manager, req)
    pool = manager.block_pool
    retained = [4 * (i + 1) for i, h in enumerate(req.block_hashes) if pool.get_cached_block(h, [1])]
    assert retained == expected
    assert all(pool.get_cached_block(h, [0]) for h in req.block_hashes)
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1


def test_exact_prompt_replay_and_continuation_boundaries(configure):
    configure(16)
    manager = make_manager()
    chunks = prefill(manager, request("parent", 32), cap=16)
    assert 28 in chunks and 32 in chunks
    assert manager.get_computed_blocks(request("replay", 32))[1] == 28
    assert manager.get_computed_blocks(request("append", 35))[1] == 32


def test_shared_prefix_recomputed_once_then_retained(configure):
    configure(16)
    manager = make_manager()
    prefill(manager, request("parent"))
    # A fork ends its shared prefix inside an earlier turn, at token 24.
    fork_hashes = [i.to_bytes(8, "big") for i in range(6)] + [b"fork0001", b"fork0002"]
    fork = request("fork", hashes=fork_hashes)
    assert manager.get_computed_blocks(fork)[1] == 16
    assert fork.ascend_mamba_shared_boundary == 24
    chunks = prefill(manager, fork, cap=16)
    assert 24 in chunks
    sibling_hashes = fork_hashes[:6] + [b"sibl0001", b"sibl0002"]
    sibling = request("sibling", hashes=sibling_hashes)
    assert manager.get_computed_blocks(sibling)[1] == 24
    assert manager.block_pool.ascend_mamba_retention.checkpoint_gap_tokens >= 8


def test_same_step_state_cannot_be_consumed(configure):
    configure(16)
    manager = make_manager()
    req = request("parent", 32)
    prefill(manager, req, cap=16, finish=False)
    mamba = manager.coordinator.single_type_managers[1]
    hits, length = manager.get_computed_blocks(request("child", 35))
    assert length == 32
    assert mamba.get_num_blocks_to_allocate("child", 35, hits.blocks[1], 32, 35) > manager.block_pool.num_gpu_blocks
    mamba.new_step_starts()
    assert mamba.get_num_blocks_to_allocate("child", 35, hits.blocks[1], 32, 35) < manager.block_pool.num_gpu_blocks


def test_uncached_free_blocks_reused_before_retained_states():
    pool = BlockPool(4, True, 4)
    pool.ascend_mamba_retention = MambaRetentionState(MambaRetentionConfig(16))
    cached, uncached, live = pool.get_new_blocks(3)
    cached.block_hash = b"cached00"
    pool.free_blocks([cached])
    pool.free_blocks([uncached])
    assert pool.get_new_blocks(1) == [uncached]
    assert live.ref_cnt == 1 and cached.ref_cnt == 0
    pool.touch([cached])
    assert cached.ref_cnt == 1
    assert pool.get_num_free_blocks() == 0


def test_shared_reference_not_recycled_until_last_release():
    pool = BlockPool(2, True, 4)
    pool.ascend_mamba_retention = MambaRetentionState(MambaRetentionConfig(16))
    block = pool.get_new_blocks(1)[0]
    pool.touch([block])
    pool.free_blocks([block])
    assert pool.get_num_free_blocks() == 0
    pool.free_blocks([block])
    assert pool.get_new_blocks(1) == [block]


def test_default_queue_order_unchanged():
    pool = BlockPool(3, True, 4)
    first, second = pool.get_new_blocks(2)
    pool.free_blocks([first, second])
    assert pool.get_new_blocks(2) == [first, second]


def test_pressure_preserves_more_complete_prefix(configure):
    def run(interval):
        configure(interval)
        manager = make_manager(num_blocks=30)
        prefill(manager, request("parent", 51))
        # Unrelated long prefill puts both attention and recurrent state in
        # the same constrained pool, forcing real evictions.
        prefill(manager, request("other", 35, [b"other" + bytes([i]) for i in range(8)]))
        return manager.get_computed_blocks(request("resume", 55))[1]

    assert run(16) > run(None)


def test_reset_drops_all_retained_checkpoints(configure):
    configure(16)
    manager = make_manager()
    prefill(manager, request("parent"))
    assert manager.reset_prefix_cache()
    assert manager.get_computed_blocks(request("replay"))[1] == 0


def test_generated_blocks_are_retained_for_next_turn(configure):
    configure(16)
    manager = make_manager()
    req = request("parent", 19)
    prefill(manager, req, finish=False)
    # Decode across two block boundaries, then replay the generated output.
    for total in range(20, 25):
        req.num_tokens = total
        req.block_hashes = [i.to_bytes(8, "big") for i in range(total // 4)]
        for group in manager.coordinator.single_type_managers:
            group.new_step_starts()
        assert manager.allocate_slots(req, 1) is not None
        req.num_computed_tokens += 1
    manager.free(req)
    assert manager.get_computed_blocks(request("next_turn", 27))[1] == 24


def test_aborted_request_releases_running_state(configure):
    configure(16)
    manager = make_manager()
    req = request("aborted", 35)
    assert manager.allocate_slots(req, 4) is not None
    manager.free(req)
    pool = manager.block_pool
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    # The skipped checkpoint has no hash and cannot become a false cache hit.
    assert manager.get_computed_blocks(request("retry", 35))[1] == 0


def test_skip_cache_read_does_not_inherit_previous_boundary(configure):
    configure(16)
    manager = make_manager()
    prefill(manager, request("parent"))
    manager.get_computed_blocks(request("warm"))
    req = request("cold")
    req.skip_reading_prefix_cache = True
    assert manager.get_computed_blocks(req)[1] == 0
    assert req.ascend_mamba_shared_boundary == 0


@pytest.mark.parametrize("value", [1, "true", None])
def test_reuse_preference_requires_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        MambaRetentionConfig(interval=16, prefer_reuse_boundaries=value)


def test_reuse_preference_requires_sparse_retention():
    with pytest.raises(ValueError, match="explicit retention interval"):
        MambaRetentionConfig(prefer_reuse_boundaries=True)


def exhaust_empty_blocks(pool):
    cached = sum(block.block_hash is not None for block in pool.blocks)
    return pool.get_new_blocks(pool.get_num_free_blocks() - cached)


def assert_free_queue_consistent(pool):
    queue = pool.free_block_queue
    current = queue.fake_free_list_head
    ids = []
    while current.next_free_block is not queue.fake_free_list_tail:
        following = current.next_free_block
        assert following is not None and following.prev_free_block is current
        assert following.ref_cnt == 0 and not following.is_null
        ids.append(following.block_id)
        assert len(ids) <= pool.num_gpu_blocks
        current = following
    assert queue.fake_free_list_tail.prev_free_block is current
    assert len(ids) == len(set(ids)) == pool.get_num_free_blocks()


def test_periodic_victims_preserve_attention_and_semantic_states(configure):
    configure(8, prefer_reuse_boundaries=True)
    manager = make_manager()
    req = request("parent")
    prefill(manager, req)
    prefill(manager, request("other", hashes=[b"other" + bytes([i]) for i in range(8)]))
    pool = manager.block_pool
    state = pool.ascend_mamba_retention
    assert len(state.checkpoints.hashes) == len(state.checkpoints.available) == 6
    assert state.snapshot()["periodic_blocks"] == 6
    periodic_ids = set(state.checkpoints.hashes)
    scratch = exhaust_empty_blocks(pool)
    assert state.preferred_periodic_reclaims == 0
    pool.enable_kv_cache_events = True
    assert {b.block_id for b in pool.get_new_blocks(6)} == periodic_ids
    assert len(pool.kv_event_queue) == 6
    assert all(event.group_idx == 1 for event in pool.kv_event_queue)
    assert not state.checkpoints.hashes and not state.checkpoints.available
    assert all(pool.get_cached_block(h, [0]) for h in req.block_hashes)
    assert pool.get_cached_block(req.block_hashes[7], [1])
    assert state.preferred_periodic_reclaims > 0
    pool.free_blocks(scratch)
    assert_free_queue_consistent(pool)


def test_actually_reused_periodic_checkpoint_is_promoted(configure):
    configure(8, prefer_reuse_boundaries=True)
    manager = make_manager()
    req = request("parent")
    prefill(manager, req)
    pool = manager.block_pool
    state = pool.ascend_mamba_retention
    checkpoint = pool.get_cached_block(req.block_hashes[3], [1])[0]
    pool.touch([checkpoint])
    assert checkpoint.ref_cnt == 1
    assert not state.checkpoints.contains(checkpoint)
    assert state.promoted_checkpoints == 1
    pool.free_blocks([checkpoint])
    exhaust_empty_blocks(pool)
    victims = pool.get_new_blocks(2)
    assert checkpoint not in victims
    assert pool.get_cached_block(req.block_hashes[3], [1])
    assert_free_queue_consistent(pool)


def test_live_checkpoint_cannot_be_preferred_victim(configure):
    configure(8, prefer_reuse_boundaries=True)
    manager = make_manager()
    req = request("running", 17)
    prefill(manager, req, finish=False)
    pool = manager.block_pool
    state = pool.ascend_mamba_retention
    live_ids = {b.block_id for b in pool.blocks if b.ref_cnt > 0 and not b.is_null}
    allocated = pool.get_new_blocks(pool.get_num_free_blocks())
    assert not live_ids.intersection(b.block_id for b in allocated)
    assert all(pool.blocks[i].ref_cnt > 0 for i in live_ids)
    assert not state.checkpoints.available
    assert_free_queue_consistent(pool)


def test_failed_admission_does_not_consume_priority_index(configure):
    configure(8, prefer_reuse_boundaries=True)
    manager = make_manager()
    prefill(manager, request("parent"))
    pool = manager.block_pool
    before = dict(pool.ascend_mamba_retention.checkpoints.available)
    with pytest.raises(ValueError):
        pool.get_new_blocks(pool.get_num_free_blocks() + 1)
    assert pool.ascend_mamba_retention.checkpoints.available == before
    assert_free_queue_consistent(pool)


def test_explicit_eviction_and_reset_clear_priority_metadata(configure):
    configure(8, prefer_reuse_boundaries=True)
    manager = make_manager()
    prefill(manager, request("parent"))
    pool = manager.block_pool
    state = pool.ascend_mamba_retention
    victim_id = next(iter(state.checkpoints.hashes))
    pool.evict_blocks({victim_id})
    assert victim_id not in state.checkpoints.hashes
    assert victim_id not in state.checkpoints.available
    assert manager.reset_prefix_cache()
    assert not state.checkpoints.hashes and not state.checkpoints.available
    assert_free_queue_consistent(pool)


def test_failed_reset_preserves_live_metadata(configure):
    configure(8, prefer_reuse_boundaries=True)
    manager = make_manager()
    req = request("running")
    prefill(manager, req, finish=False)
    index = manager.block_pool.ascend_mamba_retention.checkpoints
    before = dict(index.hashes)
    assert before
    assert not manager.reset_prefix_cache()
    assert index.hashes == before
    manager.free(req)
    assert manager.reset_prefix_cache()
    assert not index.hashes


def test_semantic_only_retention_has_no_periodic_victims(configure):
    configure(0, prefer_reuse_boundaries=True)
    manager = make_manager()
    prefill(manager, request("parent"))
    assert not manager.block_pool.ascend_mamba_retention.checkpoints.hashes


def test_admission_retry_preserves_observed_shared_junction(configure):
    configure(16)
    manager = make_manager()
    parent = request("parent")
    prefill(manager, parent)
    fork_hashes = parent.block_hashes[:6] + [b"fork0001", b"fork0002"]
    fork = request("fork", hashes=fork_hashes)
    assert manager.get_computed_blocks(fork)[1] == 16
    assert fork.ascend_mamba_shared_boundary == 24
    # Another request evicts attention KV while the fork waits for capacity.
    victim = manager.block_pool.get_cached_block(parent.block_hashes[5], [0])[0]
    manager.block_pool.evict_blocks({victim.block_id})
    assert manager.get_computed_blocks(fork)[1] == 16
    assert fork.ascend_mamba_shared_boundary == 24
    assert 24 in prefill(manager, fork, cap=16)
    sibling = request("sibling", hashes=fork_hashes[:6] + [b"sibl0001", b"sibl0002"])
    assert manager.get_computed_blocks(sibling)[1] == 24


def test_priority_metadata_stays_bounded_under_block_reuse(configure):
    configure(8, prefer_reuse_boundaries=True)
    manager = make_manager(num_blocks=30)
    pool = manager.block_pool
    for i in range(80):
        hashes = [bytes([i % 13, j]) for j in range(8)]
        prefill(manager, request(f"turn-{i}", hashes=hashes))
        index = pool.ascend_mamba_retention.checkpoints
        assert len(index.hashes) <= pool.num_gpu_blocks
        assert all(pool.blocks[block_id].block_hash == h for block_id, h in index.hashes.items())
        assert all(block.ref_cnt == 0 and index.contains(block) for block in index.available.values())
        assert_free_queue_consistent(pool)


def test_disabled_preference_preserves_allocator_order(configure):
    configure(8)
    manager = make_manager(num_blocks=40)
    prefill(manager, request("parent"))
    prefill(manager, request("other", hashes=[b"other" + bytes([i]) for i in range(8)]))
    pool = manager.block_pool
    expected = []
    block = pool.free_block_queue.fake_free_list_head.next_free_block
    while block is not pool.free_block_queue.fake_free_list_tail:
        expected.append(block.block_id)
        block = block.next_free_block
    assert [b.block_id for b in pool.get_new_blocks(len(expected))] == expected
    assert not pool.ascend_mamba_retention.checkpoints.hashes
    assert_free_queue_consistent(pool)
