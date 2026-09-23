# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
import json
from collections.abc import Iterable

import vllm.v1.core.single_type_kv_cache_manager as single_type_kv_cache_manager
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import (
    BlockHashList,
    BlockPool,
    KVCacheBlock,
    KVCacheSpec,
    MambaManager,
    MambaSpec,
)
from vllm.v1.request import Request

from vllm_ascend.core.mamba_retention import (
    MambaRetentionConfig,
    MambaRetentionState,
    checkpoint_mask,
    insert_free_block_before,
    reuse_boundaries,
)
from vllm_ascend.utils import vllm_version_is


class AscendMambaManager(MambaManager):
    def __init__(self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        if self.enable_caching:
            self.block_size = kv_cache_spec.block_size
        self.retention_state = None

    def cache_blocks(self, request: Request, num_tokens: int, **kwargs) -> None:
        state = self.retention_state
        if state is None:
            return super().cache_blocks(request, num_tokens, **kwargs)
        start = self.num_cached_block.get(request.request_id, 0)
        end = num_tokens // self.block_size
        if state.config.interval is None:
            super().cache_blocks(request, num_tokens, **kwargs)
            if state.config.diagnostics:
                state.retained_blocks += sum(
                    not block.is_null and block.block_hash is not None
                    for block in self.req_to_blocks[request.request_id][start:end]
                )
            return
        if start >= end:
            return
        boundaries = reuse_boundaries(
            request.num_prompt_tokens,
            self.block_size,
            getattr(request, "ascend_mamba_shared_boundary", 0),
        )
        # Generated tokens are commonly replayed on the next turn. Keep their
        # completed blocks as well; the sparse policy targets long prefills.
        mask = checkpoint_mask(
            start, end, self.block_size, state.config.interval, boundaries, request.num_prompt_tokens
        )
        blocks = self.req_to_blocks[request.request_id]
        self.block_pool.cache_full_blocks(
            request, blocks, start, end, self.block_size, self.kv_cache_group_id, block_mask=mask
        )
        self.num_cached_block[request.request_id] = end
        # Keep upstream's barrier: another request cannot consume states that
        # are scheduled but have not yet been produced by the worker.
        for i, (keep, block) in enumerate(zip(mask, blocks[start:end]), start=start):
            if block.is_null:
                continue
            if keep:
                assert block.block_hash is not None
                self.cached_blocks_this_step.add(block.block_hash)
                state.retained_blocks += 1
                if state.config.prefer_reuse_boundaries:
                    position = (i + 1) * self.block_size
                    if position not in boundaries and position <= request.num_prompt_tokens:
                        state.checkpoints.remember(block)
                    else:
                        state.checkpoints.discard(block)
            else:
                state.skipped_blocks += 1

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        use_eagle: bool = False,
        drop_eagle_block: bool = False,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, MambaSpec), "MambaManager can only be used for mamba groups"
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple([] for _ in range(len(kv_cache_group_ids)))
        block_size = kv_cache_spec.block_size
        state = getattr(block_pool, "ascend_mamba_retention", None)
        if state is not None:
            # Full attention is checked first by the hybrid coordinator. This
            # is an observed local shared prefix, not a guess from session IDs.
            state.candidate_tokens = max(state.candidate_tokens, max_length // alignment_tokens * alignment_tokens)
        max_num_blocks = max_length // block_size
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids):
                if block_size != alignment_tokens and (i + 1) * block_size % alignment_tokens != 0:
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                break
        return computed_blocks


def configure_mamba_retention(vllm_config, cache_manager) -> None:
    # EngineCore and Worker have separate lifetimes. Pass configuration from
    # the scheduler explicitly instead of using the worker's Ascend singleton.
    config = MambaRetentionConfig(**(vllm_config.additional_config or {}).get("mamba_cache_retention", {}))
    if config.interval is None and not config.diagnostics:
        return
    if (
        not vllm_version_is("0.22.1")
        or vllm_config.speculative_config is not None
        or vllm_config.kv_transfer_config is not None
        or vllm_config.model_config.architecture != "Qwen3_5ForConditionalGeneration"
    ):
        raise ValueError("mamba_cache_retention requires vLLM 0.22.1 Qwen3.5 without speculation or KV transfer")
    managers = [m for m in cache_manager.coordinator.single_type_managers if isinstance(m, AscendMambaManager)]
    if not managers:
        raise ValueError("mamba_cache_retention requires a hybrid Mamba cache")
    state = MambaRetentionState(config)
    for manager in managers:
        if (
            not manager.enable_caching
            or manager.mamba_cache_mode != "align"
            or manager.dcp_world_size != 1
            or manager.pcp_world_size != 1
        ):
            raise ValueError("mamba_cache_retention requires align APC without context parallelism")
        if config.interval is not None and config.interval % manager.block_size:
            raise ValueError("mamba_cache_retention.interval must be a multiple of the resolved block size")
        manager.retention_state = state
    cache_manager.block_pool.ascend_mamba_retention = state


_SCHEDULER_INIT = Scheduler.__init__


def init_scheduler(self, *args, **kwargs) -> None:
    _SCHEDULER_INIT(self, *args, **kwargs)
    configure_mamba_retention(self.vllm_config, self.kv_cache_manager)


_GET_COMPUTED_BLOCKS = KVCacheManager.get_computed_blocks
_FREE_BLOCKS = BlockPool.free_blocks
_GET_NEW_BLOCKS = BlockPool.get_new_blocks
_TOUCH_BLOCKS = BlockPool.touch
_EVICT_CACHED_BLOCK = BlockPool._maybe_evict_cached_block
_RESET_PREFIX_CACHE = BlockPool.reset_prefix_cache


def get_computed_blocks(self, request: Request):
    state = getattr(self.block_pool, "ascend_mamba_retention", None)
    if state is None:
        return _GET_COMPUTED_BLOCKS(self, request)
    state.candidate_tokens = 0
    blocks, hit_tokens = _GET_COMPUTED_BLOCKS(self, request)
    # Admission retries can see a shorter attention hit after another request
    # evicts KV. Preserve an already observed junction until this request ends.
    previous = getattr(request, "ascend_mamba_shared_boundary", 0)
    candidate = state.candidate_tokens if state.candidate_tokens > hit_tokens else 0
    request.ascend_mamba_shared_boundary = 0 if request.skip_reading_prefix_cache else max(previous, candidate)
    if state.config.diagnostics:
        state.lookups += 1
        state.full_attention_hit_tokens += state.candidate_tokens
        state.hybrid_hit_tokens += hit_tokens
        state.checkpoint_gap_tokens += max(0, state.candidate_tokens - hit_tokens)
        if state.lookups % 16 == 0:
            snapshot = state.snapshot()
            snapshot["resident_hashed_blocks"] = sum(block.block_hash is not None for block in self.block_pool.blocks)
            logger.info("Ascend Mamba retention: %s", json.dumps(snapshot, sort_keys=True))
    return blocks, hit_tokens


def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    state = getattr(self, "ascend_mamba_retention", None)
    if state is None or state.config.interval is None:
        return _FREE_BLOCKS(self, ordered_blocks)
    cached = []
    uncached = []
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            (uncached if block.block_hash is None else cached).append(block)
            if state.config.prefer_reuse_boundaries:
                state.checkpoints.release(block)
    # Backport uncached-first reuse without changing the vLLM wheel. Keep the
    # original LRU ordering among hashed blocks. Live/copy-source blocks retain
    # their references and can never enter this queue.
    queue = self.free_block_queue
    for block in reversed(uncached):
        first = queue.fake_free_list_head.next_free_block
        assert first is not None
        insert_free_block_before(queue, block, first)
    queue.append_n(cached)
    state.recycled_uncached_blocks += len(uncached)


def get_new_blocks(self, num_blocks: int):
    state = getattr(self, "ascend_mamba_retention", None)
    if state is not None and state.config.prefer_reuse_boundaries and 0 < num_blocks <= self.get_num_free_blocks():
        queue = self.free_block_queue
        current = queue.fake_free_list_head.next_free_block
        for _ in range(num_blocks):
            if not state.checkpoints.available:
                break
            assert current is not None
            # Consume scratch/empty blocks first. Only substitute a victim
            # when allocation would otherwise evict a cached block.
            if current.block_hash is not None and (periodic := state.checkpoints.take()) is not None:
                if periodic is not current:
                    queue.remove(periodic)
                    insert_free_block_before(queue, periodic, current)
                    state.preferred_periodic_reclaims += 1
                    current = periodic
            current = current.next_free_block
    # Preserve upstream allocation, refcounts, metrics and eviction events.
    return _GET_NEW_BLOCKS(self, num_blocks)


def touch_blocks(self, blocks) -> None:
    _TOUCH_BLOCKS(self, blocks)
    state = getattr(self, "ascend_mamba_retention", None)
    if state is not None and state.config.prefer_reuse_boundaries:
        for block in blocks:
            if state.checkpoints.contains(block):
                state.checkpoints.discard(block)
                state.promoted_checkpoints += 1


def evict_cached_block(self, block) -> bool:
    state = getattr(self, "ascend_mamba_retention", None)
    if state is not None:
        state.checkpoints.discard(block)
    return _EVICT_CACHED_BLOCK(self, block)


def reset_prefix_cache(self) -> bool:
    reset = _RESET_PREFIX_CACHE(self)
    state = getattr(self, "ascend_mamba_retention", None)
    if reset and state is not None:
        state.checkpoints.clear()
    return reset


single_type_kv_cache_manager.MambaManager = AscendMambaManager
if vllm_version_is("0.22.1"):
    single_type_kv_cache_manager.spec_manager_map[MambaSpec] = AscendMambaManager
    Scheduler.__init__ = init_scheduler
    KVCacheManager.get_computed_blocks = get_computed_blocks
    BlockPool.free_blocks = free_blocks
    BlockPool.get_new_blocks = get_new_blocks
    BlockPool.touch = touch_blocks
    BlockPool._maybe_evict_cached_block = evict_cached_block
    BlockPool.reset_prefix_cache = reset_prefix_cache
