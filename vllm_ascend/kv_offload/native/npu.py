# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Use the upstream CPU manager with canonical Ascend cache transfers."""

from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from vllm_ascend.kv_offload.native.cpu_npu import CpuNpuOffloadingHandlers


class NPUOffloadingSpec(CPUOffloadingSpec):
    """Override only the CUDA platform gate and DMA handler construction."""

    def get_handlers(self, kv_caches):
        if self._handlers is None:
            self._handlers = CpuNpuOffloadingHandlers(
                kv_caches=kv_caches,
                blocks_per_chunk=self.block_size_factor,
                num_cpu_blocks=self.num_blocks,
            )
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler
