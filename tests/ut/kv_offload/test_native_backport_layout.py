# SPDX-License-Identifier: Apache-2.0
"""Layout regressions; also runnable directly in the matching runtime."""

import unittest

import torch

from vllm_ascend.kv_offload.native.offloading_connector import (
    _canonicalize_split_cache,
    _make_int8_block_view,
)


class TestNativeBackportLayout(unittest.TestCase):
    def test_separate_key_value_allocations(self):
        key = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        value = key.clone() + 100
        views = _canonicalize_split_cache((key, value), 4, 64)
        self.assertEqual(len(views), 2)
        for (view, size), tensor in zip(views, (key, value)):
            self.assertEqual(size, 32)
            self.assertTrue(torch.equal(view, tensor.view(torch.int8)))

    def test_offset_and_padding_are_preserved(self):
        storage = torch.arange(100, dtype=torch.float32)
        tensor = torch.as_strided(storage, (4, 8), (12, 1), 3)
        view = _make_int8_block_view(tensor, 4, 1, 32)
        self.assertEqual(view.data_ptr(), tensor.data_ptr())
        self.assertEqual(view.stride(), (48, 1))
        self.assertTrue(torch.equal(view, tensor.contiguous().view(torch.int8)))

    def test_mixed_state_dtypes(self):
        conv = torch.ones((4, 16), dtype=torch.bfloat16)
        recurrent = torch.full((4, 32), 2, dtype=torch.float32)
        views = _canonicalize_split_cache([conv, recurrent], 4, 160)
        self.assertEqual([size for _, size in views], [32, 128])

    def test_coalesces_physical_attention_blocks(self):
        key = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        value = key.clone() + 100
        views = _canonicalize_split_cache((key, value), 4, 128)
        self.assertEqual([view.shape for view, _ in views], [(4, 64)] * 2)
        self.assertTrue(torch.equal(views[0][0], key.view(torch.int8).reshape(4, 64)))

    def test_containing_view_does_not_duplicate_components(self):
        page = torch.arange(64, dtype=torch.float32).reshape(4, 16)
        views = _canonicalize_split_cache((page[:, :8], page[:, 8:], page), 4, 64)
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0][0].data_ptr(), page.data_ptr())

    def test_invalid_layouts_rejected(self):
        with self.assertRaises(ValueError):
            _make_int8_block_view(torch.zeros(4, 8)[:, ::2], 4, 1, 16)
        with self.assertRaises(ValueError):
            _make_int8_block_view(torch.zeros(4, 8), 5, 1, 32)
        with self.assertRaises(ValueError):
            _canonicalize_split_cache((torch.zeros(4, 8),), 4, 31)
        with self.assertRaises(TypeError):
            _canonicalize_split_cache((), 4, 32)


if __name__ == "__main__":
    unittest.main()
