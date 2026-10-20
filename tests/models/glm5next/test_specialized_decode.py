# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the opt-in GLM H200 runtime; no CUDA initialization."""

import importlib.util
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

_PATH = (
    Path(__file__).parents[3] / "vllm/models/glm5next/nvidia/"
    "specialized_decode/runtime/contracts.py"
)


def load_contracts():
    spec = importlib.util.spec_from_file_location("glm53_decode_contracts", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DecodeContracts(unittest.TestCase):
    api: ClassVar[Any]

    @classmethod
    def setUpClass(cls):
        cls.api = load_contracts()

    def test_kernel_namespace_does_not_eagerly_import_model_classes(self):
        entry = _PATH.parents[3] / "__init__.py"
        spec = importlib.util.spec_from_file_location("glm53_entry_contracts", entry)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIn("Glm5NextForCausalLM", module.__all__)
        self.assertNotIn("Glm5NextForCausalLM", vars(module))
        with self.assertRaises(AttributeError):
            _ = module.missing_model

    def test_fresh_process_kernel_imports_and_registry_exports(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("Actual CUDA dependency environment required")
        code = """
import vllm.distributed
import vllm.model_executor.layers.linear
from vllm.models.glm5next import (
    Glm5NextForCausalLM, Glm5NextForConditionalGeneration, Glm5NextMTP,
)
from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    mhc_single_post_pre_tilelang,
)
from vllm.models.glm5next.nvidia.specialized_decode.ops import mhc_single
assert mhc_single_post_pre_tilelang is mhc_single.mhc_single_post_pre_tilelang
assert Glm5NextForCausalLM.__module__ == "vllm.models.glm5next.nvidia.model"
assert Glm5NextForConditionalGeneration.__module__ == (
    "vllm.models.glm5next.nvidia.model"
)
assert Glm5NextMTP is not None

from types import SimpleNamespace
from vllm.models.glm5next.nvidia.specialized_decode.runtime.decode import (
    SpecializedDecode,
)
from vllm.models.glm5next.nvidia.specialized_decode.runtime.contracts import (
    CacheBinding,
)
executor = SpecializedDecode.__new__(SpecializedDecode)
executor.binding = CacheBinding()
executor.plan_calls = executor.stock_calls = 0
executor.names = ("kda",)
assert executor.plan(SimpleNamespace(shape=(4096,)), None) is None
assert executor.plan(SimpleNamespace(shape=(4,)), None) is None

from unittest.mock import patch
import unittest
from vllm.config import CUDAGraphMode
import vllm.models.glm5next.nvidia.specialized_decode.runtime.decode as decode_module
md = SimpleNamespace(num_prefills=0, num_decodes=0, num_spec_decodes=1,
    num_spec_decode_tokens=4, num_actual_tokens=4,
    spec_state_indices_tensor=SimpleNamespace(shape=(1,4)),
    num_accepted_tokens=SimpleNamespace(shape=(1,)))
ctx = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE,
                      attn_metadata={"kda": md})
with patch.object(decode_module, "is_forward_context_available", return_value=True):
    with patch.object(decode_module, "get_forward_context", return_value=ctx):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "no bound KV"):
            executor.plan(SimpleNamespace(shape=(4,)), None)

import torch
class ForwardSpy(torch.nn.Module):
    def forward(self, **kwargs):
        return kwargs["inputs_embeds"]
wrapper = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
torch.nn.Module.__init__(wrapper)
wrapper.language_model = ForwardSpy()
sentinel = object()
assert wrapper(input_ids=None, positions=None, inputs_embeds=sentinel) is sentinel
assert wrapper(input_ids=None, positions=None, intermediate_tensors={"x": 1},
               inputs_embeds=sentinel) is None
"""
        subprocess.run([sys.executable, "-c", code], check=True, timeout=90)

    def test_default_disabled_and_invalid_backend_fails(self):
        self.assertFalse(self.api.enabled({}))
        self.assertFalse(self.api.enabled({"decode_backend": "auto"}))
        self.assertTrue(self.api.enabled({"decode_backend": "latency_specialized"}))
        with self.assertRaisesRegex(ValueError, "decode_backend"):
            self.api.enabled({"decode_backend": "typo"})

    def test_supported_deployment_and_all_refusals(self):
        valid = self.api.Deployment()
        self.assertEqual(self.api.unsupported(valid), ())
        bad = replace(
            valid,
            tp=4,
            max_num_seqs=2,
            speculative_method="mtp",
            features=("LoRA", "EP"),
        )
        reasons = self.api.unsupported(bad)
        self.assertEqual(len(reasons), 5)
        self.assertTrue(any("TP8" in x for x in reasons))
        self.assertTrue(any("DFlash K3" in x for x in reasons))
        with self.assertRaisesRegex(ValueError, "LoRA.*EP"):
            self.api.validate(bad)

    def test_uniform_verify_requires_every_layer_and_declines_other_steps(self):
        md = SimpleNamespace(
            num_prefills=0,
            num_decodes=0,
            num_spec_decodes=1,
            num_spec_decode_tokens=4,
            num_actual_tokens=4,
            spec_state_indices_tensor=SimpleNamespace(shape=(1, 4)),
            num_accepted_tokens=SimpleNamespace(shape=(1,)),
        )
        layers = ("a", "b")
        table = dict.fromkeys(layers, md)
        self.assertEqual(self.api.verification_metadata(table, layers, 4), (md, md))
        self.assertIsNone(self.api.verification_metadata(table, layers, 8))
        self.assertIsNone(self.api.verification_metadata({"a": md}, layers, 4))
        for field, value in [
            ("num_prefills", 1),
            ("num_decodes", 1),
            ("num_spec_decodes", 2),
            ("num_spec_decode_tokens", 3),
        ]:
            changed = SimpleNamespace(**vars(md))
            setattr(changed, field, value)
            self.assertIsNone(
                self.api.verification_metadata({"a": md, "b": changed}, layers, 4)
            )

    def test_partial_state_metadata_is_not_a_supported_verify(self):
        md = SimpleNamespace(
            num_prefills=0,
            num_decodes=0,
            num_spec_decodes=1,
            num_spec_decode_tokens=4,
            num_actual_tokens=4,
            spec_state_indices_tensor=SimpleNamespace(shape=(1, 3)),
            num_accepted_tokens=SimpleNamespace(shape=(1,)),
        )
        self.assertIsNone(self.api.verification_metadata({"a": md}, ("a",), 4))
        md.spec_state_indices_tensor.shape = (1, 4)
        md.num_accepted_tokens = None
        self.assertIsNone(self.api.verification_metadata({"a": md}, ("a",), 4))

    def test_vllm_multimodal_config_language_only_is_honored(self):
        ns = SimpleNamespace
        config = ns(
            parallel_config=ns(
                tensor_parallel_size=8,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                decode_context_parallel_size=1,
            ),
            model_config=ns(
                max_model_len=8192,
                quantization="fp8",
                multimodal_config=ns(language_model_only=True),
            ),
            cache_config=ns(enable_prefix_caching=False),
            speculative_config=ns(method="dflash", num_speculative_tokens=3),
            compilation_config=ns(
                mode=0, cudagraph_mode=ns(decode_mode=lambda: ns(name="FULL"))
            ),
            offload_config=ns(
                uva=ns(cpu_offload_gb=0), prefetch=ns(offload_group_size=0)
            ),
            lora_config=None,
            kv_transfer_config=None,
            weight_transfer_config=None,
            aux_output_config=ns(enable_return_routed_experts=False),
            scheduler_config=ns(max_num_seqs=1),
            kernel_config=ns(moe_backend="triton"),
            use_v2_model_runner=True,
        )
        deployment = self.api.deployment_from_vllm(config, 90, "NVIDIA H200")
        self.assertEqual(self.api.unsupported(deployment), ())
        config.kv_transfer_config = ns()
        config.weight_transfer_config = ns()
        config.cache_config.use_replayssm = True
        reasons = self.api.unsupported(
            self.api.deployment_from_vllm(config, 90, "NVIDIA H200")
        )
        self.assertEqual(len(reasons), 3)

    def test_discarded_profiling_graphs_allow_real_cache_bind(self):
        binding = self.api.CacheBinding()
        binding.bind(("profiling",))
        binding.captured = True
        self.assertTrue(binding.bind(("real",), graphs_discarded=True))
        self.assertFalse(binding.captured)
        self.assertEqual(binding.generation, 2)
        binding.captured = True
        with self.assertRaises(RuntimeError):
            binding.bind(("unexpected",))

    def test_cache_rebinding_is_allowed_before_capture_and_rejected_after(self):
        binding = self.api.CacheBinding()
        self.assertTrue(binding.bind((("conv", 10, (7, 3072, 6)),)))
        self.assertFalse(binding.bind((("conv", 10, (7, 3072, 6)),)))
        self.assertTrue(binding.bind((("conv", 20, (9, 3072, 6)),)))
        binding.captured = True
        self.assertFalse(binding.bind((("conv", 20, (9, 3072, 6)),)))
        with self.assertRaisesRegex(RuntimeError, "captured"):
            binding.bind((("conv", 30, (9, 3072, 6)),))


if __name__ == "__main__":
    unittest.main()
