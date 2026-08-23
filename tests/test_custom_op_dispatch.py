from unittest import TestCase, main

import torch

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.custom_op import CustomOp, CustomOpConfig
from nanovllm.layers.attention import KVCacheStore


class _NativeOnlyOp(CustomOp):
    op_name = "native_only"

    def __init__(self, custom_op_config: CustomOpConfig):
        self.dispatch_calls = 0
        self.native_calls = 0
        super().__init__(custom_op_config)

    def dispatch_forward(self):
        self.dispatch_calls += 1
        return super().dispatch_forward()

    def forward_native(self, value):
        self.native_calls += 1
        return value + 1


class CustomOpDispatchTest(TestCase):

    def test_platform_dispatch_is_bound_once_during_construction(self):
        op = _NativeOnlyOp(CustomOpConfig(platform="cpu"))

        self.assertEqual(op.dispatch_calls, 1)
        for value in range(3):
            self.assertEqual(op(value), value + 1)
        self.assertEqual(op.dispatch_calls, 1)
        self.assertEqual(op.native_calls, 3)

    def test_cpu_dispatch_falls_back_to_native_implementation(self):
        op = _NativeOnlyOp(CustomOpConfig(platform="cpu"))
        value = torch.tensor([1.0, 2.0])

        output = op(value)

        torch.testing.assert_close(output, value + 1)
        self.assertEqual(op.platform, "cpu")

    def test_unknown_implementation_is_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "does_not_exist"):
            SiluAndMul(
                CustomOpConfig(
                    platform="cpu",
                    overrides={"silu_and_mul": "does_not_exist"},
                )
            )

    def test_cuda_only_implementation_is_rejected_on_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            SiluAndMul(
                CustomOpConfig(
                    platform="cpu",
                    overrides={"silu_and_mul": "flashinfer"},
                )
            )

    def test_config_normalizes_and_freezes_override_choices(self):
        overrides = {"native_only": "NATIVE_TORCH"}
        config = CustomOpConfig(platform="CPU", overrides=overrides)
        overrides["native_only"] = "flashinfer"

        self.assertEqual(config.platform, "cpu")
        self.assertEqual(config.implementation_for("native_only"), "native_torch")
        with self.assertRaises(TypeError):
            config.overrides["native_only"] = "flashinfer"

    def test_kv_cache_store_ignores_negative_slots(self):
        store = KVCacheStore(
            CustomOpConfig(platform="cpu")
        )
        key = torch.arange(1, 25, dtype=torch.float32).reshape(3, 2, 4)
        value = key + 100
        k_cache = torch.full((2, 2, 2, 4), -1.0)
        v_cache = torch.full((2, 2, 2, 4), -1.0)
        slot_mapping = torch.tensor([3, -1, 0], dtype=torch.int64)
        expected_k = k_cache.clone()
        expected_v = v_cache.clone()
        expected_k.view(-1, 2, 4)[3].copy_(key[0])
        expected_v.view(-1, 2, 4)[3].copy_(value[0])
        expected_k.view(-1, 2, 4)[0].copy_(key[2])
        expected_v.view(-1, 2, 4)[0].copy_(value[2])

        result = store(key, value, k_cache, v_cache, slot_mapping)

        self.assertIsNone(result)
        torch.testing.assert_close(k_cache, expected_k)
        torch.testing.assert_close(v_cache, expected_v)


if __name__ == "__main__":
    main()
