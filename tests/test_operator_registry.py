from unittest import TestCase, main

import torch

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.operators import (
    OperatorProvider,
    OperatorRegistry,
    OperatorResolver,
)
from nanovllm.layers.rotary_embedding import RotaryEmbedding


def _identity_factory(owner):
    return owner


class CountingRegistry(OperatorRegistry):

    def __init__(self):
        super().__init__()
        self.resolve_calls = []

    def resolve(self, operator, override="auto", **capabilities):
        self.resolve_calls.append((operator, override, capabilities))
        return super().resolve(operator, override, **capabilities)


class OperatorRegistryTest(TestCase):

    def test_auto_uses_highest_priority_supported_provider(self):
        registry = OperatorRegistry()
        seen_capabilities = []

        def supports_cuda(*, device_type, layout):
            seen_capabilities.append((device_type, layout))
            return device_type == "cuda" and layout == "NHD"

        registry.register(
            "attention",
            OperatorProvider("native_torch", _identity_factory, priority=10),
        )
        registry.register(
            "attention",
            OperatorProvider(
                "flashinfer",
                _identity_factory,
                supports=supports_cuda,
                priority=100,
            ),
        )

        cpu = registry.resolve(
            "attention", device_type="cpu", layout="NHD"
        )
        cuda = registry.resolve(
            "attention", device_type="cuda", layout="NHD"
        )

        self.assertEqual(cpu.name, "native_torch")
        self.assertEqual(cuda.name, "flashinfer")
        self.assertEqual(
            seen_capabilities,
            [("cpu", "NHD"), ("cuda", "NHD")],
        )

    def test_native_alias_and_explicit_override_are_respected(self):
        registry = OperatorRegistry()
        registry.register(
            "rope",
            OperatorProvider("native_torch", _identity_factory, priority=10),
        )
        registry.register(
            "rope",
            OperatorProvider("native_cuda", _identity_factory, priority=20),
        )
        registry.register(
            "rope",
            OperatorProvider("flashinfer", _identity_factory, priority=100),
        )

        self.assertEqual(registry.resolve("rope").name, "flashinfer")
        self.assertEqual(
            registry.resolve("rope", override="native").name,
            "native_cuda",
        )
        self.assertEqual(
            registry.resolve("rope", override="native_torch").name,
            "native_torch",
        )
        with self.assertRaisesRegex(LookupError, "not registered"):
            registry.resolve("rope", override="does_not_exist")

    def test_forced_unsupported_provider_raises_instead_of_falling_back(self):
        registry = OperatorRegistry()
        registry.register(
            "norm",
            OperatorProvider("native_torch", _identity_factory, priority=10),
        )
        registry.register(
            "norm",
            OperatorProvider(
                "flashinfer",
                _identity_factory,
                supports=lambda *, device_type: device_type == "cuda",
                priority=100,
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "flashinfer"):
            registry.resolve(
                "norm", override="flashinfer", device_type="cpu"
            )
        self.assertEqual(
            registry.resolve("norm", device_type="cpu").name,
            "native_torch",
        )

    def test_resolver_merges_common_and_per_operator_capabilities(self):
        registry = OperatorRegistry()
        observed = []

        def supports(**capabilities):
            observed.append(capabilities)
            return True

        registry.register(
            "kv_store",
            OperatorProvider(
                "native_triton",
                lambda owner: owner.store,
                supports=supports,
            ),
        )

        class Owner:
            def store(self, value):
                return value

        resolver = OperatorResolver(
            registry=registry,
            device_type="cuda",
            dtype=torch.bfloat16,
        )
        owner = Owner()
        name, implementation = resolver.bind(
            "kv_store", owner, layout="NHD", head_dim=128
        )

        self.assertEqual(name, "native_triton")
        self.assertEqual(implementation("value"), "value")
        self.assertEqual(
            observed,
            [{
                "device_type": "cuda",
                "dtype": torch.bfloat16,
                "layout": "NHD",
                "head_dim": 128,
            }],
        )

    def test_layer_binds_once_and_forward_does_not_resolve_again(self):
        registry = CountingRegistry()

        def bind_test_implementation(_layer):
            def implementation(x):
                gate, value = x.chunk(2, dim=-1)
                return gate + value

            return implementation

        registry.register(
            "silu_and_mul",
            OperatorProvider(
                "test_provider", bind_test_implementation, priority=1000
            ),
        )
        resolver = OperatorResolver(registry=registry)
        layer = SiluAndMul(operator_resolver=resolver)

        self.assertEqual(layer.provider_name, "test_provider")
        self.assertEqual(len(registry.resolve_calls), 1)
        x = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        expected = x[..., :3] + x[..., 3:]

        for _ in range(3):
            torch.testing.assert_close(layer(x), expected)
        self.assertEqual(len(registry.resolve_calls), 1)

    def test_layers_bind_their_registered_native_providers(self):
        silu = SiluAndMul(
            operator_resolver=OperatorResolver(
                overrides={"silu_and_mul": "native"}
            )
        )
        norm = RMSNorm(
            8,
            operator_resolver=OperatorResolver(
                overrides={
                    "rms_norm": "native",
                    "fused_add_rms_norm": "native",
                }
            ),
        )
        rope = RotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=16,
            base=10000.0,
            operator_resolver=OperatorResolver(
                overrides={"rotary_embedding": "native"}
            ),
        )
        attention = Attention(
            num_heads=4,
            head_dim=8,
            scale=8 ** -0.5,
            num_kv_heads=2,
            operator_resolver=OperatorResolver(
                overrides={"kv_cache_store": "native"}
            ),
        )

        self.assertEqual(silu.provider_name, "native_torch")
        self.assertEqual(norm.rms_provider_name, "native_torch")
        self.assertEqual(norm.add_rms_provider_name, "native_torch")
        self.assertEqual(rope.provider_name, "native_torch")
        self.assertEqual(attention.kv_store_provider_name, "native_triton")
        self.assertIs(silu.forward_impl.__self__, silu)
        self.assertIs(norm.rms_impl.__self__, norm)
        self.assertIs(norm.add_rms_impl.__self__, norm)
        self.assertIs(rope.forward_impl.__self__, rope)


if __name__ == "__main__":
    main()
