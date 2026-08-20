from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


Supports = Callable[..., bool]
ProviderFactory = Callable[[Any], Callable]


def _supports_all(**_: Any) -> bool:
    return True


@dataclass(frozen=True)
class OperatorProvider:
    name: str
    callable: ProviderFactory
    supports: Supports = _supports_all
    priority: int = 0


class OperatorRegistry:

    def __init__(self) -> None:
        self._providers: dict[str, list[OperatorProvider]] = {}

    def register(self, operator: str, provider: OperatorProvider) -> None:
        providers = self._providers.setdefault(operator, [])
        if any(item.name == provider.name for item in providers):
            raise ValueError(
                f"provider {provider.name!r} is already registered for {operator!r}"
            )
        providers.append(provider)

    def providers(self, operator: str) -> tuple[OperatorProvider, ...]:
        return tuple(self._providers.get(operator, ()))

    def resolve(
        self,
        operator: str,
        override: str = "auto",
        **capabilities: Any,
    ) -> OperatorProvider:
        providers = self._providers.get(operator)
        if not providers:
            raise LookupError(f"no providers are registered for operator {operator!r}")

        if override == "auto":
            candidates = providers
        elif override == "native":
            candidates = [item for item in providers if item.name.startswith("native_")]
        else:
            candidates = [item for item in providers if item.name == override]

        if not candidates:
            available = ", ".join(item.name for item in providers)
            raise LookupError(
                f"provider {override!r} is not registered for {operator!r}; "
                f"available providers: {available}"
            )

        supported = [item for item in candidates if item.supports(**capabilities)]
        if not supported:
            names = ", ".join(item.name for item in candidates)
            raise RuntimeError(
                f"no requested provider supports {operator!r} with "
                f"capabilities {capabilities!r}; considered: {names}"
            )
        return max(supported, key=lambda item: item.priority)


REGISTRY = OperatorRegistry()


def register_operator(
    operator: str,
    name: str,
    *,
    supports: Supports = _supports_all,
    priority: int = 0,
) -> Callable[[ProviderFactory], ProviderFactory]:
    def decorator(factory: ProviderFactory) -> ProviderFactory:
        REGISTRY.register(
            operator,
            OperatorProvider(
                name=name,
                callable=factory,
                supports=supports,
                priority=priority,
            ),
        )
        return factory

    return decorator


class OperatorResolver:

    def __init__(
        self,
        overrides: Mapping[str, str] | None = None,
        *,
        registry: OperatorRegistry = REGISTRY,
        **capabilities: Any,
    ) -> None:
        self.overrides = dict(overrides or {})
        self.registry = registry
        self.capabilities = dict(capabilities)

    def bind(
        self,
        operator: str,
        owner: Any,
        **capabilities: Any,
    ) -> tuple[str, Callable]:
        effective_capabilities = {**self.capabilities, **capabilities}
        provider = self.registry.resolve(
            operator,
            self.overrides.get(operator, "auto"),
            **effective_capabilities,
        )
        return provider.name, provider.callable(owner)
