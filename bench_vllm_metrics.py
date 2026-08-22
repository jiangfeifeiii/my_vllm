"""Read-only exact counters for the eager vLLM comparison benchmark.

The observer subclasses vLLM's default asynchronous scheduler without changing
queue order, admission, allocation, or execution.  Counters are cumulative and
are read before and after the measured phase through EngineCore's existing
utility IPC, so no cross-process communication occurs in the timed hot path.
"""

from __future__ import annotations

import os
from types import MethodType
from typing import Any


SCHEDULER_CLASS = "bench_vllm_metrics.InstrumentedScheduler"
SNAPSHOT_METHOD = "_nanovllm_benchmark_metrics_snapshot"
PROTOCOL_VERSION = 1


try:
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.engine.core import EngineCore
except ModuleNotFoundError as error:
    if error.name != "vllm":
        raise
    AsyncScheduler = object  # type: ignore[assignment,misc]
    EngineCore = None  # type: ignore[assignment,misc]
    _VLLM_AVAILABLE = False
else:
    _VLLM_AVAILABLE = True


class InstrumentedScheduler(AsyncScheduler):  # type: ignore[misc,valid-type]
    """Default AsyncScheduler plus definition-aligned benchmark counters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _VLLM_AVAILABLE:
            raise RuntimeError("InstrumentedScheduler requires vLLM")
        super().__init__(*args, **kwargs)
        self._benchmark_computed_prompt_tokens = 0
        self._benchmark_cached_block_evictions = 0
        self._benchmark_allocated_blocks = 0
        self._benchmark_preemptions = 0
        self._benchmark_scheduler_steps = 0
        self._benchmark_allocation_depth = 0
        self._install_block_pool_observer()

    def _install_block_pool_observer(self) -> None:
        block_pool = self.kv_cache_manager.block_pool
        if getattr(block_pool, "_nanovllm_benchmark_observer", False):
            raise RuntimeError("vLLM block-pool benchmark observer installed twice")
        block_pool._nanovllm_benchmark_observer = True
        self._benchmark_original_get_new_blocks = block_pool.get_new_blocks
        self._benchmark_original_maybe_evict = block_pool._maybe_evict_cached_block
        scheduler = self

        def get_new_blocks(pool: Any, num_blocks: int) -> Any:
            scheduler._benchmark_allocation_depth += 1
            try:
                blocks = scheduler._benchmark_original_get_new_blocks(num_blocks)
            finally:
                scheduler._benchmark_allocation_depth -= 1
            scheduler._benchmark_allocated_blocks += len(blocks)
            return blocks

        def maybe_evict(pool: Any, block: Any) -> bool:
            evicted = scheduler._benchmark_original_maybe_evict(block)
            if evicted and scheduler._benchmark_allocation_depth > 0:
                scheduler._benchmark_cached_block_evictions += 1
            return evicted

        block_pool.get_new_blocks = MethodType(get_new_blocks, block_pool)
        block_pool._maybe_evict_cached_block = MethodType(maybe_evict, block_pool)

    def schedule(self, *args: Any, **kwargs: Any) -> Any:
        output = super().schedule(*args, **kwargs)
        self._benchmark_scheduler_steps += 1
        return output

    def _update_after_schedule(self, scheduler_output: Any) -> None:
        prompt_work = 0
        for request_id, scheduled_tokens in (
            scheduler_output.num_scheduled_tokens.items()
        ):
            request = self.requests[request_id]
            remaining_prompt = max(
                request.num_prompt_tokens - request.num_computed_tokens,
                0,
            )
            prompt_work += min(scheduled_tokens, remaining_prompt)
        super()._update_after_schedule(scheduler_output)
        self._benchmark_computed_prompt_tokens += prompt_work

    def _preempt_request(self, request: Any, timestamp: float) -> None:
        super()._preempt_request(request, timestamp)
        self._benchmark_preemptions += 1

    def benchmark_metrics_snapshot(self) -> dict[str, int | str | bool]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "observer_pid": os.getpid(),
            "async_scheduler": isinstance(self, AsyncScheduler),
            "block_pool_observer_installed": bool(
                getattr(
                    self.kv_cache_manager.block_pool,
                    "_nanovllm_benchmark_observer",
                    False,
                )
            ),
            "computed_prompt_tokens": self._benchmark_computed_prompt_tokens,
            "cached_block_eviction_count": self._benchmark_cached_block_evictions,
            "allocated_block_count": self._benchmark_allocated_blocks,
            "preemption_count": self._benchmark_preemptions,
            "scheduler_step_count": self._benchmark_scheduler_steps,
        }


def _engine_core_metrics_snapshot(engine_core: Any) -> dict[str, Any]:
    scheduler = engine_core.scheduler
    if not isinstance(scheduler, InstrumentedScheduler):
        raise RuntimeError("vLLM benchmark scheduler observer is not active")
    return scheduler.benchmark_metrics_snapshot()


if _VLLM_AVAILABLE:
    if hasattr(EngineCore, SNAPSHOT_METHOD):
        raise RuntimeError(f"EngineCore utility method collision: {SNAPSHOT_METHOD}")
    setattr(EngineCore, SNAPSHOT_METHOD, _engine_core_metrics_snapshot)
