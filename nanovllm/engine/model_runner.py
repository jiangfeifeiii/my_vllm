import pickle
import warnings
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist

from nanovllm.config import CUDAGraphPolicy, Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.attention_backend import (
    FlashInferAttentionBackend,
    LegacyFlashAttentionBackend,
)
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.custom_op import (
    CustomOpConfig,
    load_optional_implementations,
)
from nanovllm.utils.context import (
    BatchType,
    RuntimeExecutionMode,
    get_context,
    reset_context,
    set_context,
)
from nanovllm.utils.loader import load_model


# PyTorch 2.11 and FlashInfer 0.6.17 cannot safely initialize a second set of
# graph-aware paged wrappers after the first graph engine is torn down.  A new
# process is the isolation boundary for another capture session.
_FULL_DECODE_GRAPH_CAPTURE_ATTEMPTED = False


def _claim_full_decode_graph_capture_session() -> bool:
    global _FULL_DECODE_GRAPH_CAPTURE_ATTEMPTED
    if _FULL_DECODE_GRAPH_CAPTURE_ATTEMPTED:
        return False
    _FULL_DECODE_GRAPH_CAPTURE_ATTEMPTED = True
    return True


@dataclass
class DecodeGraphState:
    batch_size: int
    page_indices_capacity: int
    wrapper: Any
    cuda_graph: torch.cuda.CUDAGraph | None
    static_input_ids: torch.Tensor
    static_positions: torch.Tensor
    static_slot_mapping: torch.Tensor
    static_page_q_indptr: torch.Tensor
    static_page_kv_indptr: torch.Tensor
    static_page_indices: torch.Tensor
    static_page_last_page_len: torch.Tensor
    static_attention_output: torch.Tensor
    static_hidden_output: torch.Tensor


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.cudagraph_policy = config.cudagraph_mode
        self.decode_graph_states: dict[int, DecodeGraphState] = {}
        self._graph_pool = None
        self.cudagraph_capture_time_ms = 0.0
        self.cudagraph_extra_memory_bytes = 0
        self._cudagraph_stats = {
            "full_graph_replay_steps": 0,
            "eager_fallback_steps": 0,
            "graph_bucket_hits": 0,
            "graph_bucket_misses": 0,
        }
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.dtype = getattr(hf_config, "dtype", None)
        if self.dtype is None:
            self.dtype = hf_config.torch_dtype

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        torch.set_default_device("cuda")
        self.optional_provider_errors = load_optional_implementations()
        self.custom_op_config = CustomOpConfig(
            platform="cuda",
            dtype=self.dtype,
            overrides=config.operator_overrides,
        )
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        backend_kwargs = {
            "num_q_heads": hf_config.num_attention_heads // self.world_size,
            "num_kv_heads": hf_config.num_key_value_heads // self.world_size,
            "head_dim": head_dim,
            "block_size": self.block_size,
            "dtype": self.dtype,
        }
        if config.attention_backend == "flashinfer":
            self.attention_backend = FlashInferAttentionBackend(
                **backend_kwargs,
                attention_mode=config.attention_mode,
            )
        else:
            self.attention_backend = LegacyFlashAttentionBackend(
                **backend_kwargs,
            )
        self.full_decode_graph_capable = (
            self.cudagraph_policy is CUDAGraphPolicy.FULL_DECODE_ONLY
            and self.world_size == 1
            and config.attention_mode == "unified"
            and self.attention_backend.supports_full_decode_graph
            and any(
                size <= min(config.max_num_seqs, config.max_num_batched_tokens)
                for size in config.cudagraph_batch_sizes
            )
        )
        if (
            self.full_decode_graph_capable
            and not _claim_full_decode_graph_capture_session()
        ):
            warnings.warn(
                "a full-decode CUDA Graph capture session already ran in "
                "this process; falling back to Eager. Start a new process "
                "to capture another graph engine.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.full_decode_graph_capable = False
        self.model = Qwen3ForCausalLM(
            hf_config,
            attention_backend=self.attention_backend,
            custom_op_config=self.custom_op_config,
        )
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()

        graph_state_memory = 0
        if self.full_decode_graph_capable:
            torch.cuda.synchronize()
            before_states = torch.cuda.memory_allocated()
            self.initialize_decode_graph_states()
            graph_state_memory = max(
                0,
                torch.cuda.memory_allocated() - before_states,
            )

        # Graph capture happens after the KV pool exists.  Reserve at least
        # the already-measured graph-state footprint so capture cannot silently
        # push the process above gpu_memory_utilization.  The normal eager
        # activation headroom remains accounted for by allocate_kv_cache().
        self.allocate_kv_cache(
            capture_reserve_bytes=graph_state_memory,
        )
        graph_capture_memory = 0
        if self.decode_graph_states:
            torch.cuda.synchronize()
            before_capture = torch.cuda.memory_allocated()
            capture_start = perf_counter()
            self.capture_full_decode_graphs()
            torch.cuda.synchronize()
            self.cudagraph_capture_time_ms = (
                perf_counter() - capture_start
            ) * 1000
            graph_capture_memory = max(
                0,
                torch.cuda.memory_allocated() - before_capture,
            )
        self.cudagraph_extra_memory_bytes = (
            graph_state_memory + graph_capture_memory
        )
        self.reset_cudagraph_stats()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # 尝试清理已存在的共享内存
                try:
                    existing_shm = SharedMemory(name="nanovllm", create=False)
                    existing_shm.close()
                    existing_shm.unlink()  # 标记删除
                    print(f"Cleaned up existing shared memory: nanovllm")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"Error cleaning up shared memory: {e}")

                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        torch.cuda.synchronize()
        # Destroy graph execs while their wrappers and static buffers are alive.
        for state in self.decode_graph_states.values():
            if state.cuda_graph is not None:
                state.cuda_graph.reset()
                state.cuda_graph = None
        self.decode_graph_states.clear()
        self._graph_pool = None
        torch.cuda.empty_cache()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        warmup_tokens = min(
            max_num_batched_tokens,
            self.config.max_num_seqs * max_model_len,
        )
        num_full_seqs, remainder = divmod(warmup_tokens, max_model_len)
        sequence_lengths = [max_model_len] * num_full_seqs
        if remainder:
            sequence_lengths.append(remainder)
        seqs = [
            Sequence([0] * length, block_size=self.block_size)
            for length in sequence_lengths
        ]
        for seq in seqs:
            seq.num_new_tokens = len(seq)
        self.run(seqs)
        torch.cuda.empty_cache()

    def initialize_decode_graph_states(self) -> None:
        config = self.config
        max_decode_batch = min(
            config.max_num_seqs,
            config.max_num_batched_tokens,
        )
        max_pages_per_request = (
            config.max_model_len + self.block_size - 1
        ) // self.block_size
        device = torch.device("cuda", torch.cuda.current_device())
        hidden_size = config.hf_config.hidden_size
        num_q_heads = self.attention_backend.num_q_heads
        head_dim = self.attention_backend.head_dim

        for batch_size in config.cudagraph_batch_sizes:
            if batch_size > max_decode_batch:
                continue
            page_indices_capacity = batch_size * max_pages_per_request
            static_page_q_indptr = torch.empty(
                batch_size + 1, dtype=torch.int32, device=device
            )
            static_page_kv_indptr = torch.empty_like(static_page_q_indptr)
            static_page_indices = torch.empty(
                page_indices_capacity, dtype=torch.int32, device=device
            )
            static_page_last_page_len = torch.empty(
                batch_size, dtype=torch.int32, device=device
            )
            wrapper = self.attention_backend.create_full_decode_graph_wrapper(
                static_page_q_indptr,
                static_page_kv_indptr,
                static_page_indices,
                static_page_last_page_len,
            )
            self.decode_graph_states[batch_size] = DecodeGraphState(
                batch_size=batch_size,
                page_indices_capacity=page_indices_capacity,
                wrapper=wrapper,
                cuda_graph=None,
                static_input_ids=torch.empty(
                    batch_size, dtype=torch.int64, device=device
                ),
                static_positions=torch.empty(
                    batch_size, dtype=torch.int64, device=device
                ),
                static_slot_mapping=torch.empty(
                    batch_size, dtype=torch.int32, device=device
                ),
                static_page_q_indptr=static_page_q_indptr,
                static_page_kv_indptr=static_page_kv_indptr,
                static_page_indices=static_page_indices,
                static_page_last_page_len=static_page_last_page_len,
                static_attention_output=torch.empty(
                    batch_size,
                    num_q_heads,
                    head_dim,
                    dtype=self.dtype,
                    device=device,
                ),
                static_hidden_output=torch.empty(
                    batch_size,
                    hidden_size,
                    dtype=self.dtype,
                    device=device,
                ),
            )

    def reset_cudagraph_stats(self) -> None:
        for name in self._cudagraph_stats:
            self._cudagraph_stats[name] = 0

    def get_cudagraph_stats(self) -> dict:
        return {
            **self._cudagraph_stats,
            "policy": self.cudagraph_policy.value,
            "captured_batch_sizes": sorted(self.decode_graph_states),
            "capture_time_ms": self.cudagraph_capture_time_ms,
            "extra_memory_bytes": self.cudagraph_extra_memory_bytes,
        }

    def allocate_kv_cache(self, capture_reserve_bytes: int = 0):
        config = self.config
        hf_config = config.hf_config
        if capture_reserve_bytes < 0:
            raise ValueError("capture_reserve_bytes must be non-negative")
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * self.dtype.itemsize
        available_bytes = int(
            total * config.gpu_memory_utilization
            - used
            - peak
            + current
            - capture_reserve_bytes
        )
        config.num_kvcache_blocks = available_bytes // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables


    def prepare_model_input(
        self,
        seqs: list[Sequence],
        num_prefill_seqs: int | None = None,
    ):
        if not seqs:
            raise ValueError("cannot prepare an empty execution batch")
        if num_prefill_seqs is None:
            num_prefill_seqs = len(seqs)
        assert 0 <= num_prefill_seqs <= len(seqs)
        if num_prefill_seqs == 0:
            batch_type = BatchType.PURE_DECODE
        elif num_prefill_seqs == len(seqs):
            batch_type = BatchType.PURE_PREFILL
        else:
            batch_type = BatchType.MIXED
        num_prefill_tokens = sum(
            seq.num_new_tokens for seq in seqs[:num_prefill_seqs]
        )
        num_decode_tokens = sum(
            seq.num_new_tokens for seq in seqs[num_prefill_seqs:]
        )
        assert all(
            seq.num_new_tokens == 1
            for seq in seqs[num_prefill_seqs:]
        ), "decode suffix must contain one query token per sequence"

        use_flashinfer = self.config.attention_backend == "flashinfer"
        has_paged_cache = any(seq.block_table for seq in seqs)
        if has_paged_cache and not all(seq.block_table for seq in seqs):
            raise ValueError(
                "execution batches cannot mix paged and cacheless sequences"
            )
        use_page_metadata = use_flashinfer and has_paged_cache
        # FlashInfer falls back to ragged FlashAttention only during cacheless
        # model-memory warmup. Paged serving does not consume legacy KV lengths,
        # context lengths, or a dense block table.
        use_legacy_metadata = not use_flashinfer or not has_paged_cache

        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0] if use_legacy_metadata else None
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        context_lens = [] if use_legacy_metadata else None
        page_kv_indptr = [0] if use_page_metadata else None
        page_indices = [] if use_page_metadata else None
        page_last_page_len = [] if use_page_metadata else None
        seq_need_compute_logits = []
        for seq_index, seq in enumerate(seqs):
            if len(seq) == seq.num_cached_tokens + seq.num_new_tokens and seq.block_table:
                seq_need_compute_logits.append(seq_index)
            if context_lens is not None:
                context_lens.append(seq.num_context_tokens)
            input_ids.extend(seq[seq.num_cached_tokens: seq.num_context_tokens])
            positions.extend(list(range(seq.num_cached_tokens, seq.num_context_tokens)))
            seqlen_q = seq.num_new_tokens
            seqlen_k = seq.num_context_tokens
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            if cu_seqlens_k is not None:
                cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if page_indices is not None:
                assert page_kv_indptr is not None
                assert page_last_page_len is not None
                page_indices.extend(seq.block_table)
                page_kv_indptr.append(
                    page_kv_indptr[-1] + len(seq.block_table)
                )
                last_page_len = seq.num_context_tokens % self.block_size
                page_last_page_len.append(last_page_len or self.block_size)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, len(seq.block_table)):
                if i == seq.num_cached_blocks:
                    start = seq.block_table[i] * self.block_size + seq.num_cached_tokens % seq.block_size
                else:
                    start = seq.block_table[i] * self.block_size
                if i == len(seq.block_table) - 1:
                    end = seq.block_table[i] * self.block_size + seq.num_context_tokens % self.block_size \
                        if seq.num_context_tokens % self.block_size != 0 \
                            else (seq.block_table[i] + 1) * self.block_size
                else:
                    end = (seq.block_table[i] + 1) * self.block_size
                slot_mapping.extend(list(range(start, end)))
        if (
            cu_seqlens_k is not None
            and cu_seqlens_k[-1] > cu_seqlens_q[-1]
        ):    # prefix cache or decoding
            block_tables = self.prepare_block_tables(seqs)
        num_pages = len(page_indices) if page_indices is not None else None
        num_prefill_pages = (
            page_kv_indptr[num_prefill_seqs]
            if page_kv_indptr is not None
            else None
        )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        if cu_seqlens_k is not None:
            cu_seqlens_k = torch.tensor(
                cu_seqlens_k, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        if context_lens is not None:
            context_lens = torch.tensor(
                context_lens, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
        if has_paged_cache:
            seq_need_compute_logits = torch.tensor(
                seq_need_compute_logits, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
        else:
            seq_need_compute_logits = None
        if page_kv_indptr is not None:
            page_kv_indptr = torch.tensor(
                page_kv_indptr, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            page_indices = torch.tensor(
                page_indices, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            page_last_page_len = torch.tensor(
                page_last_page_len, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
        set_context(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            seq_need_compute_logits=seq_need_compute_logits,
            page_q_indptr=cu_seqlens_q if use_page_metadata else None,
            page_kv_indptr=page_kv_indptr,
            page_indices=page_indices,
            page_last_page_len=page_last_page_len,
            page_metadata_trusted=use_page_metadata,
            num_pages=num_pages,
            num_prefill_pages=num_prefill_pages,
            num_prefill_seqs=num_prefill_seqs,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            batch_type=batch_type,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        context = get_context()
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        if context.seq_need_compute_logits is not None:
            temperatures = temperatures[context.seq_need_compute_logits]
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        context = get_context()
        runtime_mode, graph_state = self.select_runtime_mode(
            context,
            input_ids,
        )
        context.runtime_mode = runtime_mode
        if runtime_mode is RuntimeExecutionMode.EAGER:
            self.attention_backend.plan(context)
            hidden_states = self.model(input_ids, positions)
        else:
            assert graph_state is not None
            hidden_states = self.replay_full_decode_graph(
                graph_state,
                context,
                input_ids,
                positions,
            )
        return self.model.compute_logits(hidden_states)

    def select_runtime_mode(
        self,
        context,
        input_ids: torch.Tensor,
    ) -> tuple[RuntimeExecutionMode, DecodeGraphState | None]:
        def eager_fallback():
            self._cudagraph_stats["eager_fallback_steps"] += 1
            return RuntimeExecutionMode.EAGER, None

        if self.cudagraph_policy is CUDAGraphPolicy.NONE:
            return RuntimeExecutionMode.EAGER, None
        if context.batch_type is not BatchType.PURE_DECODE:
            return eager_fallback()
        if self.config.attention_mode != "unified":
            return eager_fallback()
        if not self.attention_backend.supports_full_decode_graph:
            return eager_fallback()
        if self.world_size != 1:
            return eager_fallback()
        if context.num_prefill_seqs != 0:
            return eager_fallback()
        if context.num_prefill_tokens != 0:
            return eager_fallback()

        q_indptr = context.page_q_indptr
        last_page_len = context.page_last_page_len
        if not isinstance(q_indptr, torch.Tensor):
            return eager_fallback()
        if not isinstance(last_page_len, torch.Tensor):
            return eager_fallback()
        num_requests = last_page_len.numel()
        if context.num_decode_tokens != num_requests:
            return eager_fallback()
        if input_ids.numel() != num_requests:
            return eager_fallback()
        if q_indptr.numel() != num_requests + 1:
            return eager_fallback()
        if (
            getattr(context, "page_metadata_trusted", False) is not True
            and not bool(
                torch.all(q_indptr[1:] - q_indptr[:-1] == 1).item()
            )
        ):
            return eager_fallback()

        graph_state = self.decode_graph_states.get(num_requests)
        if graph_state is None:
            self._cudagraph_stats["graph_bucket_misses"] += 1
            return eager_fallback()
        self._cudagraph_stats["graph_bucket_hits"] += 1
        if graph_state.cuda_graph is None:
            return eager_fallback()
        if not self.graph_metadata_fits(graph_state, context):
            return eager_fallback()
        return RuntimeExecutionMode.FULL_GRAPH, graph_state

    def graph_metadata_fits(
        self,
        state: DecodeGraphState,
        context,
    ) -> bool:
        batch_size = state.batch_size
        tensors = (
            context.page_q_indptr,
            context.page_kv_indptr,
            context.page_indices,
            context.page_last_page_len,
        )
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            return False
        if any(
            tensor.dtype != torch.int32
            or tensor.device.type != "cuda"
            or tensor.ndim != 1
            for tensor in tensors
        ):
            return False
        q_indptr, kv_indptr, indices, last_page_len = tensors
        if q_indptr.numel() != batch_size + 1:
            return False
        if kv_indptr.numel() != batch_size + 1:
            return False
        if last_page_len.numel() != batch_size:
            return False
        if indices.numel() > state.page_indices_capacity:
            return False
        if getattr(context, "page_metadata_trusted", False) is True:
            num_pages = getattr(context, "num_pages", None)
            num_prefill_pages = getattr(context, "num_prefill_pages", None)
            if type(num_pages) is not int or num_pages != indices.numel():
                return False
            if type(num_prefill_pages) is not int or num_prefill_pages != 0:
                return False
        else:
            if int(q_indptr[0].item()) != 0:
                return False
            if int(q_indptr[-1].item()) != batch_size:
                return False
            if int(kv_indptr[0].item()) != 0:
                return False
            if int(kv_indptr[-1].item()) != indices.numel():
                return False
            if not bool(torch.all(kv_indptr[1:] >= kv_indptr[:-1]).item()):
                return False
            valid_last_page = (last_page_len > 0) & (
                last_page_len <= self.block_size
            )
            if not bool(torch.all(valid_last_page).item()):
                return False
            if indices.numel() and (
                int(indices.min().item()) < 0
                or int(indices.max().item())
                >= self.config.num_kvcache_blocks
            ):
                return False
        slot_mapping = context.slot_mapping
        if not isinstance(slot_mapping, torch.Tensor):
            return False
        if (
            slot_mapping.dtype != torch.int32
            or slot_mapping.device.type != "cuda"
            or slot_mapping.ndim != 1
            or slot_mapping.numel() != batch_size
        ):
            return False
        return True

    def replay_full_decode_graph(
        self,
        state: DecodeGraphState,
        context,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        self.attention_backend.plan_full_decode_graph(state.wrapper, context)
        state.static_input_ids.copy_(input_ids)
        state.static_positions.copy_(positions)
        state.static_slot_mapping.copy_(context.slot_mapping)
        assert state.cuda_graph is not None
        state.cuda_graph.replay()
        self._cudagraph_stats["full_graph_replay_steps"] += 1
        return state.static_hidden_output

    def run(
        self,
        seqs: list[Sequence],
        num_prefill_seqs: int | None = None,
    ):
        input_ids, positions = self.prepare_model_input(
            seqs,
            num_prefill_seqs=num_prefill_seqs,
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        seq_need_compute_logits = get_context().seq_need_compute_logits
        reset_context()
        return token_ids, seq_need_compute_logits

    @torch.inference_mode()
    def capture_full_decode_graphs(self) -> None:
        try:
            self._capture_full_decode_graphs()
        finally:
            self.attention_backend.deactivate_full_decode_graph()
            reset_context()

    @torch.inference_mode()
    def _capture_full_decode_graphs(self) -> None:
        if not self.decode_graph_states:
            return
        if self.kv_cache.size(2) < 1:
            raise RuntimeError("full decode graph capture requires one KV page")

        # Synthetic capture requests share page zero and never write it because
        # their slot mapping is -1. This is startup-only metadata, not runtime
        # graph bucket padding.
        self.kv_cache[:, :, 0].zero_()
        device = self.kv_cache.device

        for batch_size in sorted(self.decode_graph_states, reverse=True):
            state = self.decode_graph_states[batch_size]
            state.static_input_ids.zero_()
            state.static_positions.zero_()
            state.static_slot_mapping.fill_(-1)

            capture_q_indptr = torch.arange(
                batch_size + 1,
                dtype=torch.int32,
                device=device,
            )
            capture_kv_indptr = torch.arange(
                batch_size + 1,
                dtype=torch.int32,
                device=device,
            )
            capture_indices = torch.zeros(
                batch_size,
                dtype=torch.int32,
                device=device,
            )
            capture_last_page_len = torch.ones(
                batch_size,
                dtype=torch.int32,
                device=device,
            )
            set_context(
                cu_seqlens_q=capture_q_indptr,
                max_seqlen_q=1,
                max_seqlen_k=1,
                slot_mapping=state.static_slot_mapping,
                page_q_indptr=capture_q_indptr,
                page_kv_indptr=capture_kv_indptr,
                page_indices=capture_indices,
                page_last_page_len=capture_last_page_len,
                page_metadata_trusted=True,
                num_pages=batch_size,
                num_prefill_pages=0,
                num_prefill_seqs=0,
                num_prefill_tokens=0,
                num_decode_tokens=batch_size,
                batch_type=BatchType.PURE_DECODE,
                runtime_mode=RuntimeExecutionMode.FULL_GRAPH,
            )
            self.attention_backend.plan_full_decode_graph(
                state.wrapper,
                get_context(),
            )

            # The graph sees only fixed-address state buffers populated by the
            # graph-aware FlashInfer wrapper's plan above.
            set_context(
                cu_seqlens_q=state.static_page_q_indptr,
                max_seqlen_q=1,
                max_seqlen_k=1,
                slot_mapping=state.static_slot_mapping,
                page_q_indptr=state.static_page_q_indptr,
                page_kv_indptr=state.static_page_kv_indptr,
                page_indices=state.static_page_indices[:batch_size],
                page_last_page_len=state.static_page_last_page_len,
                page_metadata_trusted=True,
                num_pages=batch_size,
                num_prefill_pages=0,
                num_prefill_seqs=0,
                num_prefill_tokens=0,
                num_decode_tokens=batch_size,
                batch_type=BatchType.PURE_DECODE,
                runtime_mode=RuntimeExecutionMode.FULL_GRAPH,
            )

            self.attention_backend.activate_full_decode_graph(
                state.wrapper,
                state.static_attention_output,
            )
            try:
                state.static_hidden_output.copy_(
                    self.model(
                        state.static_input_ids,
                        state.static_positions,
                    )
                )
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self._graph_pool):
                    state.static_hidden_output.copy_(
                        self.model(
                            state.static_input_ids,
                            state.static_positions,
                        )
                    )
                if self._graph_pool is None:
                    self._graph_pool = graph.pool()
                state.cuda_graph = graph
                torch.cuda.synchronize()
            finally:
                self.attention_backend.deactivate_full_decode_graph()
                reset_context()
