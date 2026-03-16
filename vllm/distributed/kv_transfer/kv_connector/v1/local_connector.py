# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import pickle
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import zmq
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

Transfer = tuple[int, float]  # (xfer_handle, start_time)
EngineId = str
ReqId = str
logger = init_logger(__name__)


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    # To be used when logical block size does not match the kernel block size
    local_physical_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str
    tp_size: int
    is_send: bool


@dataclass
class LocalAgentHandshakeMetadata(KVConnectorHandshakeMetadata):
    engine_id: str
    # agent_metadata: bytes
    # kv_caches_base_addr: list[int]
    # device_id: int
    # num_blocks: int
    # block_lens: list[int]
    # attn_backend_name: str
    # kv_cache_layout: str
    # block_size: int
    kvcache_ipcs: dict[str, str]


class LocalConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.requests: dict[str, ReqMeta] = dict()

    def add_request(self, request_id: str, req: ReqMeta):
        self.requests[request_id] = req


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


class LocalConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.vllm_config = vllm_config
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id
        self._kv_caches: dict[str, torch.Tensor] = dict()
        self.copy_operation = None
        self.remote_kvcaches: dict[str, dict[str, torch.Tensor]] = dict()

        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_rank
        )
        self._handshake_listener_t = None
        self._requests: dict[str, Request] = dict()
        self._reqs_need_recv: dict[str, ReqMeta] = dict()
        self._reqs_need_send: dict[str, ReqMeta] = dict()

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args:
            kv_caches: dictionary of layer names, kv cache
        """
        logger.debug("inited kvcache: %s", len(kv_caches))
        self._kv_caches = kv_caches
        return

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """
        Set the xPU-specific ops for copying KV between host and device.
        Needed when host buffer is used for kv transfer (e.g., in NixlConnector)
        """
        self.copy_operation = copy_operation
        return

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        assert isinstance(self._connector_metadata, LocalConnectorMetadata)
        for req_id, req in self._connector_metadata.requests.items():
            if req.is_send:
                continue
            self.init_remote_kvcaches(
                req.remote_engine_id,
                req.remote_host,
                req.remote_port,
            )
            remote_kvcache = self.remote_kvcaches[req.remote_engine_id]
            # for layer, cache in remote_kvcache.items():
            self.copy_operation(
                remote_kvcache,
                self._kv_caches,
                req.remote_block_ids,
                req.local_block_ids,
                "h2d",
            )
            logger.debug("start load kv %s", req_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        pass

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the Executors) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        logger.debug("get_finished %s", finished_req_ids)
        assert isinstance(self._connector_metadata, LocalConnectorMetadata)
        sent = set(
            [
                req_id
                for (req_id, req) in self._connector_metadata.requests.items()
                if req.is_send
            ]
        )
        received = set(
            [
                req_id
                for (req_id, req) in self._connector_metadata.requests.items()
                if not req.is_send
            ]
        )
        self._connector_metadata.requests.clear()
        self._reqs_need_recv.clear()
        self._reqs_need_send.clear()
        return sent, received

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.

        Notes:
            - Applies to both sync- and async-loading requests.
            - Async loading: failed blocks may be reported in any forward pass
              up to and including the pass where the request ID is returned by
              `get_finished()`. Even if failures occur, the request must still
              be reported via `get_finished()`, and the failed block IDs must
              appear here no later than that same pass.
            - Sync loading: failed blocks should be reported in the forward
              pass in which they are detected.
        """
        return set()

    def shutdown(self):
        """
        Shutdown the connector. This is called when the worker process
        is shutting down to ensure that all the async operations are
        completed and the connector is cleaned up properly.
        """
        return None

    def get_kv_connector_stats(self) -> Optional["KVConnectorStats"]:
        """
        Get the KV connector stats collected during the last interval.
        """
        return None

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get the KVConnector handshake metadata for this connector.
        This metadata is used for out-of-band connector handshake
        between P/D workers.

        Returns:
            KVConnectorHandshakeMetadata: the handshake metadata.
            None if no handshake metadata is available.
        """
        logger.debug("get_handshake_metadata")

        def get_ipc(t: torch.Tensor):
            fn, args = reduce_tensor(t)
            return pickle.dumps(args)

        kvcache_ipcs = {
            layer_name: get_ipc(cache) for layer_name, cache in self._kv_caches.items()
        }
        return LocalAgentHandshakeMetadata(
            engine_id=self.engine_id,
            kvcache_ipcs=kvcache_ipcs,
        )

    def _handshake_listener(
        self,
        port: int,
        ready_event: threading.Event,
        handshake_data: LocalAgentHandshakeMetadata,
    ):
        """Background thread for getting new NIXL handshakes."""
        # NOTE(rob): this is a simple implementation. We will move
        # to a better approach via HTTP endpoint soon.

        # Listen for new requests for metadata.
        host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        path = make_zmq_path("tcp", host, port)
        logger.debug("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ready_event.set()
            while True:
                try:
                    identity, _, msg = sock.recv_multipart()
                except zmq.Again:
                    continue
                # Decode the message which contains (GET_META_MSG, rank)
                if msg != b"get_kvcache_ipcs":
                    logger.warning("Connection listener got unexpected message %s", msg)
                logger.debug(
                    "receive handshake: %s, hand: %s", identity, handshake_data
                )
                sock.send_multipart((identity, b"", pickle.dumps(handshake_data)))

    def init_remote_kvcaches(self, remote_engine_id, remote_host, remote_port):
        if remote_engine_id in self.remote_kvcaches:
            return
        path = make_zmq_path("tcp", remote_host, remote_port)
        logger.debug(
            "Querying metadata on path: %s at remote tp rank",
            path,
        )

        # Send query for the request.
        with zmq_ctx(zmq.REQ, path) as sock:
            msg = "get_kvcache_ipcs"
            # Set receive timeout to 5 seconds to avoid hanging on dead server
            sock.setsockopt(zmq.RCVTIMEO, 5000)  # milliseconds
            sock.send_string(msg)
            metadata_bytes = sock.recv()
            metadata = pickle.loads(metadata_bytes)
            assert isinstance(metadata, LocalAgentHandshakeMetadata)
            logger.debug(
                "receive %s handshake metadata: %s", remote_engine_id, metadata
            )

            def from_ipc(ipc):
                args = pickle.loads(ipc)
                return rebuild_cuda_tensor(*args)

            assert metadata.engine_id == remote_engine_id
            kvcaches_dict = {
                layer_name: from_ipc(ipc)
                for layer_name, ipc in metadata.kvcache_ipcs.items()
            }
            self.remote_kvcaches[remote_engine_id] = kvcaches_dict

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - An optional number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.

        Notes:
            The connector should only consider the largest prefix of prompt-
            tokens for which KV cache is actually available at the time of the
            call. If the cache cannot be loaded for some tokens (e.g., due to
            connectivity issues or eviction), those tokens must not be taken
            into account.
        """

        params = request.kv_transfer_params
        logger.debug(
            "LocalConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            count = len(token_ids) - num_computed_tokens

            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.

        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens that will be
                loaded from the external KV cache.
        """
        if params := request.kv_transfer_params:
            local_block_ids = (
                blocks.get_unhashed_block_ids() if num_external_tokens > 0 else []
            )
            req = ReqMeta(
                local_block_ids=local_block_ids,
                local_physical_block_ids=local_block_ids,
                remote_block_ids=params["remote_block_ids"],
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
                remote_engine_id=params["remote_engine_id"],
                tp_size=1,
                is_send=False,
            )
            logger.debug("update_state_after_alloc %s", request.request_id)
            if params["do_remote_prefill"] and num_external_tokens:
                self._requests[request.request_id] = request
                self._reqs_need_recv[request.request_id] = req
            elif params["do_remote_decode"]:
                req.is_send = True
                self._reqs_need_send[request.request_id] = req

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = LocalConnectorMetadata()
        meta.requests = self._reqs_need_recv | self._reqs_need_send
        logger.debug("build metadata %d", len(meta.requests))
        self._reqs_need_recv.clear()
        self._reqs_need_send.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector request_finished(%s), request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params:
            return False, None

        if request.request_id in self._requests:
            del self._requests[request.request_id]
        if request.request_id in self._reqs_need_recv:
            del self._reqs_need_recv[request.request_id]
        if request.request_id in self._reqs_need_send:
            del self._reqs_need_recv[request.request_id]

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode"):
            return False, None
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            logger.debug(
                "NIXLConnector request_finished(%s) waiting for %d seconds "
                "for remote decode to fetch blocks",
                request.request_id,
                envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT,
            )
        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
        )

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """
        Get the required KV cache layout for this connector.
        Args:
            vllm_config (VllmConfig): the vllm config.

        Returns:
            str: the required KV cache layout. e.g. HND, or NHD.
            None if the connector does not require a specific layout.
        """

        return "HND"

    def get_finished_count(self) -> int | None:
        """
        Get the count of requests expected to complete send/receive operations
        via this connector. This method is used to initialize the
        KVOutputAggregator, overwriting the default world_size.

        Returns:
            int: expected sending or receiving completion count.
        """

        return None

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> Optional["KVConnectorStats"]:
        """
        KVConnectorStats resolution method. This method allows dynamically
        registered connectors to return their own KVConnectorStats object,
        which can implement custom aggregation logic on the data dict.
        """
        return None

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (KVConnectorHandshakeMetadata): the handshake metadata to set.
        """
        logger.debug(f"set_xfer_handshake_metadata {metadata}")
        if self._handshake_listener_t is None:
            ready_event = threading.Event()
            self._nixl_handshake_listener_t = threading.Thread(
                target=self._handshake_listener,
                args=(
                    self.side_channel_port,
                    ready_event,
                    metadata[0],
                ),
                daemon=True,
                name="local_handshake_listener",
            )
            self._nixl_handshake_listener_t.start()
            ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[str]],
    ) -> Optional["KVConnectorPromMetrics"]:
        """
        Create a KVConnectorPromMetrics subclass which should register
        per-connector Prometheus metrics and implement observe() to
        expose connector transfer stats via Prometheus.
        """
        return None
