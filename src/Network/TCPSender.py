"""
TCPSender — sends MCP v0.1 JSON messages to Agent B over TCP.

Messages are newline-delimited JSON (one message per line).
A background thread keeps the connection alive and reconnects automatically
when the remote end drops or is not yet available.

When Agent B is unreachable, outgoing messages are held in an in-memory
queue (max_buffer messages). Once the connection is restored the background
thread drains the queue in order before processing new messages.
Oldest messages are dropped first when the buffer is full.
"""

import json
import logging
import queue
import socket
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_SENTINEL = object()  # signals the send loop to stop


class TCPSender:
    """
    Non-blocking TCP client for sending JSON messages.

    Usage:
        sender = TCPSender("192.168.1.100", 9000)
        sender.send({"type": "MCP.Detection", ...})   # fire and forget
        sender.close()
    """

    def __init__(
        self,
        host: str,
        port: int,
        retry_interval: float = 5.0,
        max_buffer: int = 256,
    ):
        self.host = host
        self.port = port
        self.retry_interval = retry_interval
        self.max_buffer = max_buffer

        self._sock: Optional[socket.socket] = None
        self._sock_lock = threading.Lock()
        self._running = True

        # FIFO queue shared between send() and the background send loop.
        # maxsize=0 means unbounded at the queue level; we enforce the cap
        # ourselves so we can drop the *oldest* entry rather than blocking.
        self._queue: queue.Queue = queue.Queue()

        # Single background thread handles both reconnecting and draining.
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(
            "TCPSender started — target %s:%d  buffer=%d",
            host, port, max_buffer,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, message: Dict[str, Any]) -> None:
        """
        Enqueue *message* for delivery to Agent B.

        Returns immediately. If the buffer is full the oldest queued message
        is discarded to make room (newest-wins policy).
        """
        if self._queue.qsize() >= self.max_buffer:
            try:
                dropped = self._queue.get_nowait()
                logger.warning(
                    "Buffer full (%d) — dropped oldest message: %s",
                    self.max_buffer,
                    dropped.get("type", "?") if isinstance(dropped, dict) else "?",
                )
            except queue.Empty:
                pass
        self._queue.put(message)

    def is_connected(self) -> bool:
        with self._sock_lock:
            return self._sock is not None

    def buffer_size(self) -> int:
        """Number of messages currently waiting to be sent."""
        return self._queue.qsize()

    def close(self):
        """Cleanly shut down — flushes the queue then stops."""
        self._running = False
        self._queue.put(_SENTINEL)  # unblock the loop if it's waiting
        self._thread.join(timeout=10)
        with self._sock_lock:
            self._close_locked()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _loop(self):
        """
        Single background loop:
          1. Ensure we have a socket (reconnect if needed).
          2. Block on the queue for the next message.
          3. Send it; on failure re-enqueue at the front and reconnect.
        """
        pending: Optional[bytes] = None  # serialised message awaiting retry

        while self._running:
            # -- ensure connection ------------------------------------------
            with self._sock_lock:
                connected = self._sock is not None
            if not connected:
                self._try_connect()
                with self._sock_lock:
                    connected = self._sock is not None
                if not connected:
                    time.sleep(self.retry_interval)
                    continue

            # -- get next payload -------------------------------------------
            if pending is None:
                try:
                    item = self._queue.get(timeout=self.retry_interval)
                except queue.Empty:
                    continue

                if item is _SENTINEL:
                    break

                pending = (
                    json.dumps(item, separators=(",", ":")) + "\n"
                ).encode("utf-8")

            # -- attempt send -----------------------------------------------
            with self._sock_lock:
                sock = self._sock
            try:
                sock.sendall(pending)
                logger.debug("Sent %d bytes", len(pending))
                pending = None  # success — clear for next message
            except OSError as exc:
                logger.warning("Send failed (%s) — will reconnect and retry", exc)
                with self._sock_lock:
                    self._close_locked()
                # pending stays set so the same message is retried after reconnect

    def _try_connect(self):
        try:
            sock = socket.create_connection((self.host, self.port), timeout=5)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._sock_lock:
                self._sock = sock
            logger.info(
                "Connected to Agent B at %s:%d  (queued=%d)",
                self.host, self.port, self._queue.qsize(),
            )
        except OSError as exc:
            logger.debug(
                "Connection to %s:%d failed: %s — retry in %.0fs",
                self.host, self.port, exc, self.retry_interval,
            )

    def _close_locked(self):
        """Close socket. Must be called with self._sock_lock held."""
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
