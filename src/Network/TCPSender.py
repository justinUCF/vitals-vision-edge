"""
TCPSender — sends MCP v0.1 JSON messages to Agent B over TCP.

Messages are newline-delimited JSON (one message per line).
A background thread keeps the connection alive and reconnects automatically
when the remote end drops or is not yet available.
"""

import json
import logging
import socket
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TCPSender:
    """
    Non-blocking TCP client for sending JSON messages.

    Usage:
        sender = TCPSender("192.168.1.100", 9000)
        sender.send({"type": "MCP.Detection", ...})   # fire and forget
        sender.close()
    """

    def __init__(self, host: str, port: int, retry_interval: float = 5.0):
        self.host = host
        self.port = port
        self.retry_interval = retry_interval

        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._running = True

        # Background thread keeps the connection live
        self._thread = threading.Thread(target=self._reconnect_loop, daemon=True)
        self._thread.start()
        logger.info("TCPSender started — target %s:%d", host, port)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, message: Dict[str, Any]) -> bool:
        """
        Serialize *message* as JSON and write it to the TCP socket.

        Returns True on success.  If there is no active connection the
        message is dropped and False is returned — the reconnect loop
        will restore connectivity for the next message.
        """
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            if self._sock is None:
                logger.debug("No TCP connection — message dropped")
                return False
            try:
                self._sock.sendall(payload)
                return True
            except OSError as exc:
                logger.warning("Send failed (%s) — will reconnect", exc)
                self._close_locked()
                return False

    def is_connected(self) -> bool:
        with self._lock:
            return self._sock is not None

    def close(self):
        """Cleanly shut down the sender."""
        self._running = False
        with self._lock:
            self._close_locked()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reconnect_loop(self):
        while self._running:
            with self._lock:
                already_connected = self._sock is not None
            if not already_connected:
                self._try_connect()
            time.sleep(self.retry_interval)

    def _try_connect(self):
        try:
            sock = socket.create_connection((self.host, self.port), timeout=5)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                self._sock = sock
            logger.info("Connected to Agent B at %s:%d", self.host, self.port)
        except OSError as exc:
            logger.debug(
                "Connection to %s:%d failed: %s — retry in %.0fs",
                self.host, self.port, exc, self.retry_interval,
            )

    def _close_locked(self):
        """Close socket. Must be called with self._lock held (or during shutdown)."""
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
