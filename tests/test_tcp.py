"""
Tests for TCPSender — buffered non-blocking TCP sender.

Spins up a local TCP server within the test process so everything
runs self-contained with no external dependencies.
"""

import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from Network.TCPSender import TCPSender


class MiniServer:
    """Minimal TCP server that accumulates received JSON lines."""

    def __init__(self, host="127.0.0.1", port=0):
        self.host = host
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(1)
        self.sock.settimeout(5)
        self.messages = []
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self):
        while self._running:
            try:
                conn, _ = self.sock.accept()
                conn.settimeout(2)
                buf = b""
                while self._running:
                    try:
                        data = conn.recv(4096)
                        if not data:
                            break
                        buf += data
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            if line.strip():
                                self.messages.append(json.loads(line))
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                conn.close()
            except socket.timeout:
                continue
            except OSError:
                break

    def wait_for(self, count, timeout=10):
        deadline = time.time() + timeout
        while len(self.messages) < count and time.time() < deadline:
            time.sleep(0.1)
        return len(self.messages) >= count

    def close(self):
        self._running = False
        self.sock.close()
        self._thread.join(timeout=3)


# ── Initialization ───────────────────────────────────────────────

def test_init_and_buffer_starts_empty():
    sender = TCPSender("127.0.0.1", 59999, retry_interval=0.5)
    assert sender.buffer_size() == 0
    sender.close()


# ── send() + delivery ────────────────────────────────────────────

def test_send_single_message():
    server = MiniServer()
    sender = TCPSender(server.host, server.port, retry_interval=0.5)
    try:
        msg = {"type": "MCP.Detection", "label": "person"}
        sender.send(msg)
        assert server.wait_for(1)
        assert server.messages[0]["label"] == "person"
    finally:
        sender.close()
        server.close()


def test_send_multiple_messages():
    server = MiniServer()
    sender = TCPSender(server.host, server.port, retry_interval=0.5)
    try:
        for i in range(5):
            sender.send({"seq": i})
        assert server.wait_for(5)
        received_seqs = [m["seq"] for m in server.messages]
        assert received_seqs == [0, 1, 2, 3, 4]
    finally:
        sender.close()
        server.close()


def test_message_is_valid_json():
    server = MiniServer()
    sender = TCPSender(server.host, server.port, retry_interval=0.5)
    try:
        sender.send({"key": "value", "number": 42})
        assert server.wait_for(1)
        msg = server.messages[0]
        assert msg["key"] == "value"
        assert msg["number"] == 42
    finally:
        sender.close()
        server.close()


# ── Buffering when server is down ────────────────────────────────

def test_buffers_when_no_server():
    sender = TCPSender("127.0.0.1", 59998, retry_interval=0.5)
    try:
        sender.send({"buffered": True})
        time.sleep(0.2)
        assert sender.buffer_size() >= 1
    finally:
        sender.close()


def test_buffer_overflow_drops_oldest():
    sender = TCPSender("127.0.0.1", 59997, retry_interval=60, max_buffer=5)
    try:
        for i in range(10):
            sender.send({"seq": i})
        time.sleep(0.2)
        assert sender.buffer_size() <= 5
    finally:
        sender.close()


# ── Reconnection ────────────────────────────────────────────────

def test_drains_buffer_on_reconnect():
    sender = TCPSender("127.0.0.1", 59996, retry_interval=0.5)
    try:
        for i in range(3):
            sender.send({"seq": i})
        time.sleep(0.3)
        assert sender.buffer_size() >= 1

        server = MiniServer(port=59996)
        assert server.wait_for(3, timeout=10)
        assert len(server.messages) == 3
        server.close()
    finally:
        sender.close()


# ── close() ──────────────────────────────────────────────────────

def test_close_is_idempotent():
    sender = TCPSender("127.0.0.1", 59995, retry_interval=0.5)
    sender.close()
    sender.close()  # should not raise
