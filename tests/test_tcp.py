"""
test_tcp.py — TCPSender unit tests

Spins up a local TCP server within the test process so everything
runs self-contained with no external dependencies.

Run:
    python tests/test_tcp.py
    pytest tests/test_tcp.py -v
"""

import sys
import json
import socket
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from Network.TCPSender import TCPSender

TEST_HOST = "127.0.0.1"
TEST_PORT = 19876   # chosen to avoid conflicts


def _header(n, title):
    print(f"\n{'='*70}")
    print(f"  TEST {n}: {title}")
    print(f"{'='*70}")

def _ok(msg):  print(f"  PASS  {msg}")


# ── mini TCP server ───────────────────────────────────────────────────────────

class _TestServer:
    """Newline-delimited JSON server for testing."""

    def __init__(self, host=TEST_HOST, port=TEST_PORT):
        self.host = host
        self.port = port
        self.received: list = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(5)
        self._sock.settimeout(5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle(self, conn):
        buf = b""
        conn.settimeout(2)
        try:
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line.strip():
                            self.received.append(json.loads(line.decode()))
                except socket.timeout:
                    break
        finally:
            conn.close()

    def stop(self):
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass

    def wait_for(self, count: int, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if len(self.received) >= count:
                return True
            time.sleep(0.05)
        return False


# ── tests ─────────────────────────────────────────────────────────────────────

def test_initialization():
    """1 — TCPSender can be created before any server is available."""
    _header(1, "Initialization without server")
    sender = TCPSender(host=TEST_HOST, port=19999, retry_interval=60)
    assert not sender.is_connected()
    _ok("Sender created, is_connected()=False as expected")
    sender.close()


def test_connects_when_server_starts():
    """2 — Sender connects automatically once a server is ready."""
    _header(2, "Auto-connect when server starts")
    server = _TestServer()
    try:
        sender = TCPSender(host=TEST_HOST, port=TEST_PORT, retry_interval=0.2)
        deadline = time.time() + 5.0
        while not sender.is_connected() and time.time() < deadline:
            time.sleep(0.1)
        assert sender.is_connected(), "Sender should connect within 5 seconds"
        _ok("Connected successfully")
    finally:
        sender.close()
        server.stop()


def test_send_single_message():
    """3 — A single JSON message is received correctly."""
    _header(3, "Send single message")
    server = _TestServer()
    sender = TCPSender(host=TEST_HOST, port=TEST_PORT, retry_interval=0.2)
    try:
        # wait for connection
        deadline = time.time() + 5.0
        while not sender.is_connected() and time.time() < deadline:
            time.sleep(0.1)

        msg = {"schema": "mcp.v0.1", "type": "MCP.Detection",
               "event_id": "evt_test001", "payload": {"label": "person", "confidence": 0.92}}
        ok = sender.send(msg)
        assert ok, "send() returned False"

        received = server.wait_for(1)
        assert received, "Message not received within timeout"
        assert server.received[0]["event_id"] == "evt_test001"
        assert server.received[0]["payload"]["label"] == "person"
        _ok(f"Message delivered: event_id={server.received[0]['event_id']}")
    finally:
        sender.close()
        server.stop()


def test_send_multiple_messages():
    """4 — Multiple messages arrive in order."""
    _header(4, "Send multiple messages in order")
    server = _TestServer()
    sender = TCPSender(host=TEST_HOST, port=TEST_PORT, retry_interval=0.2)
    try:
        deadline = time.time() + 5.0
        while not sender.is_connected() and time.time() < deadline:
            time.sleep(0.1)

        for i in range(5):
            sender.send({"seq": i, "type": "MCP.Detection"})

        received = server.wait_for(5)
        assert received, f"Expected 5 messages, got {len(server.received)}"
        seqs = [m["seq"] for m in server.received]
        assert seqs == list(range(5)), f"Out-of-order delivery: {seqs}"
        _ok(f"5 messages received in order: {seqs}")
    finally:
        sender.close()
        server.stop()


def test_send_without_connection_returns_false():
    """5 — send() returns False (not an exception) when not connected."""
    _header(5, "send() returns False when not connected")
    sender = TCPSender(host=TEST_HOST, port=19999, retry_interval=60)
    result = sender.send({"type": "MCP.Detection"})
    assert result is False
    _ok("send() returned False cleanly — no exception raised")
    sender.close()


def test_reconnect_after_server_restart():
    """6 — Sender reconnects when the server restarts."""
    _header(6, "Reconnect after server restart")
    server = _TestServer()
    sender = TCPSender(host=TEST_HOST, port=TEST_PORT, retry_interval=0.3)
    try:
        deadline = time.time() + 5.0
        while not sender.is_connected() and time.time() < deadline:
            time.sleep(0.1)
        assert sender.is_connected()
        _ok("Initial connection established")

        # Tear down server (simulates network drop)
        server.stop()
        time.sleep(0.5)

        # Restart server on same port
        server = _TestServer()
        deadline = time.time() + 6.0
        while not sender.is_connected() and time.time() < deadline:
            time.sleep(0.2)
        assert sender.is_connected(), "Should reconnect after server restart"
        _ok("Reconnected after server restart")
    finally:
        sender.close()
        server.stop()


def test_close_is_idempotent():
    """7 — Calling close() multiple times does not raise."""
    _header(7, "Idempotent close()")
    sender = TCPSender(host=TEST_HOST, port=19999, retry_interval=60)
    sender.close()
    sender.close()   # second call should not raise
    _ok("close() twice — no exception")


def test_message_is_valid_json_on_wire():
    """8 — Each message is newline-terminated valid JSON."""
    _header(8, "Wire format: newline-delimited JSON")
    raw_lines: list = []

    class _RawServer:
        def __init__(self):
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((TEST_HOST, TEST_PORT))
            self._sock.listen(1)
            self._sock.settimeout(5)
            self._stop = threading.Event()
            threading.Thread(target=self._run, daemon=True).start()

        def _run(self):
            try:
                conn, _ = self._sock.accept()
                conn.settimeout(3)
                buf = b""
                try:
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                except socket.timeout:
                    pass
                raw_lines.extend(buf.split(b"\n"))
                conn.close()
            except Exception:
                pass

        def stop(self):
            self._stop.set()
            try:
                self._sock.close()
            except OSError:
                pass

    raw_server = _RawServer()
    sender = TCPSender(host=TEST_HOST, port=TEST_PORT, retry_interval=0.2)
    try:
        deadline = time.time() + 5.0
        while not sender.is_connected() and time.time() < deadline:
            time.sleep(0.1)
        sender.send({"schema": "mcp.v0.1", "type": "MCP.Caption"})
        time.sleep(0.5)
    finally:
        sender.close()
        raw_server.stop()

    non_empty = [l for l in raw_lines if l.strip()]
    assert non_empty, "No data received on the wire"
    for line in non_empty:
        parsed = json.loads(line.decode())   # should not raise
        assert "type" in parsed
    _ok(f"Wire bytes are valid newline-delimited JSON: {non_empty[0].decode()[:60]}")


# ── runner ────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "#"*70)
    print("#" + " "*25 + "TCP SENDER TEST SUITE" + " "*22 + "#")
    print("#"*70)
    print(f"  Test server: {TEST_HOST}:{TEST_PORT}")

    tests = [
        ("initialization without server",       test_initialization),
        ("auto-connect when server starts",      test_connects_when_server_starts),
        ("send single message",                  test_send_single_message),
        ("send multiple messages in order",      test_send_multiple_messages),
        ("send() returns False when offline",    test_send_without_connection_returns_false),
        ("reconnect after server restart",       test_reconnect_after_server_restart),
        ("idempotent close()",                   test_close_is_idempotent),
        ("newline-delimited JSON wire format",   test_message_is_valid_json_on_wire),
    ]

    passed, failed, errors = 0, 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            failed += 1
            errors.append((name, exc))
            import traceback
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()
        time.sleep(0.1)   # small gap between tests to let sockets drain

    print(f"\n{'='*70}")
    print(f"  Results: {passed} passed, {failed} failed")
    if errors:
        for name, exc in errors:
            print(f"    - {name}: {exc}")
    print("="*70)
    return failed == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
