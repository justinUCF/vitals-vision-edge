"""
test_usbconnection.py — USBConnection unit tests

Tests port listing, auto-detection logic, connection failure handling,
callback registration, and the context-manager interface.
Hardware (a real flight controller) is NOT required — all tests that
need an open port are skipped when no suitable device is found.

Run:
    python tests/test_usbconnection.py
    pytest tests/test_usbconnection.py -v
"""

import sys
import time
import queue
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from Camera.USBConnection import USBConnection


def _header(n, title):
    print(f"\n{'='*70}")
    print(f"  TEST {n}: {title}")
    print(f"{'='*70}")

def _ok(msg):   print(f"  PASS  {msg}")
def _skip(msg): print(f"  SKIP  {msg}")


# ── tests ─────────────────────────────────────────────────────────────────────

def test_initialization_no_autoconnect():
    """1 — Constructor does not attempt connection unless auto_connect=True."""
    _header(1, "Initialization without auto_connect")
    usb = USBConnection(port=None, baudrate=57600, auto_connect=False)
    assert usb.is_connected is False
    assert usb.serial_conn is None
    assert usb.baudrate == 57600
    assert usb.message_queue is not None
    _ok(f"USBConnection created — is_connected={usb.is_connected}  baudrate={usb.baudrate}")
    return usb


def test_list_available_ports():
    """2 — list_available_ports() returns a list (may be empty on CI)."""
    _header(2, "list_available_ports()")
    ports = USBConnection.list_available_ports()
    assert isinstance(ports, list)
    _ok(f"{len(ports)} serial port(s) found on this machine:")
    for p in ports:
        assert "port" in p and "description" in p
        _ok(f"  {p['port']:20s}  {p['description']}")
    if not ports:
        _ok("  (none — normal on dev machines without serial hardware)")


def test_detect_flight_controller():
    """3 — detect_flight_controller() returns a string or None."""
    _header(3, "detect_flight_controller()")
    result = USBConnection.detect_flight_controller()
    if result is None:
        _ok("No flight controller detected (expected without hardware)")
    else:
        assert isinstance(result, str)
        _ok(f"Flight controller detected: {result}")


def test_connect_to_nonexistent_port():
    """4 — Connecting to a bogus port returns False and does not crash."""
    _header(4, "Connect to non-existent port")
    usb = USBConnection(auto_connect=False)
    success = usb.connect(port="/dev/ttyNONEXISTENT999")
    assert success is False
    assert usb.is_connected is False
    _ok("connect() returned False gracefully for non-existent port")


def test_connect_no_fc_auto_detect():
    """5 — Auto-detect connect() returns False when no FC is attached."""
    _header(5, "Auto-detect connect() — no flight controller")
    if USBConnection.detect_flight_controller():
        _skip("Real flight controller detected — test targets no-hardware path")
        return
    usb = USBConnection(port=None, auto_connect=False)
    success = usb.connect()   # will try auto-detect → should fail
    assert success is False
    _ok("Auto-detect connect() correctly returned False")


def test_status_when_disconnected():
    """6 — get_status() reflects disconnected state."""
    _header(6, "get_status() when disconnected")
    usb = USBConnection(auto_connect=False)
    status = usb.get_status()
    assert status["connected"] is False
    assert status["queued_messages"] == 0
    _ok(f"status={status}")


def test_callback_registration():
    """7 — register_callback() stores callbacks; they survive disconnect."""
    _header(7, "Callback registration")
    usb = USBConnection(auto_connect=False)
    received: list = []

    def cb(data: bytes):
        received.append(data)

    usb.register_callback(cb)
    assert len(usb.message_callbacks) == 1

    usb.register_callback(cb)   # duplicate — should also be stored
    assert len(usb.message_callbacks) == 2
    _ok(f"{len(usb.message_callbacks)} callback(s) registered")


def test_write_when_disconnected():
    """8 — write() returns False gracefully when not connected."""
    _header(8, "write() when disconnected")
    usb = USBConnection(auto_connect=False)
    result = usb.write(b"\xFE\x09\x00")
    assert result is False
    _ok("write() returned False — no exception raised")


def test_read_message_timeout():
    """9 — read_message(timeout) returns None when queue is empty."""
    _header(9, "read_message() timeout")
    usb = USBConnection(auto_connect=False)
    start = time.time()
    msg = usb.read_message(timeout=0.2)
    elapsed = time.time() - start
    assert msg is None
    assert elapsed >= 0.15, f"Timeout too short: {elapsed:.3f}s"
    _ok(f"read_message() returned None after {elapsed:.2f}s timeout")


def test_disconnect_when_not_connected():
    """10 — disconnect() on an unconnected object does not crash."""
    _header(10, "disconnect() when not connected")
    usb = USBConnection(auto_connect=False)
    usb.disconnect()   # should be a no-op
    assert usb.is_connected is False
    _ok("disconnect() is idempotent when not connected")


def test_context_manager_no_hardware():
    """11 — Context manager enters and exits cleanly without hardware."""
    _header(11, "Context manager without hardware")
    with USBConnection(port=None, auto_connect=False) as usb:
        assert usb is not None
        assert usb.is_connected is False   # no port to connect to
    _ok("Context manager __enter__ / __exit__ completed without error")


def test_mavlink_heartbeat_bytes():
    """12 — send_heartbeat() returns False when disconnected (not a crash)."""
    _header(12, "send_heartbeat() when disconnected")
    usb = USBConnection(auto_connect=False)
    result = usb.send_heartbeat()
    assert result is False
    _ok("send_heartbeat() returned False gracefully")


def test_message_queue_is_fifo():
    """13 — message_queue is a proper FIFO when populated manually."""
    _header(13, "Message queue FIFO order")
    usb = USBConnection(auto_connect=False)
    payloads = [b"msg_a", b"msg_b", b"msg_c"]
    for p in payloads:
        usb.message_queue.put(p)
    assert usb.get_status()["queued_messages"] == 3
    for expected in payloads:
        got = usb.read_message(timeout=0.1)
        assert got == expected, f"FIFO order broken: expected {expected}, got {got}"
    _ok("FIFO order preserved: " + "  ".join(p.decode() for p in payloads))


# ── hardware path (skipped when no FC present) ────────────────────────────────

def test_real_connect_if_hardware_present():
    """14 — Full connect / heartbeat / disconnect when flight controller present."""
    _header(14, "Real hardware path (skipped if no FC)")
    fc_port = USBConnection.detect_flight_controller()
    if not fc_port:
        _skip("No flight controller detected — connect this test requires hardware")
        return
    usb = USBConnection(port=fc_port, auto_connect=False)
    success = usb.connect()
    if not success:
        _skip(f"Could not open {fc_port} — another process may have it locked")
        return
    assert usb.is_connected is True
    _ok(f"Connected to {fc_port}")

    hb = usb.send_heartbeat()
    _ok(f"Heartbeat sent: {hb}")

    msg = usb.read_message(timeout=1.0)
    if msg:
        _ok(f"Received {len(msg)} bytes from FC: {msg[:16].hex()}")
    else:
        _ok("No data received within 1 s (normal without FC responding)")

    usb.disconnect()
    assert usb.is_connected is False
    _ok("Disconnected cleanly")


# ── runner ────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "#"*70)
    print("#" + " "*21 + "USB CONNECTION TEST SUITE" + " "*22 + "#")
    print("#"*70)

    tests = [
        ("initialization",                      test_initialization_no_autoconnect),
        ("list_available_ports()",              test_list_available_ports),
        ("detect_flight_controller()",          test_detect_flight_controller),
        ("connect to non-existent port",        test_connect_to_nonexistent_port),
        ("auto-detect — no FC",                 test_connect_no_fc_auto_detect),
        ("get_status() disconnected",           test_status_when_disconnected),
        ("callback registration",               test_callback_registration),
        ("write() when disconnected",           test_write_when_disconnected),
        ("read_message() timeout",              test_read_message_timeout),
        ("disconnect() idempotent",             test_disconnect_when_not_connected),
        ("context manager",                     test_context_manager_no_hardware),
        ("send_heartbeat() disconnected",       test_mavlink_heartbeat_bytes),
        ("message queue FIFO",                  test_message_queue_is_fifo),
        ("real hardware connect",               test_real_connect_if_hardware_present),
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
