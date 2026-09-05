import sys, os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from robot.serial_protocol import encode_command, encode_ping, encode_estop, parse_line


def test_encode_valid_command():
    assert encode_command("F") == b"CMD:F\n"
    assert encode_command("S") == b"CMD:S\n"


def test_encode_invalid_command_raises():
    with pytest.raises(ValueError):
        encode_command("X")


def test_encode_ping_and_estop():
    assert encode_ping() == b"PING\n"
    assert encode_estop() == b"ESTOP\n"


def test_parse_ack_line():
    msg_type, payload = parse_line("ACK:L\n")
    assert msg_type == "ACK"
    assert payload == "L"


def test_parse_pong_line():
    msg_type, payload = parse_line("PONG\n")
    assert msg_type == "PONG"
    assert payload is None


def test_parse_error_line():
    msg_type, payload = parse_line("ERR:invalid_command\n")
    assert msg_type == "ERR"
    assert payload == "invalid_command"


def test_parse_empty_line():
    msg_type, payload = parse_line("\n")
    assert msg_type is None
