"""
robot/serial_protocol.py

Simple line-based text protocol over serial:

  Host -> Robot:  "CMD:<letter>\n"      e.g. "CMD:L\n"
  Host -> Robot:  "PING\n"              heartbeat
  Host -> Robot:  "ESTOP\n"             emergency stop, highest priority

  Robot -> Host:  "ACK:<letter>\n"      command acknowledged
  Robot -> Host:  "PONG\n"              heartbeat reply
  Robot -> Host:  "ERR:<reason>\n"      malformed/unsupported command

Valid command letters: F B L R S
"""

VALID_COMMAND_LETTERS = {"F", "B", "L", "R", "S"}


def encode_command(letter: str) -> bytes:
    if letter not in VALID_COMMAND_LETTERS:
        raise ValueError(f"Invalid robot command letter: {letter!r}")
    return f"CMD:{letter}\n".encode("ascii")


def encode_ping() -> bytes:
    return b"PING\n"


def encode_estop() -> bytes:
    return b"ESTOP\n"


def parse_line(line: str):
    """Returns (msg_type, payload) e.g. ('ACK', 'L'), ('PONG', None), ('ERR', 'reason')."""
    line = line.strip()
    if not line:
        return None, None
    if line.startswith("ACK:"):
        return "ACK", line.split(":", 1)[1]
    if line == "PONG":
        return "PONG", None
    if line.startswith("ERR:"):
        return "ERR", line.split(":", 1)[1]
    return "UNKNOWN", line
