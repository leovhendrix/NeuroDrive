"""
main.py

Convenience entry point. This just dispatches to the individual scripts —
each is also runnable directly.

    python main.py collect --simulation
    python main.py train
    python main.py run --simulation
    python main.py run --robot-port /dev/ttyUSB0
"""
import argparse
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description="BCI robot-control system")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("collect", add_help=False)
    sub.add_parser("train", add_help=False)
    sub.add_parser("run", add_help=False)

    args, remaining = parser.parse_known_args()

    script_map = {
        "collect": "training/collect_data.py",
        "train": "training/train_bci.py",
        "run": "realtime_bci.py",
    }

    sys.argv = [script_map[args.cmd]] + remaining
    runpy.run_path(script_map[args.cmd], run_name="__main__")


if __name__ == "__main__":
    main()
