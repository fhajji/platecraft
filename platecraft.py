#!/usr/bin/env python3
"""PlateCraft entry point. Usage: python platecraft.py <command> [options]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from platecraft.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
