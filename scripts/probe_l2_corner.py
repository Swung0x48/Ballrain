"""Measure rail approach speed at 1780; switch to the known 8 u/s tracker there."""

import sys
from pathlib import Path
from probe_l2_chokes import probe

if __name__ == "__main__":
    for cap in map(float, sys.argv[3:]):
        probe(Path(sys.argv[1]), sys.argv[2], 1700, f"cap{cap:g}", True, corner_cap=cap)
