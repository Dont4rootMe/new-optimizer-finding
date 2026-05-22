"""Minimal seed program for the ShinkaEvolve baseline on circle_packing_shinka.

The ShinkaEvolve runner edits only the region between EVOLVE-BLOCK-START and
EVOLVE-BLOCK-END. The rest of the file stays fixed.

The host evaluator (UnitSquare26CirclePackingExperiment) imports `run_packing`
from this file (renamed to `implementation.py` in the per-trial workspace) and
expects it to return `(centers, radii, sum_of_radii)`.
"""

from __future__ import annotations

import numpy as np

NUM_CIRCLES = 26
SQUARE_SIZE = 1.0


# EVOLVE-BLOCK-START
def run_packing():
    grid_n = 6
    coords = np.linspace(0.1, 0.9, grid_n)
    centers = np.array([(float(x), float(y)) for x in coords for y in coords])[:NUM_CIRCLES]
    # spacing between grid points = 0.16; tangent radius = spacing/2 = 0.08.
    # Use 0.078 to leave a tiny non-overlap margin so the evaluator's
    # overlap-tolerance check passes on the seed baseline.
    radii = np.full(NUM_CIRCLES, 0.078, dtype=float)
    return centers, radii, float(radii.sum())
# EVOLVE-BLOCK-END
