"""Count empty slices inside slots that have a series (partial slots) per coat96 cache.

A slot with no series is fully empty and counted separately; a partially filled slot means some picked slices
failed to read (e.g. the old tar repeated-slice bug) or were blank.
Usage (on the pod): python3 scripts/count_cache_gaps.py /workspace/cache-oai-* /workspace/cache-coat96-0
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee import coat96  # noqa: E402

bounds = np.cumsum([0] + [s[2] for s in coat96.SLOTS])
for d in sys.argv[1:]:
    m = np.load(f"{d}/masks.npy")
    part, lost, empty = [], [], []
    hit = np.zeros(len(m), bool)  # rows (knee-visits / studies) with at least one partial slot
    for i in range(len(coat96.SLOTS)):
        s = m[:, bounds[i]:bounds[i + 1]].sum(1)
        k = bounds[i + 1] - bounds[i]
        part.append(int(((s > 0) & (s < k)).sum()))
        hit |= (s > 0) & (s < k)
        lost.append(int((k - s[s > 0]).sum()))
        empty.append(int((s == 0).sum()))
    print(Path(d).name, len(m), "rows with a partial slot", int(hit.sum()), "partial", part, "lost", lost, "empty", empty)
