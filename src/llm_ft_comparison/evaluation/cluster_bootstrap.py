"""Cluster-robust bootstrap for net-pairwise preference (netpw).

Rolling-window contexts from the same source persona/case are not
independent — they share corpus-level stylistic and topical structure
that the bootstrap should respect.  We resample at the persona level
(10 clusters, each tied to one case per paper App.~C.1) and expand
back to context-level rows before computing the statistic.

Used by ``scripts/compute_cluster_bootstrap_headline.py`` to produce
CIs for the headline Table 1 (paper-v2 negative-alpha matrix).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np


def cluster_bootstrap_netpw(
    rows: Sequence[dict],
    cluster_field: str = "persona_name",
    winner_field: str = "pairwise_winner",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Cluster-bootstrap CI on netpw = (#steered - #base) / N.

    Parameters
    ----------
    rows
        Iterable of ensemble-judged rows. Each row must carry
        ``cluster_field`` (e.g. ``persona_name``) and ``winner_field``
        (one of ``"steered"``, ``"base"``, ``"tie"``).
    cluster_field
        Row key that identifies the cluster unit. Resampling happens
        with replacement over the set of distinct cluster ids.
    winner_field
        Row key that holds the pairwise winner string.
    n_boot, seed
        Number of bootstrap iterations and PRNG seed.

    Returns
    -------
    dict with ``mean``, ``lo``, ``hi`` (2.5% / 97.5% percentile),
    ``point`` (observed netpw on the original sample), ``n_rows``,
    and ``n_clusters``.
    """
    rng = np.random.default_rng(seed)

    by_cluster: dict[object, list[dict]] = defaultdict(list)
    for r in rows:
        by_cluster[r[cluster_field]].append(r)
    cluster_ids = sorted(by_cluster.keys(), key=str)
    if not cluster_ids:
        raise ValueError("no rows to bootstrap")

    def _netpw(winners: list[str]) -> float:
        c = Counter(winners)
        n = len(winners)
        return (c["steered"] - c["base"]) / max(1, n)

    point = _netpw([r[winner_field] for r in rows])

    cluster_ids_arr = np.array(cluster_ids, dtype=object)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        picked = rng.choice(cluster_ids_arr, size=len(cluster_ids_arr), replace=True)
        resampled_winners: list[str] = []
        for cid in picked:
            resampled_winners.extend(r[winner_field] for r in by_cluster[cid])
        stats[i] = _netpw(resampled_winners)

    lo, hi = np.quantile(stats, [0.025, 0.975])
    return {
        "point": float(point),
        "mean": float(np.mean(stats)),
        "lo": float(lo),
        "hi": float(hi),
        "n_rows": len(rows),
        "n_clusters": len(cluster_ids),
    }


def add_cluster_field_from_contexts(
    rows: Iterable[dict],
    contexts_path: str,
    field: str = "persona_name",
) -> list[dict]:
    """Annotate ensemble rows with ``field`` from ``cops_contexts.jsonl``.

    Joins by ``seed_id``.  Most ensemble rows already carry
    ``persona_name``, but this helper is here for cases where the join
    is needed (e.g. CI re-computation from raw judge outputs that omit
    it).
    """
    import json

    seed_to_field: dict[str, object] = {}
    with open(contexts_path, encoding="utf-8") as f:
        for line in f:
            ctx = json.loads(line)
            seed_to_field[ctx["seed_id"]] = ctx[field]

    annotated = []
    for r in rows:
        r2 = dict(r)
        if field not in r2:
            r2[field] = seed_to_field.get(r2["seed_id"])
        annotated.append(r2)
    return annotated
