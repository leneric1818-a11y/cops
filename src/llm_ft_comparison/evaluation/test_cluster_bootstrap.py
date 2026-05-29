"""Sanity tests for cluster_bootstrap_netpw."""
from __future__ import annotations

from llm_ft_comparison.evaluation.cluster_bootstrap import cluster_bootstrap_netpw


def _rows(by_persona: dict[str, list[str]]) -> list[dict]:
    out: list[dict] = []
    for p, winners in by_persona.items():
        out.extend({"persona_name": p, "pairwise_winner": w} for w in winners)
    return out


def test_all_steered_gives_point_one():
    rows = _rows({f"p{i}": ["steered"] * 10 for i in range(5)})
    r = cluster_bootstrap_netpw(rows, n_boot=500)
    assert r["point"] == 1.0
    assert r["lo"] == 1.0 and r["hi"] == 1.0
    assert r["n_clusters"] == 5 and r["n_rows"] == 50


def test_all_base_gives_point_neg_one():
    rows = _rows({f"p{i}": ["base"] * 10 for i in range(5)})
    r = cluster_bootstrap_netpw(rows, n_boot=500)
    assert r["point"] == -1.0
    assert r["lo"] == -1.0 and r["hi"] == -1.0


def test_balanced_includes_zero():
    rows = _rows({f"p{i}": (["steered"] * 5) + (["base"] * 5) for i in range(10)})
    r = cluster_bootstrap_netpw(rows, n_boot=2000)
    assert abs(r["point"]) < 0.05
    assert r["lo"] <= 0.0 <= r["hi"]


def test_cluster_inflates_ci_vs_iid_intuition():
    """A heavily correlated-by-persona setup should give wide CIs.

    Every persona is either all-steered or all-base (extreme cluster
    correlation).  CI should span a large fraction of [-1, 1].
    """
    rows = _rows({f"p{i}": ["steered"] * 20 for i in range(5)} | {f"p{i}": ["base"] * 20 for i in range(5, 10)})
    r = cluster_bootstrap_netpw(rows, n_boot=2000, seed=42)
    assert abs(r["point"]) < 0.05  # 100 steered + 100 base = 0
    assert r["hi"] - r["lo"] > 0.5  # cluster bootstrap MUST be wide here


def test_n_clusters_and_n_rows_match_input():
    rows = _rows({"p0": ["steered"] * 7, "p1": ["base"] * 3, "p2": ["tie"] * 5})
    r = cluster_bootstrap_netpw(rows, n_boot=100)
    assert r["n_clusters"] == 3
    assert r["n_rows"] == 15


if __name__ == "__main__":
    for name in [n for n in globals() if n.startswith("test_")]:
        globals()[name]()
        print(f"PASS {name}")
