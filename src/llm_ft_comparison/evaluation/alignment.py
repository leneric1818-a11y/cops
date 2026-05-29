"""Geometric alignment metrics for comparing LoReFT-learned subspaces to
externally-derived concept directions (e.g. persona / steering vectors).

All functions accept either ``numpy.ndarray`` or ``torch.Tensor`` inputs and
return plain Python floats / numpy arrays. Subspace inputs are expected as
``(rank, embed_dim)`` row-stacked matrices; single directions as
``(embed_dim,)`` vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


ArrayLike = "np.ndarray | object"  # accept torch tensors too


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    # Accept torch tensors without importing torch at module load time.
    detach = getattr(x, "detach", None)
    if detach is not None:
        return detach().cpu().numpy()
    return np.asarray(x)


def _ensure_2d(R) -> np.ndarray:
    """Return a 2-D ``(rank, embed_dim)`` view of ``R``.

    A 1-D vector is treated as a rank-1 subspace.
    """
    arr = _to_numpy(R)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D array, got shape {arr.shape}.")
    return arr.astype(np.float64, copy=False)


def orthonormal_basis(R, atol: float = 1e-10) -> np.ndarray:
    """Return an orthonormal basis spanning the rows of ``R``.

    Uses an SVD to drop near-zero directions; the output has shape
    ``(rank_eff, embed_dim)`` where ``rank_eff`` may be smaller than
    ``R.shape[0]`` if rows are linearly dependent.
    """
    R2 = _ensure_2d(R)
    # SVD of R^T -> columns of U span the same subspace as rows of R.
    U, S, _ = np.linalg.svd(R2.T, full_matrices=False)
    keep = S > atol * (S.max() if S.size else 1.0)
    return U[:, keep].T  # (rank_eff, embed_dim)


def cosine(u, v) -> float:
    """Cosine similarity between two 1-D vectors."""
    a = _to_numpy(u).astype(np.float64).ravel()
    b = _to_numpy(v).astype(np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def subspace_projection_ratio(R, v) -> float:
    """Fraction of ``v``'s squared norm that lies inside ``span(R)``.

    Returns ``||P_R v|| / ||v||`` in ``[0, 1]``. ``1.0`` means ``v`` lies
    fully in ``span(R)``; ``0.0`` means ``v`` is orthogonal to it.
    """
    Q = orthonormal_basis(R)  # (rank, embed_dim)
    v_arr = _to_numpy(v).astype(np.float64).ravel()
    nv = np.linalg.norm(v_arr)
    if nv == 0.0:
        return 0.0
    coeff = Q @ v_arr  # (rank,)
    proj_norm = float(np.linalg.norm(coeff))
    return proj_norm / nv


def principal_angles(R_a, R_b) -> np.ndarray:
    """Principal angles (in radians, ascending) between ``span(R_a)`` and
    ``span(R_b)``.

    Computed via SVD of ``Q_a @ Q_b.T`` where ``Q_*`` are orthonormal bases.
    Returns an array of length ``min(rank_a, rank_b)``.
    """
    Qa = orthonormal_basis(R_a)
    Qb = orthonormal_basis(R_b)
    M = Qa @ Qb.T  # (rank_a, rank_b)
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    # Largest singular value -> smallest principal angle.
    sv_desc = np.sort(sv)[::-1]
    return np.arccos(sv_desc)  # ascending angles


def grassmann_distance(R_a, R_b) -> float:
    """Grassmann distance ``sqrt(sum theta_i^2)`` between two subspaces.

    Zero means the subspaces coincide; ``pi/2 * sqrt(min(rank))`` is the
    maximum (orthogonal) distance.
    """
    angles = principal_angles(R_a, R_b)
    return float(np.sqrt(np.sum(angles ** 2)))


def chordal_distance(R_a, R_b) -> float:
    """Chordal (projection F-norm) distance between two subspaces.

    ``sqrt(sum sin^2 theta_i)``. Bounded in ``[0, sqrt(min(rank))]``.
    """
    angles = principal_angles(R_a, R_b)
    return float(np.sqrt(np.sum(np.sin(angles) ** 2)))


def orthogonal_complement_within(R, v) -> np.ndarray:
    """Return the component of ``R`` orthogonal to direction ``v``.

    Used by the cross-steering validation to test whether LoReFT encodes
    *more* than the persona direction.
    """
    R2 = _ensure_2d(R)
    v_arr = _to_numpy(v).astype(np.float64).ravel()
    nv2 = float(np.dot(v_arr, v_arr))
    if nv2 == 0.0:
        return R2.copy()
    coeffs = (R2 @ v_arr) / nv2  # (rank,)
    return R2 - np.outer(coeffs, v_arr)


# ---------------------------------------------------------------------------
# Null-distribution baselines


def random_orthogonal(rank: int, embed_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a uniform-random ``(rank, embed_dim)`` matrix with orthonormal
    rows (Haar-distributed on the Stiefel manifold)."""
    if rank > embed_dim:
        raise ValueError("rank must be <= embed_dim")
    A = rng.standard_normal((embed_dim, rank))
    Q, _ = np.linalg.qr(A)
    return Q.T  # (rank, embed_dim)


@dataclass
class NullStats:
    mean: float
    std: float
    p_value: float  # one-sided: P(null >= observed)
    z_score: float
    samples: np.ndarray


def null_distribution(
    metric_fn,
    rank: int,
    embed_dim: int,
    observed: float,
    *,
    n_samples: int = 200,
    seed: int = 0,
    higher_is_more_aligned: bool = True,
    fixed_arg=None,
) -> NullStats:
    """Build a null distribution of ``metric_fn(R_random, fixed_arg)``.

    ``metric_fn`` must accept ``(R_random, fixed_arg)`` and return a scalar.
    For metrics where higher means more aligned (cosine, subspace_projection),
    set ``higher_is_more_aligned=True``; for distances set ``False``.
    """
    rng = np.random.default_rng(seed)
    samples = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        R_rand = random_orthogonal(rank, embed_dim, rng)
        samples[i] = float(metric_fn(R_rand, fixed_arg))
    mean = float(samples.mean())
    std = float(samples.std(ddof=1)) if n_samples > 1 else 0.0
    if higher_is_more_aligned:
        p_value = float((samples >= observed).mean())
    else:
        p_value = float((samples <= observed).mean())
    z_score = float((observed - mean) / std) if std > 0 else 0.0
    return NullStats(mean=mean, std=std, p_value=p_value, z_score=z_score, samples=samples)


# ---------------------------------------------------------------------------
# Convenience wrappers used by analyze_reft_persona_alignment.py


def alignment_report(R, v, *, n_null_samples: int = 200, seed: int = 0) -> dict:
    """Compute the standard alignment-vs-direction battery and return a dict.

    Includes:
      * primary_cosine: cos(top-PC of R, v)
      * subspace_projection_ratio
      * grassmann_distance / chordal_distance against ``span(v)``
      * null statistics for primary_cosine and subspace_projection_ratio
    """
    R2 = _ensure_2d(R)
    v_arr = _to_numpy(v).astype(np.float64).ravel()
    embed_dim = R2.shape[1]
    rank = R2.shape[0]
    if v_arr.shape[0] != embed_dim:
        raise ValueError(
            f"R has embed_dim={embed_dim} but v has length {v_arr.shape[0]}"
        )

    # Top principal direction of R (unit vector)
    Q = orthonormal_basis(R2)
    if Q.shape[0] == 0:
        return {
            "rank_effective": 0,
            "primary_cosine": 0.0,
            "subspace_projection_ratio": 0.0,
            "grassmann_distance": float(np.pi / 2),
            "chordal_distance": 1.0,
            "null": {},
        }
    top_pc = Q[0]
    primary_cos = cosine(top_pc, v_arr)
    proj_ratio = subspace_projection_ratio(R2, v_arr)
    gd = grassmann_distance(R2, v_arr)
    cd = chordal_distance(R2, v_arr)

    null_proj = null_distribution(
        lambda Rr, vv: subspace_projection_ratio(Rr, vv),
        rank=rank,
        embed_dim=embed_dim,
        observed=proj_ratio,
        n_samples=n_null_samples,
        seed=seed,
        higher_is_more_aligned=True,
        fixed_arg=v_arr,
    )
    null_cos = null_distribution(
        lambda Rr, vv: abs(cosine(Rr[0], vv)),
        rank=rank,
        embed_dim=embed_dim,
        observed=abs(primary_cos),
        n_samples=n_null_samples,
        seed=seed + 1,
        higher_is_more_aligned=True,
        fixed_arg=v_arr,
    )

    return {
        "rank_effective": int(Q.shape[0]),
        "primary_cosine": primary_cos,
        "subspace_projection_ratio": proj_ratio,
        "grassmann_distance": gd,
        "chordal_distance": cd,
        "null": {
            "subspace_projection_ratio": {
                "mean": null_proj.mean,
                "std": null_proj.std,
                "p_value": null_proj.p_value,
                "z_score": null_proj.z_score,
            },
            "absolute_primary_cosine": {
                "mean": null_cos.mean,
                "std": null_cos.std,
                "p_value": null_cos.p_value,
                "z_score": null_cos.z_score,
            },
        },
    }


def pairwise_subspace_report(subspaces: dict) -> dict:
    """Cross-rank stability check: principal angles and grassmann distance
    between every pair of given subspaces.

    ``subspaces`` is a mapping ``label -> (rank, embed_dim) array``. Returns
    a nested dict keyed by ``(label_a, label_b)`` with ``principal_angles``
    and ``grassmann_distance`` entries.
    """
    labels = list(subspaces.keys())
    out: dict = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            angles = principal_angles(subspaces[a], subspaces[b])
            out[f"{a}__vs__{b}"] = {
                "principal_angles_rad": angles.tolist(),
                "grassmann_distance": float(np.sqrt(np.sum(angles ** 2))),
                "chordal_distance": float(np.sqrt(np.sum(np.sin(angles) ** 2))),
            }
    return out


__all__ = [
    "cosine",
    "orthonormal_basis",
    "subspace_projection_ratio",
    "principal_angles",
    "grassmann_distance",
    "chordal_distance",
    "orthogonal_complement_within",
    "random_orthogonal",
    "null_distribution",
    "NullStats",
    "alignment_report",
    "pairwise_subspace_report",
]
