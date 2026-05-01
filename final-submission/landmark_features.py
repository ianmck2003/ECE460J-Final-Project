# Wrist-centered hand landmark features plus extra geometry terms to help tell R, U, and V apart.
# Base features are 63 dims, model input is 68 (63 + 5 geometry extras).

from __future__ import annotations

import numpy as np

N_BASE = 63
N_AUX = 5
FEATURE_DIM = N_BASE + N_AUX


def base_vector_from_mediapipe(lm) -> list[float]:
    # Wrist-centered normalization, scaled to max absolute value.
    wrist_x, wrist_y, wrist_z = lm[0].x, lm[0].y, lm[0].z
    coords = []
    for point in lm:
        coords += [
            point.x - wrist_x,
            point.y - wrist_y,
            point.z - wrist_z,
        ]
    max_val = max(abs(v) for v in coords) or 1.0
    return [float(v / max_val) for v in coords]


def _as_points(coords63) -> np.ndarray:
    return np.asarray(coords63, dtype=np.float64).reshape(21, 3)


def max_normalize_points(P: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(P)) or 1.0)
    return P / m


def mirror_base_x(base63) -> np.ndarray:
    # Flips the hand left/right by negating all x components.
    x = np.asarray(base63, dtype=np.float32).copy()
    if x.ndim == 1:
        x[0::3] *= -1.0
    else:
        x[:, 0::3] *= -1.0
    return x


def random_pose_augment(
    base63,
    rng: np.random.Generator,
    *,
    noise_std: float = 0.018,
    max_angle_rad: float = 0.38,
) -> np.ndarray:
    # Applies a small random rotation and noise, then re-normalizes.
    P = _as_points(base63).copy()
    theta = float(rng.uniform(-max_angle_rad, max_angle_rad))
    c, s = np.cos(theta), np.sin(theta)
    x0 = P[:, 0] * c - P[:, 1] * s
    x1 = P[:, 0] * s + P[:, 1] * c
    P[:, 0] = x0
    P[:, 1] = x1
    P += rng.normal(0.0, noise_std, P.shape)
    P = max_normalize_points(P)
    return P.reshape(63).astype(np.float32)


def expand_features(base63) -> np.ndarray:
    # Appends 5 geometry features between the index and middle fingers to help separate R, U, and V.
    P = _as_points(base63)
    p5, p7, p8 = P[5], P[7], P[8]
    p9, p11, p12 = P[9], P[11], P[12]

    def nrm(v: np.ndarray) -> np.ndarray:
        l = np.linalg.norm(v)
        return v / (l + 1e-8)

    v_idx = p8 - p5
    v_mid = p12 - p9
    cos_mcp_to_tip = float(np.clip(np.dot(nrm(v_idx), nrm(v_mid)), -1.0, 1.0))

    w_idx = p8 - p7
    w_mid = p12 - p11
    cos_distal = float(np.clip(np.dot(nrm(w_idx), nrm(w_mid)), -1.0, 1.0))

    diff = p8 - p12
    dist_tips = float(np.linalg.norm(diff))
    dist_tips_xy = float(np.linalg.norm(diff[:2]))
    # Signed 2D "cross" (index vs middle MCP→tip): spread vs parallel vs crossed.
    cp_xy = float(v_idx[0] * v_mid[1] - v_idx[1] * v_mid[0])

    aux = np.array(
        [dist_tips, dist_tips_xy, cos_mcp_to_tip, cos_distal, cp_xy],
        dtype=np.float32,
    )
    base = np.asarray(base63, dtype=np.float32)
    return np.concatenate([base, aux], axis=0)


def expand_feature_matrix(X63: np.ndarray) -> np.ndarray:
    n = X63.shape[0]
    out = np.empty((n, FEATURE_DIM), dtype=np.float32)
    for i in range(n):
        out[i] = expand_features(X63[i])
    return out
