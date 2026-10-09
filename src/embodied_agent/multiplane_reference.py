"""Analytical axial-depth observability for known reference planes."""

import numpy as np


def rotation_vector(vector):
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    if angle == 0:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def plane_depth(rays, position, rotation, normal, offset):
    rays, position, rotation, normal = [
        np.asarray(v, dtype=float) for v in (rays, position, rotation, normal)
    ]
    if (
        rays.ndim != 2
        or rays.shape[1] != 3
        or position.shape != (3,)
        or rotation.shape != (3, 3)
        or normal.shape != (3,)
        or not all(np.isfinite(v).all() for v in (rays, position, rotation, normal))
        or not np.isfinite(offset)
        or not np.isclose(np.linalg.norm(normal), 1)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
    ):
        raise ValueError("invalid calibrated plane inputs")
    denominator = (rays @ rotation.T) @ normal
    if np.any(np.abs(denominator) < 1e-10):
        raise ValueError("ray parallel to reference")
    depth = (offset - normal @ position) / denominator
    if np.any(depth <= 0):
        raise ValueError("reference behind camera")
    return depth


def residuals(rays, planes, parameters):
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (7,) or not np.isfinite(parameters).all():
        raise ValueError("seven finite perturbations required")
    p = np.array([0.4, 0, 1.2])
    rotation = rotation_vector(parameters[3:6])
    return np.array(
        [
            plane_depth(rays, p, np.eye(3), n, d)
            + parameters[6]
            - plane_depth(rays, p + parameters[:3], rotation, n, d)
            for n, d in planes
        ]
    )


def observability(rays, planes):
    scales = np.array([0.001] * 3 + [np.deg2rad(0.1)] * 3 + [0.001])
    columns = []
    for i in range(7):
        delta = np.zeros(7)
        delta[i] = 1e-6
        columns.append(
            ((residuals(rays, planes, delta) - residuals(rays, planes, -delta)) / (2e-6)).ravel()
            * scales[i]
        )
    jacobian = np.array(columns).T
    _, singular, vectors = np.linalg.svd(jacobian, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-7))
    return {
        "rank": rank,
        "parameter_count": 7,
        "singular_values_scaled_m": singular.tolist(),
        "parameter_scales": scales.tolist(),
        "null_vectors_scaled_coordinates": vectors[rank:].tolist(),
    }
