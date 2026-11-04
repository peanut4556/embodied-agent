"""Bounded rigid reference alignment with an unused out-of-plane check point."""

import math

import numpy as np

from .reference_confirmation import ReferenceConfirmation, point3


class RigidReferenceConfirmation(ReferenceConfirmation):
    """Fit three references; the checkpoint never participates in the fit.

    All observations must be from the target capture. Catalog provenance and
    unmodelled local distortions remain outside this geometric check.
    """

    def __init__(self, start_time, reference_points, checkpoint):
        super().__init__(start_time, reference_points)
        if "checkpoint" in self.reference_points:
            raise ValueError("checkpoint must be separate from fitting references")
        self.checkpoint = point3(checkpoint)
        points = np.array(list(self.reference_points.values()))
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        height = abs(np.dot(np.array(self.checkpoint) - points[0], normal)) / np.linalg.norm(normal)
        if height < 0.015:
            raise ValueError("checkpoint must be at least 15 mm off the fitting plane")

    def correct(self, target, references):
        try:
            point = np.array(point3(target))
            if not isinstance(references, dict) or set(references) != {
                *self.reference_points,
                "checkpoint",
            }:
                raise ValueError("all fitting references and checkpoint required")
            expected = np.array(list(self.reference_points.values()))
            observed = np.array([point3(references[key]) for key in self.reference_points])
            center, measured_center = expected.mean(axis=0), observed.mean(axis=0)
            source, destination = observed - measured_center, expected - center
            if (
                min(
                    np.linalg.svd(source, compute_uv=False)[1],
                    np.linalg.svd(destination, compute_uv=False)[1],
                )
                < 0.03
            ):
                raise ValueError("reference geometry too narrow for rigid alignment")
            u, _, vt = np.linalg.svd(source.T @ destination)
            rotation = vt.T @ u.T
            if np.linalg.det(rotation) < 0:
                vt[-1] *= -1
                rotation = vt.T @ u.T
            angle = math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
            if angle > 5:
                raise ValueError("rotation exceeds five-degree budget")
            if np.linalg.norm(measured_center - center) > 0.03:
                raise ValueError("reference center translation exceeds budget")
            fitted = source @ rotation.T + center
            residual = float(np.linalg.norm(fitted - expected, axis=1).max())
            if residual > 0.003:
                raise ValueError("rigid fitting residual exceeds budget")
            check_point = np.array(point3(references["checkpoint"]))
            check_residual = float(
                np.linalg.norm(
                    rotation @ (check_point - measured_center) + center - self.checkpoint
                )
            )
            if check_residual > 0.003:
                raise ValueError("independent checkpoint mismatch")
            corrected = rotation @ (point - measured_center) + center
            radius = float(np.linalg.norm(corrected - center))
            if radius > 0.25:
                raise ValueError("target outside validated reference neighborhood")
            if np.linalg.norm(corrected - point) > 0.03:
                raise ValueError("target correction exceeds budget")
            if not np.isfinite(corrected).all():
                raise ValueError("nonfinite rigid correction")
            return tuple(corrected.tolist()), {
                "accepted": True,
                "rotation_degrees": angle,
                "rotation": rotation.tolist(),
                "observed_center": measured_center.tolist(),
                "reference_center": center.tolist(),
                "max_residual": residual,
                "checkpoint_residual": check_residual,
                "target_radius": radius,
            }
        except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError) as exc:
            return None, {"accepted": False, "error": str(exc)[:512]}

    def result(self):
        return {
            **super().result(),
            "schema_version": 2,
            "algorithm": "rigid_three_reference_one_checkpoint_v1",
            "checkpoint": self.checkpoint,
            "rotation_limit_degrees": 5,
            "target_radius_limit_m": 0.25,
            "checkpoint_limit_m": 0.003,
        }
