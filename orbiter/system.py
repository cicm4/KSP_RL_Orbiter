"""Captured KSP body hierarchy and three-dimensional Kepler ephemerides.

Orbital bases come from actual parent-frame KSP positions, avoiding implicit
right-handed/left-handed Euler-angle conventions. This is a fixed stock-system
snapshot; capture again for planet packs or modified gravitational parameters.
"""

from __future__ import annotations

from importlib import resources
import json
import math
from pathlib import Path

import numpy as np


class SystemCatalog:
    def __init__(self, record):
        self.record = record
        self.bodies = dict(record["bodies"])
        if not self.bodies:
            raise ValueError("Body catalog cannot be empty")
        for name, body in self.bodies.items():
            if body.get("name", name) != name:
                raise ValueError("Catalog body name disagrees with its key")
            for field in ("gravitational_parameter", "equatorial_radius"):
                if not math.isfinite(body[field]) or body[field] <= 0:
                    raise ValueError(f"{name}.{field} must be positive and finite")
            self.ancestors(name)  # also validates absent parents and cycles

    @classmethod
    def load(cls, path=None):
        source = (
            Path(path)
            if path
            else resources.files("orbiter").joinpath("data/stock_system.json")
        )
        return cls(json.loads(source.read_text()))

    def ancestors(self, name):
        result = []
        while name is not None:
            if name not in self.bodies or name in result:
                raise ValueError(
                    "Catalog has a missing body or cyclic parent hierarchy"
                )
            result.append(name)
            name = self.bodies[name]["parent"]
        return result

    def route(self, source, target):
        origin, destination = self.ancestors(source), self.ancestors(target)
        common = next((body for body in origin if body in destination), None)
        if common is None:
            raise ValueError(
                "Source and destination have no common gravitational primary"
            )
        return origin[: origin.index(common) + 1] + list(
            reversed(destination[: destination.index(common)])
        )

    def state(self, name, ut, *, relative_to=None):
        """Position/velocity relative to an ancestor in the captured inertial axes."""
        if not math.isfinite(ut):
            raise ValueError("Universal time must be finite")
        ancestors = self.ancestors(name)
        relative_to = relative_to or ancestors[-1]
        if relative_to not in ancestors:
            raise ValueError("Ephemeris reference body must be an ancestor")
        position, velocity = np.zeros(3), np.zeros(3)
        for current in ancestors[: ancestors.index(relative_to)]:
            body = self.bodies[current]
            orbit = body["orbit"]
            mu = self.bodies[body["parent"]]["gravitational_parameter"]
            a, eccentricity = orbit["semi_major_axis"], orbit["eccentricity"]
            if not 0 <= eccentricity < 1 or a <= 0:
                raise ValueError(
                    "Catalog ephemerides require bound elliptic body orbits"
                )
            mean = (
                orbit["mean_anomaly_at_epoch"]
                + math.sqrt(mu / a**3) * (ut - orbit["epoch"])
            ) % (2 * math.pi)
            anomaly = mean if eccentricity < 0.8 else math.pi
            for _ in range(30):
                difference = (anomaly - eccentricity * math.sin(anomaly) - mean) / (
                    1 - eccentricity * math.cos(anomaly)
                )
                anomaly -= difference
                if abs(difference) < 1e-13:
                    break
            else:
                raise ValueError("Catalog eccentric anomaly did not converge")
            p, q = np.asarray(orbit["basis_p"]), np.asarray(orbit["basis_q"])
            cosine, sine = math.cos(anomaly), math.sin(anomaly)
            beta = math.sqrt(1 - eccentricity**2)
            radius = a * (1 - eccentricity * cosine)
            position += a * ((cosine - eccentricity) * p + beta * sine * q)
            velocity += math.sqrt(mu * a) / radius * (-sine * p + beta * cosine * q)
        return position, velocity

    def default_target_altitude(self, name):
        body = self.bodies[name]
        return max(
            20000.0, 0.1 * body["equatorial_radius"], body["atmosphere_depth"] + 20000.0
        )


def catalog_from_capture(record):
    """Retain physical data and fit only orbital-axis orientation, not dynamics."""
    bodies = {}
    for name, raw in record["bodies"].items():
        body = {
            key: raw[key]
            for key in (
                "name",
                "parent",
                "gravitational_parameter",
                "equatorial_radius",
                "atmosphere_depth",
                "sphere_of_influence",
                "has_solid_surface",
            )
        }
        body["orbit"] = None
        if raw["orbit"] is not None:
            samples = raw["ephemeris_parent_nonrotating"]
            first = samples[0]
            second = max(
                samples[1:],
                key=lambda item: abs(
                    math.sin(item["true_anomaly_rad"] - first["true_anomaly_rad"])
                ),
            )
            nu1, nu2 = first["true_anomaly_rad"], second["true_anomaly_rad"]
            separation = math.sin(nu2 - nu1)
            if abs(separation) < 1e-10:
                raise ValueError(
                    f"{name}: ephemeris samples cannot identify orbital axes"
                )
            r1, r2 = np.asarray(first["position_m"]), np.asarray(second["position_m"])
            r1, r2 = r1 / np.linalg.norm(r1), r2 / np.linalg.norm(r2)
            p = (r1 * math.sin(nu2) - r2 * math.sin(nu1)) / separation
            q = (-r1 * math.cos(nu2) + r2 * math.cos(nu1)) / separation
            p /= np.linalg.norm(p)
            q -= float(q @ p) * p
            q /= np.linalg.norm(q)
            body["orbit"] = {
                key: raw["orbit"][key]
                for key in (
                    "semi_major_axis",
                    "eccentricity",
                    "mean_anomaly_at_epoch",
                    "epoch",
                    "period",
                )
            }
            body["orbit"].update(basis_p=p.tolist(), basis_q=q.tolist())
            body["reference_samples"] = samples
        bodies[name] = body
    return dict(
        schema_version=1,
        source=record["source"],
        captured_at_utc=record["captured_at_utc"],
        scope="Fixed KSP stock-system constants and nonrotating Kepler ephemerides; no spacecraft transfer validation.",
        bodies=bodies,
    )
