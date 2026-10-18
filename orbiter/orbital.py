"""Three-dimensional two-body propagation and zero-revolution Lambert transfers.

Vectors must share one nonrotating Cartesian frame and SI units. These equations
work in either handedness because propagation uses linear vector combinations.
No KSP state or controls are accessed here.
"""

from __future__ import annotations

import math

import numpy as np


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite coordinates")
    return result


def stumpff(z):
    """Universal-variable C(z), S(z), with series near the origin."""
    if abs(z) < 1e-5:
        return (
            0.5 - z / 24 + z * z / 720 - z**3 / 40320,
            1 / 6 - z / 120 + z * z / 5040 - z**3 / 362880,
        )
    if z > 0:
        root = math.sqrt(z)
        return (1 - math.cos(root)) / z, (root - math.sin(root)) / (root**3)
    root = math.sqrt(-z)
    if root > 700:
        raise ValueError("Universal anomaly exceeds the hyperbolic numeric range")
    return (math.cosh(root) - 1) / (-z), (math.sinh(root) - root) / (root**3)


def propagate(position, velocity, dt, mu):
    """Universal Kepler propagation, valid for elliptical/hyperbolic coasts."""
    r0, v0 = _vector(position, "position"), _vector(velocity, "velocity")
    if not math.isfinite(dt) or not math.isfinite(mu) or mu <= 0:
        raise ValueError("dt must be finite and mu positive and finite")
    radius = float(np.linalg.norm(r0))
    if radius <= 0:
        raise ValueError("position cannot be zero")
    if dt == 0:
        return r0.copy(), v0.copy()
    sqrt_mu = math.sqrt(mu)
    radial_product = float(r0 @ v0) / sqrt_mu
    alpha = 2 / radius - float(v0 @ v0) / mu

    def equation(chi):
        z = alpha * chi * chi
        c, s = stumpff(z)
        value = (
            radial_product * chi * chi * c
            + (1 - alpha * radius) * chi**3 * s
            + radius * chi
            - sqrt_mu * dt
        )
        derivative = (
            radial_product * chi * (1 - z * s)
            + (1 - alpha * radius) * chi * chi * c
            + radius
        )
        return value, derivative

    direction = 1 if dt > 0 else -1
    extent = max(1.0, sqrt_mu * abs(dt) / radius)
    low, high = (0.0, extent) if direction > 0 else (-extent, 0.0)
    for _ in range(100):
        if equation(low)[0] <= 0 <= equation(high)[0]:
            break
        if direction > 0:
            high *= 2
        else:
            low *= 2
    else:
        raise ValueError("Could not bracket universal Kepler anomaly")
    chi = (low + high) / 2
    for _ in range(100):
        value, derivative = equation(chi)
        if value < 0:
            low = chi
        else:
            high = chi
        candidate = chi - value / derivative if derivative > 0 else math.nan
        candidate = candidate if low < candidate < high else (low + high) / 2
        if abs(candidate - chi) <= 1e-11 * max(1.0, abs(chi)):
            chi = candidate
            break
        chi = candidate
    else:
        raise ValueError("Universal Kepler solve did not converge")
    c, s = stumpff(alpha * chi * chi)
    f, g = 1 - chi * chi / radius * c, dt - chi**3 / sqrt_mu * s
    position = f * r0 + g * v0
    current_radius = float(np.linalg.norm(position))
    fdot = sqrt_mu / (current_radius * radius) * (alpha * chi**3 * s - chi)
    gdot = 1 - chi * chi / current_radius * c
    return position, fdot * r0 + gdot * v0


def lambert(position1, position2, duration, mu, *, long_way=False):
    """Return departure/arrival velocities for a zero-revolution transfer.

    The short and long angular paths are explicit choices. Multi-revolution and
    exactly collinear/antipodal boundary cases are deliberately rejected.
    """
    p1, p2 = _vector(position1, "position1"), _vector(position2, "position2")
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(mu) or mu <= 0:
        raise ValueError("duration and mu must be positive and finite")
    r1, r2 = float(np.linalg.norm(p1)), float(np.linalg.norm(p2))
    if min(r1, r2) <= 0:
        raise ValueError("Lambert endpoints cannot be zero")
    cosine = float(np.clip(p1 @ p2 / (r1 * r2), -1.0, 1.0))
    if 1 - abs(cosine) < 1e-12:
        raise ValueError("Collinear Lambert endpoints are singular")
    sine = math.sqrt(max(0.0, 1 - cosine * cosine)) * (-1 if long_way else 1)
    a = sine * math.sqrt(r1 * r2 / (1 - cosine))
    sqrt_mu = math.sqrt(mu)

    def transfer_time(z):
        c, s = stumpff(z)
        if c <= 0:
            return math.nan, math.nan
        y = r1 + r2 + a * (z * s - 1) / math.sqrt(c)
        if y <= 0:
            return math.nan, y
        chi = math.sqrt(y / c)
        return (chi**3 * s + a * math.sqrt(y)) / sqrt_mu, y

    low, high = -4 * math.pi**2, 4 * math.pi**2 - 1e-6
    # For A>0 the lowest valid y boundary precedes the zero-time limit.
    if not math.isfinite(transfer_time(low)[0]):
        invalid, valid = low, high
        for _ in range(90):
            middle = (invalid + valid) / 2
            if math.isfinite(transfer_time(middle)[0]):
                valid = middle
            else:
                invalid = middle
        low = valid
    t_low, _ = transfer_time(low)
    if t_low > duration:
        # Fast hyperbolic transfers may need a more negative anomaly.
        for _ in range(20):
            candidate = low * 2
            candidate_time, _ = transfer_time(candidate)
            if not math.isfinite(candidate_time):
                break
            low = candidate
            if candidate_time <= duration:
                break
    if transfer_time(low)[0] > duration or transfer_time(high)[0] < duration:
        raise ValueError("Requested Lambert duration is outside the bracketed branch")
    y = math.nan
    for _ in range(100):
        z = (low + high) / 2
        current, y = transfer_time(z)
        if not math.isfinite(current) or current < duration:
            low = z
        else:
            high = z
        if math.isfinite(current) and abs(current - duration) <= 1e-10 * max(
            duration, 1.0
        ):
            break
    else:
        raise ValueError("Lambert solve did not converge")
    f, g, gdot = 1 - y / r1, a * math.sqrt(y / mu), 1 - y / r2
    if abs(g) < 1e-12:
        raise ValueError("Lambert Lagrange g is singular")
    return (p2 - f * p1) / g, (gdot * p2 - p1) / g


def elements(position, velocity, mu):
    r, v = _vector(position, "position"), _vector(velocity, "velocity")
    radius = float(np.linalg.norm(r))
    energy = float(v @ v) / 2 - mu / radius
    h = float(np.linalg.norm(np.cross(r, v)))
    eccentricity = math.sqrt(max(0.0, 1 + 2 * energy * h * h / (mu * mu)))
    p = h * h / mu
    return dict(
        energy=energy,
        eccentricity=eccentricity,
        periapsis=p / (1 + eccentricity),
        apoapsis=p / (1 - eccentricity)
        if energy < 0 and eccentricity < 1
        else math.inf,
    )


def rocket_delta_v(mass, dry_mass, isp):
    if (
        not all(math.isfinite(x) for x in (mass, dry_mass, isp))
        or not 0 < dry_mass <= mass
        or isp <= 0
    ):
        raise ValueError("Need finite wet mass >= positive dry mass and positive Isp")
    return isp * 9.80665 * math.log(mass / dry_mass)
