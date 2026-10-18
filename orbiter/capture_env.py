"""Separate 3D finite-thrust arrival/capture skill, not a transfer mission.

Initial states are already near the destination on incoming hyperbolas. Reaching
these states from Kerbin requires a separate encounter planner and executor.
All dynamics use destination-centred inertial vectors; no aerocapture is modeled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

G0 = 9.80665
TASK = "orbital_capture_v2"


@dataclass
class CaptureConfig:
    # Nominal moon fixture; live catalog measurements supersede these values.
    body_name: str = "Mun"
    mu: float = 6.5138398e10
    radius: float = 200000.0
    atmosphere_depth: float = 0.0
    sphere_of_influence: float = 2429559.0
    target_altitude: float = 50000.0
    target_altitude_randomization: float = 0.2
    orbit_tolerance: float = 5000.0
    surface_clearance: float = 1000.0
    atmosphere_clearance: float = 1000.0
    plane_tolerance_deg: float = 5.0
    mass: float = 4000.0
    dry_mass: float = 1000.0
    thrust: float = 60000.0
    acceleration_gravity_limit: float = 32.0
    isp: float = 345.0
    control_dt: float = 2.0
    physics_dt: float = 0.25
    max_time: float = 1800.0
    pointing_rate_deg_s: float = 30.0
    gamma: float = 0.999
    randomization: float = 0.05
    arrival_vinf_fraction: float = 0.4
    arrival_vinf_by_body: dict[str, float] = field(default_factory=dict)
    arrival_radius_ratio: float = 1.04
    arrival_periapsis_error: float = 0.005
    arrival_plane_error_deg: float = 3.0
    success_hold_steps: int = 2

    def __post_init__(self):
        positive = (
            "mu",
            "radius",
            "sphere_of_influence",
            "target_altitude",
            "orbit_tolerance",
            "mass",
            "dry_mass",
            "thrust",
            "acceleration_gravity_limit",
            "isp",
            "control_dt",
            "physics_dt",
            "max_time",
            "pointing_rate_deg_s",
            "plane_tolerance_deg",
        )
        nonnegative = (
            "atmosphere_depth",
            "surface_clearance",
            "atmosphere_clearance",
            "arrival_vinf_fraction",
            "arrival_periapsis_error",
            "arrival_plane_error_deg",
        )
        for name in (
            positive
            + nonnegative
            + (
                "gamma",
                "randomization",
                "target_altitude_randomization",
                "arrival_radius_ratio",
            )
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
            if name in positive and value <= 0 or name in nonnegative and value < 0:
                raise ValueError(f"invalid {name}")
        if self.mass <= self.dry_mass:
            raise ValueError("mass must exceed dry_mass")
        if not 0 < self.gamma <= 1 or not 0 <= self.randomization < 0.5:
            raise ValueError("invalid gamma/randomization")
        if not 0 <= self.target_altitude_randomization < 1:
            raise ValueError("target_altitude_randomization must be in [0,1)")
        if self.arrival_radius_ratio < 1 or not 0 <= self.arrival_periapsis_error < 0.2:
            raise ValueError("invalid arrival geometry")
        if self.goal_radius + self.orbit_tolerance >= self.sphere_of_influence:
            raise ValueError(
                "target orbit must fit inside the destination sphere of influence"
            )
        if self.goal_radius - self.orbit_tolerance <= self.safe_radius:
            raise ValueError(
                "target band must lie above the surface/atmosphere clearance"
            )
        if (
            isinstance(self.success_hold_steps, bool)
            or not isinstance(self.success_hold_steps, int)
            or self.success_hold_steps < 1
        ):
            raise ValueError("success_hold_steps must be a positive integer")
        if not isinstance(self.arrival_vinf_by_body, dict):
            raise ValueError(
                "arrival_vinf_by_body must map body names to finite nonnegative fractions"
            )
        for name, value in self.arrival_vinf_by_body.items():
            if (
                not isinstance(name, str)
                or not name
                or name.strip() != name
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    "arrival_vinf_by_body must map body names to finite nonnegative fractions"
                )
        self.arrival_vinf_by_body = dict(self.arrival_vinf_by_body)

    @property
    def goal_radius(self):
        return self.radius + self.target_altitude

    @property
    def velocity_scale(self):
        return math.sqrt(self.mu / self.goal_radius)

    @property
    def time_scale(self):
        return self.goal_radius / self.velocity_scale

    @property
    def safe_radius(self):
        return self.radius + max(
            self.surface_clearance, self.atmosphere_depth + self.atmosphere_clearance
        )

    @classmethod
    def from_dict(cls, data):
        return cls(**dict(data))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, allow_nan=False) + "\n"
        )

    @classmethod
    def for_body(cls, body, **overrides):
        """Convert a captured catalog row; explicit overrides retain SI units."""
        aliases = {
            "body_name": ("name", "body_name"),
            "mu": ("gravitational_parameter_m3_s2", "gravitational_parameter", "mu"),
            "radius": ("equatorial_radius_m", "equatorial_radius", "radius"),
            "atmosphere_depth": ("atmosphere_depth_m", "atmosphere_depth"),
            "sphere_of_influence": (
                "sphere_of_influence_m",
                "sphere_of_influence",
                "soi",
            ),
        }
        values = {}
        for name, keys in aliases.items():
            value = next((body[key] for key in keys if key in body), None)
            if value is None:
                if name == "atmosphere_depth":
                    value = 0.0
                else:
                    raise ValueError(f"catalog body lacks {name}")
            values[name] = value
        if "target_altitude" not in overrides:
            values["target_altitude"] = max(
                50000.0, values["atmosphere_depth"] + 20000.0
            )
        values.update(overrides)
        return cls(**values)


def orbital_state(r, v, mu):
    radius = float(np.linalg.norm(r))
    energy = float(v @ v) / 2 - mu / radius
    h = np.cross(r, v)
    hmag = float(np.linalg.norm(h))
    eccentricity_vector = np.cross(v, h) / mu - r / radius
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    p = hmag * hmag / mu
    periapsis = p / (1 + eccentricity)
    apoapsis = p / (1 - eccentricity) if energy < 0 and eccentricity < 1 else math.inf
    return dict(
        radius=radius,
        energy=energy,
        h=h,
        momentum=hmag,
        eccentricity=eccentricity,
        eccentricity_vector=eccentricity_vector,
        periapsis_radius=periapsis,
        apoapsis_radius=apoapsis,
    )


def rtn_basis(r, v):
    radial = np.asarray(r, dtype=float) / np.linalg.norm(r)
    h = np.cross(r, v)
    norm = np.linalg.norm(h)
    if norm < 1e-12:
        axis = np.eye(3)[int(np.argmin(np.abs(radial)))]
        normal = np.cross(radial, axis)
        normal /= np.linalg.norm(normal)
    else:
        normal = h / norm
    transverse = np.cross(normal, radial)
    return np.column_stack((radial, transverse, normal))


def control_acceleration(config, telemetry):
    """Action-unit acceleration in m/s² at this control sample.

    The cap is expressed in destination gravity at the target radius. It keeps
    action resolution comparable on tiny moons and planets without increasing
    the engine's physical thrust. A sampled throttle is held for one interval.
    """
    mass, thrust = float(telemetry["mass"]), float(telemetry["thrust"])
    dry_mass = float(telemetry["dry_mass"])
    if (
        not all(math.isfinite(value) for value in (mass, thrust, dry_mass))
        or mass <= 0
        or dry_mass <= 0
        or thrust < 0
        or mass < dry_mass
    ):
        raise ValueError("invalid capture actuator telemetry")
    physical = thrust / mass if mass > dry_mass + 1e-9 else 0.0
    return min(
        physical, config.acceleration_gravity_limit * config.mu / config.goal_radius**2
    )


def decode_capture_action(config, telemetry, action):
    """Decode a v2 RTN acceleration action to ``(throttle, RTN unit direction)``.

    Norms above one saturate without changing direction. Zero returns a zero
    direction and zero throttle. The live adapter maps the returned RTN direction
    through its current right-handed RTN basis before commanding attitude.
    """
    command = np.asarray(action, dtype=float)
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError("capture action must be a finite RTN 3-vector")
    magnitude = float(np.linalg.norm(command))
    if not math.isfinite(magnitude):
        raise ValueError("capture action magnitude must be finite")
    if magnitude == 0:
        return 0.0, np.zeros(3)
    authority = control_acceleration(config, telemetry)
    physical = float(telemetry["thrust"]) / float(telemetry["mass"])
    throttle = min(1.0, magnitude) * authority / physical if physical > 0 else 0.0
    return float(throttle), command / magnitude


def build_capture_observation(config, telemetry):
    """Shared 27-float policy contract from SI destination-frame telemetry.

    Required telemetry: r/v/direction/goal_normal (3-vectors), mass/initial_mass,
    dry_mass/thrust/isp/time. Frame parity is handled by the live adapter before
    calling this builder. This module uses ordinary right-handed cross products.
    """
    d, c = telemetry, config
    r, v = np.asarray(d["r"], dtype=float), np.asarray(d["v"], dtype=float)
    state = orbital_state(r, v, c.mu)
    basis = rtn_basis(r, v)
    local_v = basis.T @ v
    goal_normal = np.array(d["goal_normal"], dtype=float, copy=True)
    goal_normal /= np.linalg.norm(goal_normal)
    direction = np.array(d["direction"], dtype=float, copy=True)
    direction /= max(np.linalg.norm(direction), 1e-12)
    L, V, T = c.goal_radius, c.velocity_scale, c.time_scale
    fuel = max(0.0, d["mass"] - d["dry_mass"])
    initial_fuel = max(1e-12, d["initial_mass"] - d["dry_mass"])
    available_dv = d["isp"] * G0 * math.log(max(d["mass"] / d["dry_mass"], 1.0))
    # A finite geometric time-to-periapsis cue; avoids divergent anomaly formulae
    # at a parabolic orbit. The state vectors contain the exact orbital state.
    approaching = max(0.0, -local_v[0])
    tta = max(0.0, state["radius"] - state["periapsis_radius"]) / max(
        approaching, 0.02 * V
    )
    values = [
        state["radius"] / L,
        local_v[0] / V,
        local_v[1] / V,
        state["energy"] / (V * V),
        state["eccentricity"],
        state["periapsis_radius"] / L,
        state["apoapsis_radius"] / L
        if math.isfinite(state["apoapsis_radius"])
        else 10.0,
        *(basis.T @ goal_normal),
        *(basis.T @ direction),
        fuel / initial_fuel,
        d["mass"] / d["initial_mass"],
        available_dv / V,
        math.log1p(control_acceleration(c, d) * T / V),
        math.log1p(d["isp"] * G0 / V),
        c.radius / L,
        c.safe_radius / L,
        math.log1p(c.sphere_of_influence / L),
        min(tta / T, 10.0),
        d["time"] / c.max_time,
        (state["radius"] - L) / L,
        1.0 if state["energy"] < 0 else -1.0,
        state["momentum"] / (L * V),
        c.control_dt / T,
    ]
    return np.clip(np.asarray(values, dtype=np.float32), -10.0, 10.0)


def capture_teacher(env):
    """Transparent finite-burn state feedback; never receives future simulator state."""
    c, d = env.config, env.telemetry()
    if d["orbit_in_band"]:
        return np.zeros(3, dtype=np.float32)
    r, v = env.r, env.v
    radius = float(np.linalg.norm(r))
    basis = rtn_basis(r, v)
    vr, vt, _ = basis.T @ v
    tau = max(3 * c.control_dt, 0.055 * c.time_scale)
    circular = math.sqrt(c.mu / radius)
    radial_acc = (
        c.mu / radius**2
        - vt * vt / radius
        - 2 * vr / tau
        - (radius - c.goal_radius) / tau**2
    )
    transverse_acc = vr * vt / radius - (vt - circular) / tau
    request = radial_acc * basis[:, 0] + transverse_acc * basis[:, 1]
    # Stabilize position/velocity in the requested orbital plane as well.
    normal_projection = float(basis[:, 2] @ env.goal_normal)
    if abs(normal_projection) > 0.2:
        plane_position, plane_velocity = (
            float(r @ env.goal_normal),
            float(v @ env.goal_normal),
        )
        desired_plane_acc = -2 * plane_velocity / tau - plane_position / tau**2
        required_normal = (
            desired_plane_acc
            + c.mu / radius**3 * plane_position
            - request @ env.goal_normal
        ) / normal_projection
        request += required_normal * basis[:, 2]
    authority = control_acceleration(c, d)
    if authority <= 0:
        return np.zeros(3, dtype=np.float32)
    local = basis.T @ request / authority
    magnitude = float(np.linalg.norm(local))
    if magnitude > 1:
        local /= magnitude
    return np.asarray(local, dtype=np.float32)


class CaptureEnv(gym.Env):
    """Goal-conditioned capture only: reset starts already at the destination."""

    metadata = {"render_modes": []}
    task = TASK

    def __init__(self, config=None, body_catalog=None):
        self.base_config = config or CaptureConfig()
        self.config = self.base_config
        if body_catalog and "bodies" in body_catalog:
            body_catalog = body_catalog["bodies"]
        if isinstance(body_catalog, list):
            body_catalog = {body["name"]: body for body in body_catalog}
        self.body_catalog = dict(body_catalog or {})
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (27,), np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        c = self.base_config
        if self.body_catalog:
            name = options.get("body") or str(
                self.np_random.choice(sorted(self.body_catalog))
            )
            if name not in self.body_catalog:
                raise ValueError(f"Unknown destination {name!r}")
            vehicle = {
                key: getattr(c, key)
                for key in (
                    "mass",
                    "dry_mass",
                    "thrust",
                    "acceleration_gravity_limit",
                    "isp",
                    "control_dt",
                    "physics_dt",
                    "max_time",
                    "gamma",
                    "randomization",
                    "target_altitude_randomization",
                    "orbit_tolerance",
                    "surface_clearance",
                    "atmosphere_clearance",
                    "plane_tolerance_deg",
                    "pointing_rate_deg_s",
                    "arrival_vinf_fraction",
                    "arrival_vinf_by_body",
                    "arrival_radius_ratio",
                    "arrival_periapsis_error",
                    "arrival_plane_error_deg",
                    "success_hold_steps",
                )
            }
            c = CaptureConfig.for_body(self.body_catalog[name], **vehicle)
        elif options.get("body", c.body_name) != c.body_name:
            raise ValueError("body reset requires an entry in body_catalog")
        if "target_altitude" in options:
            c = replace(c, target_altitude=float(options["target_altitude"]))
        elif c.target_altitude_randomization:
            low = max(
                c.target_altitude * (1 - c.target_altitude_randomization),
                np.nextafter(c.safe_radius + c.orbit_tolerance - c.radius, math.inf),
            )
            high = min(
                c.target_altitude * (1 + c.target_altitude_randomization),
                np.nextafter(
                    c.sphere_of_influence - c.orbit_tolerance - c.radius, -math.inf
                ),
            )
            if low >= high:
                raise ValueError("no safe target range remains for randomized goals")
            c = replace(c, target_altitude=float(self.np_random.uniform(low, high)))
        vehicle_overrides = {
            key: float(options[key])
            for key in ("mass", "dry_mass", "thrust", "isp")
            if key in options
        }
        if vehicle_overrides:
            c = replace(c, **vehicle_overrides)
        self.config = c
        scale = lambda: 1 + self.np_random.uniform(-c.randomization, c.randomization)
        self.initial_mass = c.mass if "mass" in options else c.mass * scale()
        if self.initial_mass <= c.dry_mass:
            raise ValueError("randomized mass must exceed dry mass")
        self.mass, self.dry_mass = self.initial_mass, c.dry_mass
        self.thrust = c.thrust if "thrust" in options else c.thrust * scale()
        self.isp = c.isp if "isp" in options else c.isp * scale()
        self.time, self._hold, self._ended = 0.0, 0, False
        # QR gives arbitrary 3D orientation; reverse one column for a right-handed basis.
        matrix, _ = np.linalg.qr(self.np_random.normal(size=(3, 3)))
        if np.linalg.det(matrix) < 0:
            matrix[:, 2] *= -1
        self.goal_normal = matrix[:, 2].copy()
        periapsis = c.goal_radius * (
            1
            + self.np_random.uniform(
                -c.arrival_periapsis_error, c.arrival_periapsis_error
            )
        )
        start_radius = max(periapsis, c.goal_radius * c.arrival_radius_ratio)
        arrival_fraction = c.arrival_vinf_by_body.get(
            c.body_name, c.arrival_vinf_fraction
        )
        vinf = c.velocity_scale * arrival_fraction * scale()
        h = periapsis * math.sqrt(vinf * vinf + 2 * c.mu / periapsis)
        vt = h / start_radius
        vr = -math.sqrt(max(0.0, vinf * vinf + 2 * c.mu / start_radius - vt * vt))
        tilt = math.radians(
            self.np_random.uniform(
                -c.arrival_plane_error_deg, c.arrival_plane_error_deg
            )
        )
        # The arrival plane can differ from the requested plane.
        arrival_tangent = math.cos(tilt) * matrix[:, 1] + math.sin(tilt) * matrix[:, 2]
        self.r = start_radius * matrix[:, 0]
        self.v = vr * matrix[:, 0] + vt * arrival_tangent
        for key in ("r", "v"):
            if key in options:
                value = np.asarray(options[key], dtype=float)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError(f"{key} must be a finite 3-vector")
                setattr(self, key, value.copy())
        if not c.safe_radius < np.linalg.norm(self.r) < c.sphere_of_influence:
            raise ValueError(
                "initial position must be above the safe altitude and inside the destination SOI"
            )
        if "goal_normal" in options:
            normal = np.asarray(options["goal_normal"], dtype=float)
            if (
                normal.shape != (3,)
                or not np.isfinite(normal).all()
                or np.linalg.norm(normal) < 1e-10
            ):
                raise ValueError("goal_normal must be a finite nonzero 3-vector")
            self.goal_normal = normal / np.linalg.norm(normal)
        self.direction = (
            -self.v / np.linalg.norm(self.v)
            if np.linalg.norm(self.v) > 1e-12
            else -self.r / np.linalg.norm(self.r)
        )
        self.max_radius = float(np.linalg.norm(self.r))
        self._potential_prev = self.potential()
        return self.observation(), self.info("")

    def telemetry(self):
        c = self.config
        state = orbital_state(self.r, self.v, c.mu)
        hnorm = state["momentum"]
        cosine = float(state["h"] @ self.goal_normal / max(hnorm, 1e-12))
        plane_error = math.degrees(math.acos(float(np.clip(cosine, -1, 1))))
        in_band = (
            state["energy"] < 0
            and state["periapsis_radius"] > c.safe_radius
            and abs(state["periapsis_radius"] - c.goal_radius) <= c.orbit_tolerance
            and abs(state["apoapsis_radius"] - c.goal_radius) <= c.orbit_tolerance
            and plane_error <= c.plane_tolerance_deg
        )
        return dict(
            **state,
            r=self.r.copy(),
            v=self.v.copy(),
            direction=self.direction.copy(),
            goal_normal=self.goal_normal.copy(),
            mass=self.mass,
            initial_mass=self.initial_mass,
            dry_mass=self.dry_mass,
            thrust=self.thrust,
            isp=self.isp,
            time=self.time,
            body=c.body_name,
            altitude=state["radius"] - c.radius,
            apoapsis=state["apoapsis_radius"] - c.radius,
            periapsis=state["periapsis_radius"] - c.radius,
            plane_error_deg=plane_error,
            orbit_in_band=in_band,
            control_acceleration=control_acceleration(
                c, dict(mass=self.mass, dry_mass=self.dry_mass, thrust=self.thrust)
            ),
            physical_max_acceleration=self.thrust / self.mass
            if self.mass > self.dry_mass + 1e-9
            else 0.0,
            available_delta_v=self.isp * G0 * math.log(self.mass / self.dry_mass),
        )

    def observation(self):
        return build_capture_observation(self.config, self.telemetry())

    def potential(self):
        d, c = self.telemetry(), self.config
        energy_error = d["energy"] / (c.velocity_scale**2) + 0.5
        momentum_error = d["momentum"] / (c.goal_radius * c.velocity_scale) - 1
        plane_error = 1 - math.cos(math.radians(d["plane_error_deg"]))
        radius_error = d["radius"] / c.goal_radius - 1
        return -float(
            min(
                30.0,
                4 * energy_error**2
                + 8 * momentum_error**2
                + 4 * plane_error
                + radius_error**2,
            )
        )

    def teacher_action(self):
        return capture_teacher(self)

    def _derivative(self, state, mass, thrust_vector):
        r, v = state[:3], state[3:]
        return np.concatenate(
            (v, -self.config.mu * r / np.linalg.norm(r) ** 3 + thrust_vector / mass)
        )

    def step(self, action):
        if self._ended:
            raise RuntimeError("reset required after capture episode")
        throttle, local_direction = decode_capture_action(
            self.config, self.telemetry(), action
        )
        target = rtn_basis(self.r, self.v) @ local_direction
        elapsed = 0.0
        reason = ""
        previous_mass = self.mass
        duration = min(
            self.config.control_dt, max(0.0, self.config.max_time - self.time)
        )
        while elapsed < duration - 1e-9:
            dt = min(self.config.physics_dt, duration - elapsed)
            mdot = (
                throttle * self.thrust / (self.isp * G0)
                if self.mass > self.dry_mass + 1e-9
                else 0.0
            )
            if mdot > 0:
                dt = min(dt, (self.mass - self.dry_mass) / mdot)
            if dt <= 1e-10:
                self.mass = self.dry_mass
                break
            if throttle > 0:
                dot = float(np.clip(self.direction @ target, -1, 1))
                angle = math.acos(dot)
                turn = min(angle, math.radians(self.config.pointing_rate_deg_s) * dt)
                if angle < 1e-10:
                    self.direction = target.copy()
                else:
                    axis = np.cross(self.direction, target)
                    if np.linalg.norm(axis) < 1e-10:
                        # At exactly 180 degrees there is no preferred turn axis.
                        reference = np.eye(3)[int(np.argmin(np.abs(self.direction)))]
                        axis = np.cross(self.direction, reference)
                    axis /= np.linalg.norm(axis)
                    self.direction = math.cos(turn) * self.direction + math.sin(
                        turn
                    ) * np.cross(axis, self.direction)
                    self.direction /= np.linalg.norm(self.direction)
            force = throttle * self.thrust * self.direction if mdot > 0 else np.zeros(3)
            mass = max(self.dry_mass, self.mass - 0.5 * mdot * dt)
            z = np.concatenate((self.r, self.v))
            f = lambda y: self._derivative(y, mass, force)
            k1 = f(z)
            k2 = f(z + 0.5 * dt * k1)
            k3 = f(z + 0.5 * dt * k2)
            k4 = f(z + dt * k3)
            z += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            self.r, self.v = z[:3], z[3:]
            self.mass = max(self.dry_mass, self.mass - mdot * dt)
            elapsed += dt
            self.time += dt
            radius = float(np.linalg.norm(self.r))
            self.max_radius = max(self.max_radius, radius)
            if radius <= self.config.safe_radius:
                reason = "unsafe_altitude"
                break
            if radius >= self.config.sphere_of_influence:
                reason = "sphere_of_influence_exit"
                break
        d = self.telemetry()
        self._hold = self._hold + 1 if d["orbit_in_band"] else 0
        if not reason and self._hold >= self.config.success_hold_steps:
            reason = "orbit_captured"
        if not reason and self.mass <= self.dry_mass + 1e-7 and not d["orbit_in_band"]:
            reason = "out_of_fuel"
        if (
            not reason
            and d["energy"] >= 0
            and self.r @ self.v > 0
            and d["radius"] > 1.6 * self.config.goal_radius
        ):
            reason = "missed_capture"
        terminated = bool(reason)
        truncated = not terminated and self.time >= self.config.max_time - 1e-8
        if truncated:
            reason = "time_limit"
        following = self.potential()
        shaping = (
            self.config.gamma * (0.0 if terminated else following)
            - self._potential_prev
        )
        task = 100.0 if reason == "orbit_captured" else -100.0 if terminated else 0.0
        fuel_cost = 0.1 * (previous_mass - self.mass) / self.initial_mass
        reward = shaping + task - 0.01 * elapsed / self.config.time_scale - fuel_cost
        self._potential_prev = following
        self._ended = terminated or truncated
        info = self.info(reason)
        info.update(reward_shaping=shaping, reward_task=task, reward_fuel=-fuel_cost)
        return self.observation(), float(reward), terminated, truncated, info

    def info(self, reason):
        d = self.telemetry()
        return dict(
            task=TASK,
            body=self.config.body_name,
            success=reason == "orbit_captured",
            termination_reason=reason,
            target_altitude=self.config.target_altitude,
            orbit_tolerance=self.config.orbit_tolerance,
            time=self.time,
            altitude=d["altitude"],
            apoapsis=d["apoapsis"],
            periapsis=d["periapsis"],
            eccentricity=d["eccentricity"],
            plane_error_deg=d["plane_error_deg"],
            mass=self.mass,
            fuel_remaining=self.mass - self.dry_mass,
            available_delta_v=d["available_delta_v"],
            energy=d["energy"],
            max_altitude=self.max_radius - self.config.radius,
            arrival_already_at_destination=True,
        )
