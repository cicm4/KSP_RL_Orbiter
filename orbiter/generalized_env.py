"""Experimental GPU-batched, vehicle-conditioned Kerbin ascent surrogate.

This is a new contract. Existing 16-float flight policies cannot load it. Its
sequential liquid stages, planar aerodynamics and attitude setpoint model do
not represent arbitrary KSP vessels. Float64 RK4 translation follows env.py;
neural observations are float32 and remain on the selected Torch device.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from .env import EnvConfig, G0, HeadlessOrbiterEnv, Stage

TASK = "vehicle_conditioned_ascent_v3"
FEATURE_SCHEMA = "ascent16_log_q_twr_stage4x8_log_thrust_globals6_log_slew_v3"
MAX_STAGES = 4
OBSERVATIONS = 54


@dataclass(frozen=True)
class VehicleEnvelope:
    """A training distribution, not a claim of live KSP compatibility."""

    payload_min: float = 150.0
    payload_max: float = 15000.0
    stages_min: int = 2
    stages_max: int = 4
    stage_wet_to_upper_min: float = 1.2
    stage_wet_to_upper_max: float = 2.8
    stage_dry_fraction_min: float = 0.08
    stage_dry_fraction_max: float = 0.16
    initial_twr_min: float = 1.35
    initial_twr_max: float = 2.1
    upper_twr_min: float = 0.9
    upper_twr_max: float = 2.5
    isp_min: float = 300.0
    isp_max: float = 380.0
    sea_level_isp_ratio_min: float = 0.65
    sea_level_isp_ratio_max: float = 0.9
    drag_area_per_tonne_min: float = 0.025
    drag_area_per_tonne_max: float = 0.18
    attitude_rate_min: float = 4.0
    attitude_rate_max: float = 18.0
    attitude_tau_min: float = 0.35
    attitude_tau_max: float = 1.5
    target_min: float = 90000.0
    target_max: float = 130000.0
    minimum_ideal_delta_v: float = 4800.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in asdict(self):
            if key.endswith("_min"):
                if getattr(self, key) > getattr(self, key[:-4] + "_max"):
                    raise ValueError(f"invalid {key} range")
        if not 1 <= self.stages_min <= self.stages_max <= MAX_STAGES:
            raise ValueError("one to four sequential stages are supported")
        if not isinstance(self.stages_min, int) or not isinstance(self.stages_max, int):
            raise ValueError("stage counts must be integers")
        if self.stage_dry_fraction_max >= 1 or self.sea_level_isp_ratio_max > 1:
            raise ValueError("invalid mass/Isp fractions")
        if self.initial_twr_min <= 1:
            raise ValueError(
                "pad TWR must exceed one; this distribution is for launches"
            )
        maximum_per_stage = (
            self.isp_max
            * G0
            * math.log(
                (1 + self.stage_wet_to_upper_max)
                / (1 + self.stage_dry_fraction_min * self.stage_wet_to_upper_max)
            )
        )
        if self.stages_max * maximum_per_stage < self.minimum_ideal_delta_v:
            raise ValueError("envelope cannot pass ideal delta-v screen")


def assess_vehicle(config, *, minimum_delta_v=4800.0):
    """Conservative screening diagnostics; acceptance is not a mission proof."""
    failures = []
    if len(config.stages) > MAX_STAGES:
        failures.append("more_than_four_sequential_stages")
    wet_mass = config.payload_mass + sum(
        s.dry_mass + s.fuel_mass for s in config.stages
    )
    first = config.stages[0]
    sea_level_twr = first.thrust_vac * first.isp_sl / first.isp_vac / wet_mass / G0
    if sea_level_twr <= 1:
        failures.append("insufficient_pad_twr")
    ideal_delta_v = 0.0
    for index, stage in enumerate(config.stages):
        ignition_mass = config.payload_mass + sum(
            s.dry_mass + s.fuel_mass for s in config.stages[index:]
        )
        ideal_delta_v += (
            stage.isp_vac
            * G0
            * math.log(ignition_mass / (ignition_mass - stage.fuel_mass))
        )
    if ideal_delta_v < minimum_delta_v:
        failures.append("insufficient_ideal_delta_v")
    if config.attitude_rate <= 0:
        failures.append("no_attitude_authority")
    if config.target_altitude - config.orbit_tolerance <= config.atmosphere_depth:
        failures.append("goal_band_intersects_atmosphere")
    return dict(
        accepted=not failures,
        failures=failures,
        sea_level_twr=sea_level_twr,
        ideal_delta_v_m_s=ideal_delta_v,
        screen_only=True,
        live_ksp_validated=False,
    )


def normalization_audit(envelope):
    """Conservative analytic feature maxima for the declared training envelope.

    Stage ignition masses include all upper wet stages. Exhausting one stage
    changes its mass by at most (1+w)/(1+dry*w); that bounds instantaneous TWR
    before staging as well as the static thrust descriptor. This audit concerns
    craft/actuator information, not the redundant clipped apoapsis diagnostic.
    """
    e = envelope
    vc_min = math.sqrt(3.5316e12 / (600000 + e.target_max))
    tc_min = (600000 + e.target_min) ** 1.5 / math.sqrt(3.5316e12)
    tc_max = (600000 + e.target_max) ** 1.5 / math.sqrt(3.5316e12)
    mass_ratio = (1 + e.stage_wet_to_upper_max) / (
        1 + e.stage_dry_fraction_min * e.stage_wet_to_upper_max
    )
    vacuum_twr_max = max(e.initial_twr_max / e.sea_level_isp_ratio_min, e.upper_twr_max)
    instantaneous_twr_max = vacuum_twr_max * mass_ratio
    propellant_fraction = (
        e.stage_wet_to_upper_max
        / (1 + e.stage_wet_to_upper_max)
        * (1 - e.stage_dry_fraction_min)
    )
    vacuum_twr_min = min(e.initial_twr_min / e.sea_level_isp_ratio_max, e.upper_twr_min)
    # An active world starts the control interval with sub-escape energy;
    # add a full interval at peak acceleration and maximum atmosphere rotation.
    max_air_speed = (
        math.sqrt(2 * 3.5316e12 / 600000)
        + 2 * instantaneous_twr_max * G0
        + (2 * math.pi / 21549.425 * 670000)
    )
    maximum_q = 0.5 * 1.225 * max_air_speed**2
    bounds = dict(
        stage_mask=1,
        stage_dry_mass_fraction=1,
        stage_fuel_mass_fraction=1,
        log_stage_thrust=math.log1p(vacuum_twr_max),
        exhaust_velocity_over_circular=e.isp_max * G0 / vc_min,
        sea_level_isp_ratio=e.sea_level_isp_ratio_max,
        stage_delta_v_over_circular=e.isp_max * G0 * math.log(mass_ratio) / vc_min,
        stage_burn_time_over_orbital=propellant_fraction
        * e.isp_max
        / vacuum_twr_min
        / tc_min,
        payload_mass_fraction=1,
        drag_area_per_tonne=e.drag_area_per_tonne_max,
        log_attitude_slew=math.log1p(e.attitude_rate_max * tc_max / 360),
        attitude_tau_over_orbital=e.attitude_tau_max / tc_min,
        target_altitude_over_radius=e.target_max / 600000,
        stage_count_fraction=1,
        log_current_twr=math.log1p(instantaneous_twr_max),
        log_dynamic_pressure=math.log1p(maximum_q / 40000),
    )
    return dict(
        feature_schema=FEATURE_SCHEMA,
        upper_bounds=bounds,
        all_craft_and_actuator_features_below_clip=all(
            value < 10 for value in bounds.values()
        ),
        legacy_state_diagnostics_remain_clipped=[
            "altitude",
            "velocity",
            "apoapsis",
            "periapsis",
            "energy",
        ],
        task_scope="sub-escape sequential liquid launches; no structural or aero-stability proof",
    )


def craft_context(config, stages, *, initial_mass, drag_area, attitude_rate):
    """Versioned 38-float morphology context, shared with CPU reference tests."""
    vc = math.sqrt(config.mu / (config.radius + config.target_altitude))
    tc = (config.radius + config.target_altitude) / vc
    rows = []
    for index in range(MAX_STAGES):
        if index >= len(stages):
            rows.extend([0.0] * 8)
            continue
        stage = stages[index]
        ignition_mass = config.payload_mass + sum(
            item.dry_mass + item.fuel_mass for item in stages[index:]
        )
        dv = (
            stage.isp_vac
            * G0
            * math.log(ignition_mass / (ignition_mass - stage.fuel_mass))
        )
        burn_time = stage.fuel_mass * stage.isp_vac * G0 / stage.thrust_vac
        rows.extend(
            [
                1.0,
                stage.dry_mass / initial_mass,
                stage.fuel_mass / initial_mass,
                math.log1p(stage.thrust_vac / initial_mass / G0),
                stage.isp_vac * G0 / vc,
                stage.isp_sl / stage.isp_vac,
                dv / vc,
                burn_time / tc,
            ]
        )
    rows.extend(
        [
            config.payload_mass / initial_mass,
            drag_area / initial_mass * 1000,
            math.log1p(attitude_rate * tc / 360),
            config.attitude_tau / tc,
            config.target_altitude / config.radius,
            len(stages) / MAX_STAGES,
        ]
    )
    return np.clip(np.asarray(rows, dtype=np.float32), -10, 10)


def cpu_observation(env):
    base = env.observation().copy()
    # Read raw telemetry before the legacy builder's [-10,10] clipping.
    # Otherwise applying a log afterwards would preserve the old alias.
    base[12] = math.log1p(env.telemetry()["dynamic_pressure"] / 40000)
    base[13] = math.log1p(env.available_thrust() / env.mass / G0)
    return np.concatenate(
        (
            base,
            craft_context(
                env.config,
                env.stages,
                initial_mass=env.initial_mass,
                drag_area=env.drag_area,
                attitude_rate=env.attitude_rate,
            ),
        )
    )


class BatchedAscent:
    """Independent RK4 worlds on one device, with explicit auto-reset by caller.

    Each sampled craft is a distinct payload/stage/performance descriptor. Ideal
    delta-v and pad TWR are screening lower bounds, not finite-burn feasibility
    proofs. Unsuccessful physically plausible craft remain policy failures.
    """

    def __init__(
        self,
        count=1024,
        *,
        device="cpu",
        seed=0,
        envelope=None,
        physics_dt=0.25,
        max_time=1600.0,
        orbit_tolerance=20000.0,
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        if not math.isfinite(physics_dt) or physics_dt <= 0 or physics_dt > 2:
            raise ValueError("physics_dt must be in (0,2]")
        if not math.isfinite(max_time) or max_time <= 0:
            raise ValueError("max_time must be positive")
        self.count, self.device = count, torch.device(device)
        self.envelope = envelope or VehicleEnvelope()
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.dtype = torch.float64
        self.physics_dt, self.max_time = physics_dt, max_time
        self.orbit_tolerance = orbit_tolerance
        self.mu, self.radius = 3.5316e12, 600000.0
        self.rotation_period, self.atmosphere_depth = 21549.425, 70000.0
        self.control_dt, self.gamma = 2.0, 0.999
        self.atmosphere = torch.tensor(
            EnvConfig().atmosphere, device=self.device, dtype=self.dtype
        )
        self._atmosphere_heights = self.atmosphere[:, 0].contiguous()
        self._atmosphere_height_delta = self._atmosphere_heights.diff()
        self._atmosphere_lower = self.atmosphere[:-1, 1:].contiguous()
        self._atmosphere_delta = self.atmosphere[1:, 1:] - self._atmosphere_lower
        self._atmosphere_log_lower = self._atmosphere_lower.clamp(min=1e-30).log()
        self._atmosphere_log_delta = (
            self.atmosphere[1:, 1:].clamp(min=1e-30).log() - self._atmosphere_log_lower
        )
        self._atmosphere_positive = (self._atmosphere_lower > 0) & (
            self.atmosphere[1:, 1:] > 0
        )
        self.ids = torch.arange(count, device=self.device)
        self.stage_ids = torch.arange(MAX_STAGES, device=self.device)[None, :]
        self.state = self.zeros(count, 4)
        self.stages = self.zeros(count, MAX_STAGES, 5)
        for name in (
            "payload",
            "initial_mass",
            "drag_area",
            "attitude_rate",
            "attitude_tau",
            "target",
            "fuel",
            "pitch",
            "throttle",
            "time",
            "previous_potential",
            "episode_return",
            "ideal_delta_v",
        ):
            setattr(self, name, self.zeros(count))
        self.stage = torch.zeros(count, dtype=torch.long, device=self.device)
        self.stage_count = self.stage.clone()
        self.reset()

    def zeros(self, *shape):
        return torch.zeros(shape, device=self.device, dtype=self.dtype)

    def uniform(self, size, low, high):
        return low + (high - low) * torch.rand(
            size, device=self.device, dtype=self.dtype, generator=self.generator
        )

    @torch.no_grad()
    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.count, dtype=torch.bool, device=self.device)
        selected = self.ids[mask]
        n, e = selected.numel(), self.envelope
        if n == 0:
            return self.observation()
        payload = self.uniform(
            (n,), math.log(e.payload_min), math.log(e.payload_max)
        ).exp()
        counts = torch.randint(
            e.stages_min,
            e.stages_max + 1,
            (n,),
            generator=self.generator,
            device=self.device,
        )
        sampled = self.zeros(n, MAX_STAGES, 5)
        upper = payload.clone()
        ideal_dv = self.zeros(n)
        # Backward recursion makes every stage carry the full wet upper stack.
        for index in reversed(range(MAX_STAGES)):
            present = index < counts
            wet = upper * self.uniform(
                (n,), e.stage_wet_to_upper_min, e.stage_wet_to_upper_max
            )
            dry = wet * self.uniform(
                (n,), e.stage_dry_fraction_min, e.stage_dry_fraction_max
            )
            fuel = wet - dry
            isp = self.uniform((n,), e.isp_min, e.isp_max)
            ratio = self.uniform(
                (n,), e.sea_level_isp_ratio_min, e.sea_level_isp_ratio_max
            )
            twr = self.uniform(
                (n,),
                e.initial_twr_min if index == 0 else e.upper_twr_min,
                e.initial_twr_max if index == 0 else e.upper_twr_max,
            )
            thrust = twr * (upper + wet) * G0 / (ratio if index == 0 else 1)
            row = torch.stack((dry, fuel, thrust, isp, isp * ratio), dim=-1)
            sampled[:, index] = torch.where(present[:, None], row, 0)
            ideal_dv += torch.where(
                present, isp * G0 * ((upper + wet) / (upper + dry)).log(), 0
            )
            upper += torch.where(present, wet, 0)
        # Every row must satisfy the declared screen; resample any rejected craft
        # rather than quietly changing its mass or marking it policy-successful.
        self.stages[selected] = sampled
        self.payload[selected], self.initial_mass[selected] = payload, upper
        self.stage_count[selected] = counts
        self.ideal_delta_v[selected] = ideal_dv
        self.drag_area[selected] = (
            upper
            / 1000
            * self.uniform((n,), e.drag_area_per_tonne_min, e.drag_area_per_tonne_max)
        )
        self.attitude_rate[selected] = self.uniform(
            (n,), e.attitude_rate_min, e.attitude_rate_max
        )
        self.attitude_tau[selected] = self.uniform(
            (n,), e.attitude_tau_min, e.attitude_tau_max
        )
        self.target[selected] = self.uniform((n,), e.target_min, e.target_max)
        self.stage[selected] = 0
        self.fuel[selected] = sampled[:, 0, 1]
        self.state[selected] = torch.tensor(
            [
                self.radius + 80,
                0,
                0,
                (self.radius + 80) * 2 * math.pi / self.rotation_period,
            ],
            device=self.device,
            dtype=self.dtype,
        )
        self.pitch[selected], self.throttle[selected], self.time[selected] = 90, 0, 0
        self.episode_return[selected] = 0
        self.previous_potential[selected] = self.potential()[selected]
        rejected = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        rejected[selected] = ideal_dv < e.minimum_ideal_delta_v
        # This is bounded: an impossible envelope is a configuration error.
        for _ in range(1):
            if bool(rejected.any()):
                if getattr(self, "_resampling_depth", 0) >= 100:
                    raise ValueError("envelope cannot pass ideal delta-v screen")
                self._resampling_depth = getattr(self, "_resampling_depth", 0) + 1
                self.reset(rejected)
                self._resampling_depth -= 1
        return self.observation()

    def atmosphere_at(self, altitude):
        h = altitude.clamp(min=0)
        index = torch.searchsorted(self._atmosphere_heights, h.contiguous(), right=True)
        index = index.clamp(1, len(self._atmosphere_heights) - 1) - 1
        f = (
            (h - self._atmosphere_heights[index]) / self._atmosphere_height_delta[index]
        ).clamp(0, 1)[:, None]
        out = torch.where(
            self._atmosphere_positive[index],
            (
                self._atmosphere_log_lower[index]
                + f * self._atmosphere_log_delta[index]
            ).exp(),
            self._atmosphere_lower[index] + f * self._atmosphere_delta[index],
        )
        return torch.where((h >= self.atmosphere_depth)[:, None], 0, out)

    def mass(self):
        future = self.stage_ids > self.stage[:, None]
        current_dry = self.stages[self.ids, self.stage, 0]
        return (
            self.payload
            + current_dry
            + self.fuel
            + (self.stages[:, :, :2].sum(-1) * future).sum(-1)
        )

    def telemetry(self):
        x, y, vx, vy = self.state.unbind(-1)
        radius = torch.hypot(x, y)
        vr, vt = (x * vx + y * vy) / radius, (x * vy - y * vx) / radius
        energy = (vx * vx + vy * vy) / 2 - self.mu / radius
        momentum = x * vy - y * vx
        eccentricity = (
            (1 + 2 * energy * momentum.square() / self.mu**2).clamp(min=0).sqrt()
        )
        parameter = momentum.square() / self.mu
        peri = parameter / (1 + eccentricity) - self.radius
        apo = torch.where(
            (energy < 0) & (eccentricity < 1),
            parameter / (1 - eccentricity).clamp(min=1e-15) - self.radius,
            torch.inf,
        )
        rho = self.atmosphere_at(radius - self.radius)[:, 0]
        relative_vt = vt - 2 * math.pi / self.rotation_period * radius
        q = 0.5 * rho * (vr.square() + relative_vt.square())
        return dict(
            altitude=radius - self.radius,
            vertical_speed=vr,
            horizontal_speed=vt,
            energy=energy,
            momentum=momentum,
            periapsis=peri,
            apoapsis=apo,
            dynamic_pressure=q,
        )

    def potential(self):
        d = self.telemetry()
        target_r = self.radius + self.target
        return -8 * (
            (d["energy"] / (-self.mu / (2 * target_r)) - 1).abs()
            + (d["momentum"] / (self.mu * target_r).sqrt() - 1).abs()
        )

    def available_thrust(self):
        s = self.stages[self.ids, self.stage]
        pressure = self.atmosphere_at(
            torch.hypot(self.state[:, 0], self.state[:, 1]) - self.radius
        )[:, 1]
        isp = s[:, 3] + (s[:, 4] - s[:, 3]) * (pressure / 101325).clamp(max=1)
        return torch.where(self.fuel > 1e-7, s[:, 2] * isp / s[:, 3], 0)

    def context(self):
        vc = (self.mu / (self.radius + self.target)).sqrt()
        tc = (self.radius + self.target) / vc
        dry, fuel, thrust, isp, isp_sl = self.stages.unbind(-1)
        wet = dry + fuel
        ignition_mass = self.payload[:, None] + torch.flip(
            torch.cumsum(torch.flip(wet, (-1,)), -1), (-1,)
        )
        active = self.stage_ids < self.stage_count[:, None]
        dv = isp * G0 * (ignition_mass / (ignition_mass - fuel).clamp(min=1e-20)).log()
        burn = fuel * isp * G0 / thrust.clamp(min=1e-20)
        context = torch.stack(
            (
                active.to(self.dtype),
                dry / self.initial_mass[:, None],
                fuel / self.initial_mass[:, None],
                (thrust / self.initial_mass[:, None] / G0).log1p(),
                isp * G0 / vc[:, None],
                isp_sl / isp.clamp(min=1e-20),
                dv / vc[:, None],
                burn / tc[:, None],
            ),
            -1,
        ).flatten(1)
        global_context = torch.stack(
            (
                self.payload / self.initial_mass,
                self.drag_area / self.initial_mass * 1000,
                (self.attitude_rate * tc / 360).log1p(),
                self.attitude_tau / tc,
                self.target / self.radius,
                self.stage_count / MAX_STAGES,
            ),
            -1,
        )
        return torch.cat((context, global_context), -1)

    def observation(self):
        d = self.telemetry()
        vc = (self.mu / (self.radius + self.target)).sqrt()
        stage_fuel = self.stages[self.ids, self.stage, 1]
        values = torch.stack(
            (
                d["altitude"] / self.target,
                d["vertical_speed"] / vc,
                d["horizontal_speed"] / vc,
                (d["apoapsis"] / self.target).clamp(-10, 10),
                d["periapsis"] / self.radius,
                self.fuel / stage_fuel,
                self.mass() / self.initial_mass,
                self.stage / (self.stage_count - 1).clamp(min=1),
                (self.pitch * math.pi / 180).sin(),
                (self.pitch * math.pi / 180).cos(),
                self.throttle,
                self.time / self.max_time,
                (d["dynamic_pressure"] / 40000).log1p(),
                (self.available_thrust() / self.mass() / G0).log1p(),
                (self.target - d["altitude"]) / self.target,
                d["energy"] / (-self.mu / (2 * (self.radius + self.target))),
            ),
            -1,
        )
        return torch.cat((values, self.context()), -1).clamp(-10, 10).float()

    def teacher(self):
        d = self.telemetry()
        alt, vr, vt, apo = (
            d[key]
            for key in ("altitude", "vertical_speed", "horizontal_speed", "apoapsis")
        )
        target = self.target + 1500
        circular = (self.mu / (self.radius + alt)).sqrt()
        pitch = (0.12 * (target - alt) / vt.clamp(min=100) - 0.22 * vr).clamp(-10, 25)
        pitch = torch.where(apo < target - 1000, pitch.clamp(min=10), pitch)
        throttle = torch.ones_like(pitch)
        coast = (apo >= target - 1000) & (alt < self.target - 7000) & (vr > 30)
        circularize = (apo >= target - 1000) & ~coast
        throttle = torch.where(coast, 0, throttle)
        throttle = torch.where(
            circularize, ((circular - vt) / 100).clamp(0, 1), throttle
        )
        pitch = torch.where(circularize, (-0.3 * vr).clamp(-10, 15), pitch)
        turn = (alt >= 1500) & (alt < 40000) & (apo < target)
        pitch = torch.where(turn, (88 - (alt - 1500) / 450).clamp(min=8), pitch)
        pitch = torch.where(alt < 1500, 90, pitch)
        throttle = torch.where(alt < 1500, 1, throttle)
        return torch.stack((2 * throttle - 1, (pitch + 10) / 50 - 1), -1).float()

    def derivative(self, state, mass, pitch, thrust):
        x, y, vx, vy = state.unbind(-1)
        radius = torch.hypot(x, y)
        ux, uy = x / radius, y / radius
        rho = self.atmosphere_at(radius - self.radius)[:, 0]
        omega = 2 * math.pi / self.rotation_period
        avx, avy = vx + omega * y, vy - omega * x
        drag = 0.5 * rho * self.drag_area * torch.hypot(avx, avy) / mass
        p = pitch * math.pi / 180
        ax = (
            -self.mu * x / radius**3
            - drag * avx
            + thrust / mass * (p.sin() * ux - p.cos() * uy)
        )
        ay = (
            -self.mu * y / radius**3
            - drag * avy
            + thrust / mass * (p.sin() * uy + p.cos() * ux)
        )
        return torch.stack((vx, vy, ax, ay), -1)

    @torch.no_grad()
    def step(self, action):
        if action.shape != (self.count, 2) or not bool(torch.isfinite(action).all()):
            raise ValueError("actions must be finite [worlds,2] values")
        action = action.to(self.dtype).clamp(-1, 1)
        self.throttle = (action[:, 0] + 1) / 2
        target_pitch = (action[:, 1] + 1) * 50 - 10
        remaining = torch.minimum(
            torch.full_like(self.time, self.control_dt),
            (self.max_time - self.time).clamp(min=0),
        )
        elapsed = torch.zeros_like(remaining)
        impact = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        # A bounded extra iteration per possible stage handles fuel exhaustion
        # inside a physical step without negative fuel or lost control time.
        standard_substeps = math.ceil(self.control_dt / self.physics_dt)
        for substep in range(standard_substeps + MAX_STAGES):
            continuing = (remaining > 1e-9) & ~impact
            # Normal fixed RK4 substeps stay asynchronous on the GPU. A single
            # host check after the normal interval decides whether any world
            # needs the rare fuel-boundary tail; no per-substep synchronization.
            if substep >= standard_substeps and not bool(continuing.any()):
                break
            advance = (
                (self.fuel <= 1e-7) & (self.stage < self.stage_count - 1) & continuing
            )
            self.stage += advance.long()
            self.fuel = torch.where(
                advance, self.stages[self.ids, self.stage, 1], self.fuel
            )
            s = self.stages[self.ids, self.stage]
            mdot = torch.where(
                self.fuel > 1e-7, self.throttle * s[:, 2] / (s[:, 3] * G0), 0
            )
            dt = torch.minimum(remaining, torch.full_like(remaining, self.physics_dt))
            dt = torch.where(
                mdot > 0, torch.minimum(dt, self.fuel / mdot.clamp(min=1e-20)), dt
            )
            dt = torch.where(continuing, dt, 0)
            pitch_change = ((target_pitch - self.pitch) / self.attitude_tau).clamp(
                min=-self.attitude_rate, max=self.attitude_rate
            ) * dt
            self.pitch += pitch_change
            thrust = self.available_thrust() * self.throttle
            mass = self.mass() - mdot * dt / 2
            half = dt[:, None] / 2
            k1 = self.derivative(self.state, mass, self.pitch, thrust)
            k2 = self.derivative(self.state + half * k1, mass, self.pitch, thrust)
            k3 = self.derivative(self.state + half * k2, mass, self.pitch, thrust)
            k4 = self.derivative(
                self.state + dt[:, None] * k3, mass, self.pitch, thrust
            )
            self.state += dt[:, None] * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            self.fuel = (self.fuel - mdot * dt).clamp(min=0)
            self.time += dt
            elapsed += dt
            remaining = (remaining - dt).clamp(min=0)
            impact |= torch.hypot(self.state[:, 0], self.state[:, 1]) <= self.radius
        if bool(((remaining > 1e-8) & ~impact).any()):
            raise RuntimeError("integration exhausted stage/substep bound")
        d = self.telemetry()
        success = (
            (d["energy"] < 0)
            & (
                d["periapsis"]
                >= torch.maximum(
                    torch.full_like(self.target, 71000),
                    self.target - self.orbit_tolerance,
                )
            )
            & (d["apoapsis"] <= self.target + self.orbit_tolerance)
        )
        escape = d["energy"] >= 0
        fuel_empty = (self.stage == self.stage_count - 1) & (self.fuel <= 1e-7)
        terminated = success | impact | escape | fuel_empty
        truncated = ~terminated & (self.time >= self.max_time - 1e-8)
        phi = self.potential()
        shaping = self.gamma * torch.where(terminated, 0, phi) - self.previous_potential
        reward = (
            shaping
            + torch.where(terminated, torch.where(success, 100, -30), 0)
            - 0.003 * elapsed
        )
        self.previous_potential = phi
        self.episode_return += reward
        # Reason codes: success, impact, escape, fuel, time. Success has priority.
        reason = torch.where(
            success,
            1,
            torch.where(
                impact,
                2,
                torch.where(
                    escape, 3, torch.where(fuel_empty, 4, torch.where(truncated, 5, 0))
                ),
            ),
        )
        info = dict(
            success=success,
            reason=reason,
            episode_return=self.episode_return.clone(),
            time=self.time.clone(),
            apoapsis=d["apoapsis"],
            periapsis=d["periapsis"],
        )
        return self.observation(), reward.float(), terminated, truncated, info

    def export_config(self, index):
        """Freeze one generated craft into the independent scalar simulator."""
        values = self.stages[index, : int(self.stage_count[index])].cpu().tolist()
        return EnvConfig(
            payload_mass=float(self.payload[index]),
            stages=tuple(Stage(*row) for row in values),
            drag_area=float(self.drag_area[index]),
            attitude_rate=float(self.attitude_rate[index]),
            attitude_tau=float(self.attitude_tau[index]),
            target_altitude=float(self.target[index]),
            max_time=self.max_time,
            physics_dt=self.physics_dt,
            randomization=0,
            orbit_tolerance=self.orbit_tolerance,
        )

    def cpu_reference(self, index):
        return HeadlessOrbiterEnv(self.export_config(index))
