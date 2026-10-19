"""Versioned, vehicle-conditioned ascent with randomized planar aerodynamics.

This experimental surrogate adds angle-dependent drag and signed lift to the
frozen v3 vehicle environment. It shares that environment's integration,
engines, staging and reward. The coefficients describe effective areas rather
than KSP's full part-, Mach- or staging-dependent aerodynamic system.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math

import numpy as np
import torch

from .env import EnvConfig, G0
from .generalized_env import (
    BatchedAscent,
    MAX_STAGES,
    VehicleEnvelope,
    cpu_observation as base_cpu_observation,
    normalization_audit as base_normalization_audit,
)

TASK = "vehicle_conditioned_aero_ascent_v4"
FEATURE_SCHEMA = "ascent16_log_q_twr_stage4x8_log_thrust_globals6_log_slew_aero2_v4"
OBSERVATIONS = 56


@dataclass(frozen=True)
class AeroVehicleEnvelope(VehicleEnvelope):
    """Independent effective-area ranges in square metres per initial tonne.

    The defaults include the locally fitted Transfer rocket's alpha-drag and
    lift coefficients (about 0.4784 and 0.3099 m²/t). Inclusion is a distribution
    choice, not evidence that a policy transfers to that or another KSP craft.
    """

    drag_alpha_area_per_tonne_min: float = 0.0
    drag_alpha_area_per_tonne_max: float = 1.0
    lift_area_per_tonne_min: float = 0.0
    lift_area_per_tonne_max: float = 0.7

    def __post_init__(self):
        # VehicleEnvelope validates asdict(self), so calling its method directly
        # would incorrectly reject these intentionally zero-inclusive ranges.
        VehicleEnvelope(
            **{item.name: getattr(self, item.name) for item in fields(VehicleEnvelope)}
        )
        for prefix in ("drag_alpha_area_per_tonne", "lift_area_per_tonne"):
            low, high = (getattr(self, prefix + suffix) for suffix in ("_min", "_max"))
            for value in (low, high):
                try:
                    valid = (
                        not isinstance(value, bool)
                        and math.isfinite(value)
                        and value >= 0
                    )
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    raise ValueError(f"{prefix} must be finite and nonnegative")
            if low > high:
                raise ValueError(f"invalid {prefix} range")


def normalization_audit(envelope):
    """Audit the two additional normalized coefficients before clipping."""
    result = base_normalization_audit(envelope)
    result["feature_schema"] = FEATURE_SCHEMA
    result["upper_bounds"].update(
        drag_alpha_area_per_tonne=envelope.drag_alpha_area_per_tonne_max,
        lift_area_per_tonne=envelope.lift_area_per_tonne_max,
    )
    result["all_craft_and_actuator_features_below_clip"] = all(
        value < 10 for value in result["upper_bounds"].values()
    )
    result["task_scope"] += "; angle-dependent planar areas, not KSP aero fidelity"
    return result


def cpu_observation(env):
    """Independent scalar observation builder using the frozen 54-float prefix."""
    return np.concatenate(
        (
            base_cpu_observation(env),
            np.clip(
                np.asarray(
                    [env.config.drag_alpha_area, env.config.lift_area],
                    dtype=np.float64,
                )
                / env.initial_mass
                * 1000,
                -10,
                10,
            ).astype(np.float32),
        )
    )


class BatchedAeroAscent(BatchedAscent):
    """v3 RK4 worlds with two independently sampled aerodynamic coefficients."""

    def __init__(self, count=1024, *, envelope=None, **kwargs):
        if envelope is not None and not isinstance(envelope, AeroVehicleEnvelope):
            raise TypeError("the v4 environment requires an AeroVehicleEnvelope")
        # The parent constructor dispatches reset() and observation(). Allocate
        # lazily in reset after the parent establishes count/device/dtype.
        self.drag_alpha_area = self.lift_area = None
        super().__init__(count, envelope=envelope or AeroVehicleEnvelope(), **kwargs)

    @torch.no_grad()
    def reset(self, mask=None):
        if self.drag_alpha_area is None:
            self.drag_alpha_area, self.lift_area = (
                self.zeros(self.count),
                self.zeros(self.count),
            )
        if mask is None:
            mask = torch.ones(self.count, dtype=torch.bool, device=self.device)
        super().reset(mask)
        selected = self.ids[mask]
        n = selected.numel()
        if n:
            for name in ("drag_alpha_area", "lift_area"):
                prefix = name + "_per_tonne"
                low, high = (
                    getattr(self.envelope, prefix + suffix)
                    for suffix in ("_min", "_max")
                )
                # Fixed zero coefficients consume no RNG draws, making this
                # exact ablation share v3 craft sampling even after resets.
                coefficient = (
                    self.zeros(n) + low
                    if low == high
                    else self.uniform((n,), low, high)
                )
                getattr(self, name)[selected] = (
                    self.initial_mass[selected] / 1000 * coefficient
                )
        return self.observation()

    def observation(self):
        aero = torch.stack((self.drag_alpha_area, self.lift_area), -1)
        aero = (aero / self.initial_mass[:, None] * 1000).clamp(-10, 10).float()
        return torch.cat((super().observation(), aero), -1)

    def derivative(self, state, mass, pitch, thrust):
        # Preserve the parent's gravity, constant drag and thrust exactly; add
        # only the two force terms also present in the independent scalar model.
        derivative = super().derivative(state, mass, pitch, thrust)
        x, y, vx, vy = state.unbind(-1)
        radius = torch.hypot(x, y)
        ux, uy = x / radius, y / radius
        omega = 2 * math.pi / self.rotation_period
        avx, avy = vx + omega * y, vy - omega * x
        speed = torch.hypot(avx, avy)
        radial_speed, east_speed = avx * ux + avy * uy, -avx * uy + avy * ux
        alpha = pitch * math.pi / 180 - torch.atan2(radial_speed, east_speed)
        rho = self.atmosphere_at(radius - self.radius)[:, 0]
        pressure_over_speed_mass = 0.5 * rho * speed / mass
        drag = pressure_over_speed_mass * self.drag_alpha_area * alpha.sin().square()
        lift = pressure_over_speed_mass * self.lift_area * (2 * alpha).sin()
        correction = torch.stack(
            (-drag * avx + lift * avy, -drag * avy - lift * avx), -1
        )
        return torch.cat((derivative[:, :2], derivative[:, 2:] + correction), -1)

    def export_config(self, index):
        """Freeze every physical parameter, including a loaded atmosphere."""
        return replace(
            super().export_config(index),
            mu=self.mu,
            radius=self.radius,
            rotation_period=self.rotation_period,
            atmosphere_depth=self.atmosphere_depth,
            atmosphere=tuple(tuple(row) for row in self.atmosphere.cpu().tolist()),
            control_dt=self.control_dt,
            gamma=self.gamma,
            drag_alpha_area=float(self.drag_alpha_area[index]),
            lift_area=float(self.lift_area[index]),
        )

    @torch.no_grad()
    def load_configs(self, configs):
        """Reset a full batch to an explicit, paired Kerbin evaluation bank.

        Each EnvConfig must disable randomization, residual actions and the
        curriculum. All worlds must share global physics, timing and atmosphere;
        targets and craft may differ. Subsequent reset() draws from the envelope
        again, so evaluation callers should track completed worlds explicitly.
        """
        configs = tuple(configs)
        if len(configs) != self.count or any(
            not isinstance(c, EnvConfig) for c in configs
        ):
            raise ValueError("provide exactly one EnvConfig per world")
        shared = (
            "mu",
            "radius",
            "rotation_period",
            "atmosphere_depth",
            "atmosphere",
            "control_dt",
            "physics_dt",
            "max_time",
            "gamma",
            "orbit_tolerance",
        )
        first = configs[0]
        for config in configs:
            config.__post_init__()
            if config.randomization or config.residual or config.curriculum:
                raise ValueError(
                    "explicit cases require no randomization, residual or curriculum"
                )
            if len(config.stages) > MAX_STAGES:
                raise ValueError("at most four sequential stages are supported")
            if config.payload_mass <= 0:
                raise ValueError(
                    "the inherited morphology context requires positive payload mass"
                )
            if config.atmosphere_depth != 70000:
                raise ValueError(
                    "the inherited ascent success criterion requires Kerbin's 70 km atmosphere"
                )
            if not 0 < config.physics_dt <= 2:
                raise ValueError("physics_dt must be in (0,2]")
            if any(getattr(config, key) != getattr(first, key) for key in shared):
                raise ValueError(
                    "explicit cases must share global physics, timing and atmosphere"
                )
        # Validate the complete bank before changing the current environment.
        for name in shared:
            if name != "atmosphere":
                setattr(self, name, getattr(first, name))
        self.atmosphere = torch.tensor(
            first.atmosphere, device=self.device, dtype=self.dtype
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
        self.stages.zero_()
        for index, config in enumerate(configs):
            self.stages[index, : len(config.stages)] = torch.tensor(
                [
                    [
                        stage.dry_mass,
                        stage.fuel_mass,
                        stage.thrust_vac,
                        stage.isp_vac,
                        stage.isp_sl,
                    ]
                    for stage in config.stages
                ],
                device=self.device,
                dtype=self.dtype,
            )
            self.stage_count[index] = len(config.stages)
            self.payload[index] = config.payload_mass
            self.initial_mass[index] = config.payload_mass + sum(
                stage.dry_mass + stage.fuel_mass for stage in config.stages
            )
            self.target[index] = config.target_altitude
            self.drag_area[index] = config.drag_area
            self.drag_alpha_area[index] = config.drag_alpha_area
            self.lift_area[index] = config.lift_area
            self.attitude_rate[index], self.attitude_tau[index] = (
                config.attitude_rate,
                config.attitude_tau,
            )
        self.stage.zero_()
        self.fuel.copy_(self.stages[:, 0, 1])
        self.state[:] = torch.tensor(
            [
                self.radius + 80,
                0,
                0,
                (self.radius + 80) * 2 * math.pi / self.rotation_period,
            ],
            device=self.device,
            dtype=self.dtype,
        )
        self.pitch.fill_(90)
        self.throttle.zero_()
        self.time.zero_()
        self.episode_return.zero_()
        self.episode_task_return.zero_()
        self.discounted_task_return.zero_()
        self.episode_discount.fill_(1)
        self.steps.zero_()
        upper = self.payload.clone()
        self.ideal_delta_v.zero_()
        for stage in reversed(range(MAX_STAGES)):
            dry, fuel, _, isp, _ = self.stages[:, stage].unbind(-1)
            self.ideal_delta_v += (
                isp * G0 * ((upper + dry + fuel) / (upper + dry).clamp(min=1e-20)).log()
            )
            upper += dry + fuel
        self.previous_potential.copy_(self.potential())
        return self.observation()
