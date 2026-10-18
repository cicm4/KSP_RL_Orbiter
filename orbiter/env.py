"""Fast planar launch surrogate, not an exact reproduction of Unity/KSP physics.

SI units, inertial Cartesian translation, rotating atmosphere, inverse-square
gravity, pressure-dependent engines, propellant consumption and automatic staging.
Pitch is an autopilot attitude SETPOINT, never a raw pitch-torque input.
"""
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

G0 = 9.80665


@dataclass(frozen=True)
class Stage:
    dry_mass: float
    fuel_mass: float
    thrust_vac: float
    isp_vac: float
    isp_sl: float


@dataclass
class EnvConfig:
    mu: float = 3.5316e12
    radius: float = 600_000.0
    rotation_period: float = 21_549.425
    atmosphere_depth: float = 70_000.0
    target_altitude: float = 80_000.0
    orbit_tolerance: float = 10_000.0
    payload_mass: float = 1500.0
    # Explicit generic two-stage training vehicle, replaced by a captured craft
    # before any claim about transfer to a particular KSP rocket.
    stages: tuple[Stage, ...] = field(default_factory=lambda: (
        Stage(3500, 22_000, 650_000, 320, 280),
        Stage(1000, 6000, 120_000, 345, 85),
    ))
    drag_area: float = 1.5  # Cd*A, m^2; needs flight-data identification.
    # Optional local aerodynamic approximation, in effective square metres.
    # Zero defaults preserve older profiles. Fit each craft/flight regime;
    # one coefficient set does not reproduce KSP's Mach and staging dependence.
    drag_alpha_area: float = 0.0
    lift_area: float = 0.0
    attitude_rate: float = 8.0  # degrees/second, autopilot approximation.
    attitude_tau: float = 0.8
    circularization_margin: float = 7000.0  # Teacher coast cutoff below target, metres.
    control_dt: float = 2.0
    physics_dt: float = 0.25
    max_time: float = 700.0
    gamma: float = 0.999
    randomization: float = 0.05
    curriculum: bool = False
    residual: bool = False
    # Approximate stock Kerbin density/pressure, superseded by live capture.
    atmosphere: tuple[tuple[float, float, float], ...] = (
        (0., 1.225, 101325.), (5000., .735, 54050.),
        (10000., .414, 26436.), (15000., .195, 12045.),
        (20000., .089, 5475.), (25000., .040, 2511.),
        (30000., .018, 1172.), (40000., .0039, 287.),
        (50000., .00098, 80.), (60000., .00031, 20.),
        (70000., 0., 0.),
    )

    def __post_init__(self):
        def finite_number(value):
            try:
                return not isinstance(value, bool) and math.isfinite(value)
            except (TypeError, ValueError):
                return False

        if any(not finite_number(value) or value < 0
               for value in (self.drag_area, self.drag_alpha_area, self.lift_area)):
            raise ValueError('aerodynamic areas must be finite and nonnegative')
        for name in ('mu', 'radius', 'rotation_period', 'atmosphere_depth',
                     'target_altitude', 'orbit_tolerance', 'payload_mass',
                     'attitude_rate', 'attitude_tau', 'circularization_margin',
                     'control_dt', 'physics_dt', 'max_time', 'gamma', 'randomization'):
            if not finite_number(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        for name in ('mu', 'radius', 'target_altitude', 'attitude_tau',
                     'control_dt', 'physics_dt', 'max_time'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.rotation_period == 0:
            raise ValueError('rotation_period must be nonzero')
        for name in ('atmosphere_depth', 'orbit_tolerance', 'payload_mass',
                     'attitude_rate', 'circularization_margin'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative')
        if not 0 < self.gamma <= 1:
            raise ValueError('gamma must be in (0, 1]')
        if not 0 <= self.randomization < 0.5:
            raise ValueError('randomization must be in [0, .5)')
        if not self.stages or any(
                not finite_number(value) or value <= 0
                for stage in self.stages
                for value in (stage.dry_mass, stage.fuel_mass, stage.thrust_vac, stage.isp_vac, stage.isp_sl)):
            raise ValueError('each stage needs finite positive mass, thrust, and Isp')
        if self.atmosphere_depth > 0 and len(self.atmosphere) < 2:
            raise ValueError('an atmospheric body needs at least two atmosphere rows')
        previous_altitude = None
        for row in self.atmosphere:
            if (not isinstance(row, (tuple, list)) or len(row) != 3
                    or any(not finite_number(value) or value < 0 for value in row)):
                raise ValueError('atmosphere rows need three finite nonnegative values')
            if previous_altitude is not None and row[0] <= previous_altitude:
                raise ValueError('atmosphere altitudes must be strictly increasing')
            previous_altitude = row[0]

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['stages'] = tuple(Stage(**s) for s in data['stages'])
        if 'atmosphere' in data:
            data['atmosphere'] = tuple(tuple(row) for row in data['atmosphere'])
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + '\n')


def orbital_elements(x, y, vx, vy, mu, radius):
    r = math.hypot(x, y)
    speed2 = vx * vx + vy * vy
    energy = speed2 / 2 - mu / r
    angular_momentum = x * vy - y * vx
    eccentricity = math.sqrt(max(0., 1 + 2 * energy * angular_momentum ** 2 / mu ** 2))
    p = angular_momentum ** 2 / mu
    periapsis = p / (1 + eccentricity) - radius
    # A hyperbolic/parabolic orbit has no finite apoapsis.
    apoapsis = p / (1 - eccentricity) - radius if energy < 0 and eccentricity < 1 else math.inf
    return apoapsis, periapsis, energy, angular_momentum


def shaped_reward(previous_potential, next_potential, gamma, terminated, success, dt):
    """Absorbing terminal states have Phi=0; time limits retain bootstrap Phi.

    The same gamma MUST be used by the learner. Terminal cancellation prevents
    a partial ascent or a crash from cashing out shaping as task success.
    """
    shaping = gamma * (0. if terminated else next_potential) - previous_potential
    task = (100. if success else -30.) if terminated else 0.
    time_cost = -0.003 * dt
    return shaping + task + time_cost, shaping, task


def build_observation(config, telemetry, *, fuel_fraction, mass_fraction, stage,
                      pitch, throttle, elapsed, thrust_to_weight):
    """Shared SI telemetry-to-policy contract for the surrogate and live KSP."""
    c, d = config, telemetry
    v = math.sqrt(c.mu / (c.radius + c.target_altitude))
    bounded_apo = 10. if not math.isfinite(d['apoapsis']) else np.clip(d['apoapsis']/c.target_altitude, -10, 10)
    return np.clip(np.array([
        d['altitude']/c.target_altitude, d['vertical_speed']/v,
        d['horizontal_speed']/v, bounded_apo,
        d['periapsis']/c.radius, fuel_fraction,
        mass_fraction, stage/max(len(c.stages)-1, 1),
        math.sin(math.radians(pitch)), math.cos(math.radians(pitch)),
        throttle, elapsed/c.max_time, d['dynamic_pressure']/40000,
        thrust_to_weight,
        (c.target_altitude-d['altitude'])/c.target_altitude,
        d['energy']/(-c.mu/(2*(c.radius+c.target_altitude))),
    ], dtype=np.float32), -10, 10)


class HeadlessOrbiterEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config=None):
        self.config = config or EnvConfig()
        self.action_space = spaces.Box(-1., 1., (2,), np.float32)
        self.observation_space = spaces.Box(-10., 10., (16,), np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.config
        scale = lambda: 1 + self.np_random.uniform(-c.randomization, c.randomization)
        self.stages = tuple(replace(s, fuel_mass=s.fuel_mass * scale(), thrust_vac=s.thrust_vac * scale()) for s in c.stages)
        self.drag_area = c.drag_area * scale()
        self.attitude_rate = c.attitude_rate * scale()
        self.stage = 0
        self.fuel = self.stages[0].fuel_mass
        self.initial_mass = c.payload_mass + sum(s.dry_mass + s.fuel_mass for s in self.stages)
        self.x, self.y = c.radius + 80., 0.
        self.vx, self.vy = 0., self.x * 2 * math.pi / c.rotation_period
        self.pitch, self.throttle, self.t = 90., 0., 0.
        self.max_altitude, self.max_q = 80., 0.
        self._ended = False
        self._potential_prev = self.potential()
        return self.observation(), self.info('')

    @property
    def mass(self):
        return self.config.payload_mass + self.stages[self.stage].dry_mass + self.fuel + sum(s.dry_mass + s.fuel_mass for s in self.stages[self.stage + 1:])

    def atmosphere_at(self, altitude):
        if altitude >= self.config.atmosphere_depth:
            return 0., 0.
        table = self.config.atmosphere
        h = max(0., altitude)
        for lower, upper in zip(table, table[1:]):
            if h <= upper[0]:
                f = (h - lower[0]) / (upper[0] - lower[0])
                # Log interpolation wherever both endpoints are positive.
                return tuple(math.exp(math.log(a) * (1 - f) + math.log(b) * f) if a > 0 and b > 0 else a * (1 - f) + b * f for a, b in zip(lower[1:], upper[1:]))
        return 0., 0.

    def telemetry(self):
        c = self.config
        r = math.hypot(self.x, self.y)
        vr = (self.x * self.vx + self.y * self.vy) / r
        vt = (self.x * self.vy - self.y * self.vx) / r
        apo, peri, energy, momentum = orbital_elements(self.x, self.y, self.vx, self.vy, c.mu, c.radius)
        rho, _ = self.atmosphere_at(r - c.radius)
        surface_vt = vt - 2 * math.pi / c.rotation_period * r
        return dict(altitude=r-c.radius, vertical_speed=vr, horizontal_speed=vt,
                    apoapsis=apo, periapsis=peri, energy=energy, momentum=momentum,
                    dynamic_pressure=.5*rho*(vr*vr+surface_vt*surface_vt))

    def potential(self):
        c = self.config
        d = self.telemetry()
        target_radius = c.radius + c.target_altitude
        target_energy = -c.mu / (2 * target_radius)
        target_momentum = math.sqrt(c.mu * target_radius)
        # Energy alone rewards vertical suborbital hops. Angular momentum
        # supplies the missing incentive to build useful horizontal velocity.
        return -8 * (abs(d['energy'] / target_energy - 1) + abs(d['momentum'] / target_momentum - 1))

    def observation(self):
        return build_observation(self.config, self.telemetry(),
            fuel_fraction=self.fuel/self.stages[self.stage].fuel_mass,
            mass_fraction=self.mass/self.initial_mass, stage=self.stage,
            pitch=self.pitch, throttle=self.throttle, elapsed=self.t,
            thrust_to_weight=self.available_thrust()/self.mass/G0)

    def available_thrust(self):
        s = self.stages[self.stage]
        _, pressure = self.atmosphere_at(math.hypot(self.x, self.y)-self.config.radius)
        isp = s.isp_vac + (s.isp_sl-s.isp_vac)*min(1., pressure/101325.)
        return s.thrust_vac * isp/s.isp_vac if self.fuel > 1e-7 else 0.

    def _derivative(self, state, mass, pitch, thrust):
        x, y, vx, vy = state
        r = math.hypot(x, y)
        ux, uy = x/r, y/r
        rho, _ = self.atmosphere_at(r-self.config.radius)
        omega = 2 * math.pi/self.config.rotation_period
        avx, avy = vx+omega*y, vy-omega*x
        p = math.radians(pitch)
        speed = math.hypot(avx, avy)
        effective_drag_area, effective_lift_area = self.drag_area, 0.
        if speed > 0 and (self.config.drag_alpha_area or self.config.lift_area):
            radial_speed = avx*ux + avy*uy
            east_speed = -avx*uy + avy*ux
            alpha = p - math.atan2(radial_speed, east_speed)
            effective_drag_area += self.config.drag_alpha_area * math.sin(alpha)**2
            effective_lift_area = self.config.lift_area * math.sin(2*alpha)
        drag = .5*rho*effective_drag_area*speed/mass
        lift = .5*rho*effective_lift_area*speed/mass
        # (avy, -avx) is the planar normal pointing radially outward for
        # eastward horizontal flight. Lift is perpendicular to relative airspeed.
        ax = -self.config.mu*x/r**3 - drag*avx + lift*avy + thrust/mass*(math.sin(p)*ux-math.cos(p)*uy)
        ay = -self.config.mu*y/r**3 - drag*avy - lift*avx + thrust/mass*(math.sin(p)*uy+math.cos(p)*ux)
        return np.array([vx, vy, ax, ay])

    def step(self, action):
        if self._ended:
            raise RuntimeError('reset() required after episode end')
        a = np.asarray(action, dtype=float)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError('action must contain two finite values')
        a = np.clip(a, -1, 1)
        if self.config.residual:
            from .guidance import expert_action
            base = expert_action(self)
            a = np.clip(base + a*np.array([.25, .2]), -1, 1)
        self.throttle = float((a[0]+1)/2)
        target_pitch = float((a[1]+1)*50-10)
        c = self.config
        elapsed = 0.
        impact = False
        control_duration = min(c.control_dt, max(0., c.max_time-self.t))
        while elapsed < control_duration-1e-9:
            dt = min(c.physics_dt, control_duration-elapsed)
            if self.fuel <= 1e-7 and self.stage < len(self.stages)-1:
                self.stage += 1
                self.fuel = self.stages[self.stage].fuel_mass
            s = self.stages[self.stage]
            mdot = self.throttle*s.thrust_vac/(s.isp_vac*G0) if self.fuel > 1e-7 else 0.
            if mdot > 0:
                dt = min(dt, self.fuel/mdot)
            if dt <= 1e-9:
                self.fuel = 0.
                continue
            pitch_change = np.clip((target_pitch-self.pitch)/c.attitude_tau, -self.attitude_rate, self.attitude_rate)*dt
            self.pitch += float(pitch_change)
            thrust = self.available_thrust()*self.throttle
            mass = self.mass-mdot*dt/2
            state = np.array([self.x, self.y, self.vx, self.vy])
            f = lambda z: self._derivative(z, mass, self.pitch, thrust)
            k1 = f(state)
            k2 = f(state+dt*k1/2)
            k3 = f(state+dt*k2/2)
            k4 = f(state+dt*k3)
            self.x, self.y, self.vx, self.vy = state+dt*(k1+2*k2+2*k3+k4)/6
            self.fuel = max(0., self.fuel-mdot*dt)
            self.t += dt
            elapsed += dt
            d = self.telemetry()
            self.max_altitude = max(self.max_altitude, d['altitude'])
            self.max_q = max(self.max_q, d['dynamic_pressure'])
            if d['altitude'] <= 0:
                impact = True
                break
        d = self.telemetry()
        success = (d['energy'] < 0 and d['periapsis'] >= max(c.atmosphere_depth+1000, c.target_altitude-c.orbit_tolerance)
                   and d['apoapsis'] <= c.target_altitude+c.orbit_tolerance)
        reason = ('orbit_achieved' if success else 'impact' if impact else
                  'escape' if d['energy'] >= 0 else
                  'out_of_fuel' if self.stage == len(self.stages)-1 and self.fuel <= 1e-7 else '')
        terminated = bool(reason)
        truncated = bool(not terminated and self.t >= c.max_time-1e-8)
        if truncated:
            reason = 'time_limit'
        phi = self.potential()
        reward, shaping, task = shaped_reward(self._potential_prev, phi, c.gamma, terminated, success, elapsed)
        self._potential_prev = phi
        self._ended = terminated or truncated
        info = self.info(reason)
        info.update(reward_shaping=shaping, reward_task=task)
        return self.observation(), float(reward), terminated, truncated, info

    def info(self, reason):
        return dict(self.telemetry(), success=reason=='orbit_achieved', termination_reason=reason,
                    fuel_remaining=self.fuel, fuel_remaining_frac=self.fuel/self.stages[self.stage].fuel_mass,
                    mass=self.mass, stage=self.stage, pitch=self.pitch, throttle=self.throttle,
                    time=self.t, max_altitude=self.max_altitude, max_q=self.max_q)
