"""GPU 3D finite-thrust orbital benchmark across the captured stock system.

One policy shares all 17 bodies. Planet/moon cases begin on local inbound
hyperbolas; Sun cases begin in a bound heliocentric orbit requiring adjustment.
This is NOT an interbody transfer or a surface launch benchmark. The active
liquid propulsion stage is represented by mass/dry mass/thrust/Isp and a finite
pointing rate. Aerocapture, heating, terrain, and stage graphs are not modeled.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import torch

from .capture_env import CaptureConfig
from .system import SystemCatalog
from .tensor_memo import TensorMemo

TASK = "stock_all_body_batched_orbital_v1"
OBSERVATIONS = 32
# v2: log-scaled remaining delta-v (v1 pins it at the clip for every small
# body), apse/plane errors in units of the success tolerance, and a bounded
# tolerance-scaled shaping term. Without it the Sun's start potential is ~1e-5
# and its band is effectively a sparse reward.
VERSIONS = {1: (TASK, OBSERVATIONS), 2: ("stock_all_body_batched_orbital_v2", 35)}
G0 = 9.80665


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-30)


class BatchedCapture:
    observations, actions = OBSERVATIONS, 3

    def __init__(self, count=1024, *, device="cpu", seed=0, bodies=None, version=1):
        if count < 1:
            raise ValueError("count must be positive")
        if version not in VERSIONS:
            raise ValueError("unknown orbital benchmark version")
        self.version = version
        self.task, self.observations = VERSIONS[version]
        self.count, self.device, self.dtype = count, torch.device(device), torch.float64
        self._telemetry_memo, self._observation_memo = TensorMemo(), TensorMemo()
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.catalog = SystemCatalog.load()
        self.names = tuple(sorted(bodies or self.catalog.bodies))
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("body names must be distinct and nonempty")
        records = [self.catalog.bodies[n] for n in self.names]
        self.catalog_sha256 = hashlib.sha256(json.dumps(
            records, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.ids = torch.arange(count, device=self.device)
        self.body = self.ids % len(self.names)
        self.body_mu = self.tensor([b["gravitational_parameter"] for b in records])
        self.body_radius = self.tensor([b["equatorial_radius"] for b in records])
        self.body_atmosphere = self.tensor([b["atmosphere_depth"] for b in records])
        # Sun has no finite SOI. Its artificial outer numerical guard is
        # explicitly marked in exported cases and never a stock-system claim.
        self.body_soi = self.tensor([b["sphere_of_influence"] or 1e18 for b in records])
        self.body_sun = torch.tensor([b["parent"] is None for b in records],
                                     device=self.device)
        for name in ("mu", "radius", "atmosphere_depth", "soi", "goal", "safe",
                     "tolerance", "V", "T", "control_dt", "physics_dt", "max_time",
                     "initial_mass", "mass", "dry_mass", "thrust", "isp", "time",
                     "pointing_rate", "previous_potential", "episode_return"):
            setattr(self, name, self.zeros(count))
        self.r, self.v, self.direction, self.goal_normal = [self.zeros(count, 3) for _ in range(4)]
        self.hold = torch.zeros(count, device=self.device, dtype=torch.long)
        self.reset(initial=True)

    def zeros(self, *shape):
        return torch.zeros(*shape, device=self.device, dtype=self.dtype)

    def tensor(self, x):
        return torch.tensor(x, device=self.device, dtype=self.dtype)

    def uniform(self, n, low, high):
        return low + (high-low) * torch.rand(n, generator=self.generator,
                                           device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def reset(self, mask=None, *, initial=False):
        mask = torch.ones(self.count, device=self.device, dtype=torch.bool) if mask is None else mask
        ids = self.ids[mask]
        n = ids.numel()
        if not n:
            return self.observation()
        if not initial:
            self.body[ids] = torch.randint(len(self.names), (n,), generator=self.generator,
                                          device=self.device)
        b = self.body[ids]
        self.mu[ids], self.radius[ids] = self.body_mu[b], self.body_radius[b]
        self.atmosphere_depth[ids], self.soi[ids] = self.body_atmosphere[b], self.body_soi[b]
        altitude = torch.maximum(torch.full_like(self.radius[ids], 50000),
                                 torch.maximum(self.body_atmosphere[b]+20000,
                                               .1*self.body_radius[b]))
        altitude *= self.uniform(n, .9, 1.2)
        altitude = torch.maximum(altitude, self.body_atmosphere[b]+10000)
        self.goal[ids] = self.radius[ids] + altitude
        self.safe[ids] = self.radius[ids] + self.atmosphere_depth[ids] + 1000
        self.tolerance[ids] = torch.minimum(5000+torch.zeros_like(altitude),
                                            .15*(self.goal[ids]-self.safe[ids]))
        self.V[ids] = (self.mu[ids]/self.goal[ids]).sqrt()
        self.T[ids] = self.goal[ids]/self.V[ids]
        # Fixed dimensionless decision interval, eight RK4 physical substeps.
        self.control_dt[ids] = .01*self.T[ids]
        self.physics_dt[ids] = self.control_dt[ids]/8
        self.max_time[ids] = 4*self.T[ids]
        self.initial_mass[ids] = self.uniform(n, math.log(300), math.log(30000)).exp()
        self.mass[ids] = self.initial_mass[ids]
        self.dry_mass[ids] = self.mass[ids]*self.uniform(n, .05, .4)
        self.thrust[ids] = self.mass[ids]*self.uniform(n, math.log(.5), math.log(30)).exp()
        self.isp[ids] = self.uniform(n, 280, 450)
        self.pointing_rate[ids] = self.uniform(n, 5, 45)*math.pi/180
        sun = self.body_sun[b]
        radial = unit(torch.randn(n, 3, device=self.device, dtype=self.dtype,
                                  generator=self.generator))
        raw = torch.randn(n, 3, device=self.device, dtype=self.dtype, generator=self.generator)
        normal = unit(raw-(raw*radial).sum(-1, keepdim=True)*radial)
        tangent = torch.linalg.cross(normal, radial)
        tilt = self.uniform(n, -3, 3)*math.pi/180
        tilt = torch.where(sun, tilt/60, tilt)
        arrival_tangent = tilt.cos()[:, None]*tangent + tilt.sin()[:, None]*normal
        peri = self.goal[ids]*self.uniform(n, .995, 1.005)
        start_r = self.goal[ids]*torch.where(sun, 1.0005, 1.04)
        vinf = self.V[ids]*self.uniform(n, .2, .7)
        h = peri*(vinf.square()+2*self.mu[ids]/peri).sqrt()
        vt = h/start_r
        vr = -(vinf.square()+2*self.mu[ids]/start_r-vt.square()).clamp(min=0).sqrt()
        # Heliocentric tasks are bound orbit adjustment, not unphysical local
        # arrival from an exterior Sun SOI with a fictitious launch surface.
        vt = torch.where(sun, (self.mu[ids]/start_r).sqrt()*self.uniform(n, .998, 1.002), vt)
        vr = torch.where(sun, -.001*self.V[ids], vr)
        self.r[ids] = start_r[:, None]*radial
        self.v[ids] = vr[:, None]*radial + vt[:, None]*arrival_tangent
        self.goal_normal[ids] = normal
        self.direction[ids] = -unit(self.v[ids])
        self.time[ids], self.hold[ids], self.episode_return[ids] = 0, 0, 0
        self.previous_potential[ids] = self.potential()[ids]
        return self.observation()

    # Every tensor attribute read by _telemetry/_observation. Exactness of the
    # memo depends on these lists; tests audit them by recording attribute use.
    TELEMETRY_INPUTS = ("r", "v", "mu", "goal_normal", "safe", "goal", "tolerance",
                        "thrust", "mass", "dry_mass")
    OBSERVATION_INPUTS = TELEMETRY_INPUTS + (
        "direction", "initial_mass", "isp", "V", "T", "radius", "soi", "time",
        "max_time", "control_dt")

    def telemetry(self):
        """Osculating-orbit telemetry, memoized exactly on the current state."""
        return self._telemetry_memo.get(
            [getattr(self, name) for name in self.TELEMETRY_INPUTS], (), self._telemetry)

    def _telemetry(self):
        rmag = self.r.norm(dim=-1)
        radial = self.r/rmag[:, None]
        h = torch.linalg.cross(self.r, self.v)
        hmag = h.norm(dim=-1)
        normal = unit(h)
        tangent = torch.linalg.cross(normal, radial)
        basis = torch.stack((radial, tangent, normal), -1)
        local_v = torch.bmm(basis.transpose(1, 2), self.v[:, :, None]).squeeze(-1)
        energy = self.v.square().sum(-1)/2-self.mu/rmag
        ecc = (1+2*energy*hmag.square()/self.mu.square()).clamp(min=0).sqrt()
        parameter = hmag.square()/self.mu
        peri = parameter/(1+ecc)
        apo = torch.where((energy < 0)&(ecc < 1), parameter/(1-ecc).clamp(min=1e-15), torch.inf)
        cosine = (normal*self.goal_normal).sum(-1).clamp(-1, 1)
        plane = cosine.acos()*180/math.pi
        in_band = ((energy < 0)&(peri > self.safe)&((peri-self.goal).abs() <= self.tolerance)
                   &((apo-self.goal).abs() <= self.tolerance)&(plane <= 5))
        authority = torch.minimum(self.thrust/self.mass, 32*self.mu/self.goal.square())
        authority = torch.where(self.mass > self.dry_mass+1e-9, authority, 0)
        return dict(radius=rmag, basis=basis, local_v=local_v, energy=energy,
                    eccentricity=ecc, momentum=hmag, peri=peri, apo=apo,
                    plane=plane, cosine=cosine, in_band=in_band, authority=authority)

    def observation(self):
        """Float32 policy observation, memoized exactly on the current state.

        Each call returns the same tensor object while the state is unchanged;
        consumers copy it (History) and must not modify it in place.
        """
        return self._observation_memo.get(
            [getattr(self, name) for name in self.OBSERVATION_INPUTS], (self.version,),
            self._observation)

    def _observation(self):
        d = self.telemetry()
        local_normal = torch.bmm(d['basis'].transpose(1, 2), self.goal_normal[:, :, None]).squeeze(-1)
        local_direction = torch.bmm(d['basis'].transpose(1, 2), self.direction[:, :, None]).squeeze(-1)
        fuel = (self.mass-self.dry_mass)/(self.initial_mass-self.dry_mass)
        dv = self.isp*G0*(self.mass/self.dry_mass).log()
        tta = (d['radius']-d['peri']).clamp(min=0)/torch.maximum(
            -d['local_v'][:, 0], .02*self.V)
        scalars = [d['radius']/self.goal, d['local_v'][:, 0]/self.V,
                   d['local_v'][:, 1]/self.V, d['energy']/self.V.square(),
                   d['eccentricity'], d['peri']/self.goal,
                   torch.nan_to_num(d['apo']/self.goal, posinf=10),
                   *local_normal.unbind(-1), *local_direction.unbind(-1),
                   fuel, self.mass/self.initial_mass,
                   dv/self.V if self.version == 1 else (dv/self.V).log1p(),
                   (d['authority']*self.T/self.V).log1p(),
                   (self.isp*G0/self.V).log1p(), self.radius/self.goal,
                   self.safe/self.goal, (self.soi/self.goal).log1p(),
                   (tta/self.T).clamp(max=10), self.time/self.max_time,
                   (d['radius']-self.goal)/self.goal,
                   torch.where(d['energy'] < 0, 1., -1.),
                   d['momentum']/(self.goal*self.V), self.control_dt/self.T,
                   (self.mu/1e10).log10(), (self.radius/600000).log10(),
                   (self.thrust/self.mass*self.T/self.V).log1p(),
                   self.dry_mass/self.initial_mass, self.tolerance/self.goal]
        if self.version >= 2:
            scalars += [(d['peri']-self.goal)/self.tolerance,
                        torch.nan_to_num((d['apo']-self.goal)/self.tolerance, posinf=10),
                        d['plane']/5]
        return torch.stack(scalars, -1).clamp(-10, 10).float()

    def potential(self):
        d = self.telemetry()
        phi = -(4*(d['energy']/self.V.square()+.5).square()
                +8*(d['momentum']/(self.goal*self.V)-1).square()
                +4*(1-d['cosine'])+(d['radius']/self.goal-1).square()).clamp(max=30)
        if self.version >= 2:
            # Logarithmic in tolerance units: informative from hundreds of
            # tolerances (Sun starts) down to the band, bounded for escapes.
            peri = ((d['peri']-self.goal).abs()/self.tolerance).clamp(max=1e6)
            apo = torch.nan_to_num((d['apo']-self.goal).abs()/self.tolerance,
                                   posinf=1e6).clamp(max=1e6)
            phi = phi - .25*(peri.log1p()+apo.log1p())
        return phi

    def teacher(self):
        d = self.telemetry()
        vr, vt = d['local_v'][:, 0], d['local_v'][:, 1]
        tau = torch.maximum(3*self.control_dt, .055*self.T)
        ar = self.mu/d['radius'].square()-vt.square()/d['radius']-2*vr/tau-(d['radius']-self.goal)/tau.square()
        at = vr*vt/d['radius']-(vt-(self.mu/d['radius']).sqrt())/tau
        local_goal = torch.bmm(d['basis'].transpose(1, 2), self.goal_normal[:, :, None]).squeeze(-1)
        plane_position = (self.r*self.goal_normal).sum(-1)
        plane_velocity = (self.v*self.goal_normal).sum(-1)
        an = (-2*plane_velocity/tau-plane_position/tau.square()
              +self.mu/d['radius'].pow(3)*plane_position
              -ar*local_goal[:, 0]-at*local_goal[:, 1])/local_goal[:, 2].clamp(min=.2)
        action = torch.stack((ar, at, an), -1)/d['authority'].clamp(min=1e-30)[:, None]
        action /= action.norm(dim=-1, keepdim=True).clamp(min=1)
        return torch.where(d['in_band'][:, None], 0, action).float()

    def derivative(self, state, mass, force):
        return self._derivative(state, -self.mu[:, None], force/mass[:, None])

    @staticmethod
    def _derivative(state, negative_mu, acceleration):
        # Same operation order as derivative(); the two state-independent
        # factors are hoisted out of the four RK4 stages by step().
        r, v = state[:, :3], state[:, 3:]
        return torch.cat((v, negative_mu*r/r.norm(dim=-1)[:, None].pow(3)+acceleration), -1)

    @torch.no_grad()
    def step(self, action, active=None):
        if action.shape != (self.count, 3) or not bool(torch.isfinite(action).all()):
            raise ValueError("finite [worlds,3] RTN actions required")
        active = torch.ones(self.count, device=self.device, dtype=torch.bool) if active is None else active
        d = self.telemetry()
        action = action.to(self.dtype)
        magnitude = action.norm(dim=-1)
        throttle = magnitude.clamp(max=1)*d['authority']/(self.thrust/self.mass)
        target = torch.bmm(d['basis'], unit(action)[:, :, None]).squeeze(-1)
        remaining = torch.minimum(self.control_dt, (self.max_time-self.time).clamp(min=0))
        reason = torch.zeros(self.count, device=self.device, dtype=torch.long)
        initial_mass = self.mass.clone()
        # Quantities fixed for the whole control interval are evaluated once;
        # each is the identical expression the substep previously recomputed.
        commanded_thrust = throttle*self.thrust
        burn_rate = commanded_thrust/(self.isp*G0)
        fuel_floor = self.dry_mass+1e-9
        throttling = throttle > 0
        negative_mu = -self.mu[:, None]
        axes = torch.arange(3, device=self.device)
        for _ in range(8):
            continuing = active & (reason == 0) & (remaining > 1e-10)
            mdot = torch.where(self.mass > fuel_floor, burn_rate, 0)
            dt = torch.minimum(self.physics_dt, remaining)
            dt = torch.where(mdot > 0, torch.minimum(dt, (self.mass-self.dry_mass)/mdot.clamp(min=1e-30)), dt)
            dt = torch.where(continuing, dt, 0)
            dot = (self.direction*target).sum(-1).clamp(-1, 1)
            angle = dot.acos()
            turn = torch.minimum(angle, self.pointing_rate*dt)
            axis = torch.linalg.cross(self.direction, target)
            # Exact one-hot of the least-aligned axis; F.one_hot would add two
            # host-synchronizing range checks on CPU devices.
            fallback = (self.direction.abs().argmin(-1)[:, None] == axes).to(self.dtype)
            axis = torch.where((axis.norm(dim=-1) < 1e-10)[:, None],
                               torch.linalg.cross(self.direction, fallback), axis)
            rotated = (turn.cos()[:, None]*self.direction
                       +turn.sin()[:, None]*torch.linalg.cross(unit(axis), self.direction))
            desired = torch.where((angle < 1e-10)[:, None], target, unit(rotated))
            self.direction = torch.where((throttling&continuing)[:, None], desired, self.direction)
            force = torch.where((mdot > 0)[:, None], commanded_thrust[:, None]*self.direction, 0)
            mass = torch.maximum(self.dry_mass, self.mass-.5*mdot*dt)
            acceleration = force/mass[:, None]
            z = torch.cat((self.r, self.v), -1); step_dt = dt[:, None]; half = step_dt/2
            k1 = self._derivative(z, negative_mu, acceleration)
            k2 = self._derivative(z+half*k1, negative_mu, acceleration)
            k3 = self._derivative(z+half*k2, negative_mu, acceleration)
            k4 = self._derivative(z+step_dt*k3, negative_mu, acceleration)
            z += step_dt*(k1+2*k2+2*k3+k4)/6
            self.r, self.v = z[:, :3], z[:, 3:]
            self.mass = torch.maximum(self.dry_mass, self.mass-mdot*dt)
            self.time += dt; remaining = (remaining-dt).clamp(min=0)
            radius = self.r.norm(dim=-1)
            reason = torch.where(continuing & (radius <= self.safe), 2, reason)
            reason = torch.where(continuing & (radius >= self.soi), 3, reason)
        d = self.telemetry()
        self.hold = torch.where(active, torch.where(d['in_band'], self.hold+1, 0), self.hold)
        success = active & (reason == 0) & (self.hold >= 2)
        reason = torch.where(success, 1, reason)
        reason = torch.where(active & (reason == 0) & (self.mass <= self.dry_mass+1e-7) & ~d['in_band'], 4, reason)
        reason = torch.where(active & (reason == 0) & (d['energy'] >= 0)
                             & ((self.r*self.v).sum(-1) > 0) & (d['radius'] > 1.6*self.goal), 6, reason)
        reason = torch.where(active & (reason == 0) & (self.time >= self.max_time-1e-8), 5, reason)
        done = reason > 0
        following = self.potential()
        task = torch.where(success, 100., torch.where(done, -30., 0.))
        reward = (task-.01*self.control_dt/self.T-.1*(initial_mass-self.mass)/self.initial_mass
                  +.999*torch.where(done, 0, following)-self.previous_potential)
        reward = torch.where(active, reward, 0)
        self.previous_potential = torch.where(active, following, self.previous_potential)
        self.episode_return += reward
        return self.observation(), reward.float(), done, dict(success=success, reason=reason,
                    body=self.body.clone(), periapsis=d['peri']-self.radius,
                    apoapsis=d['apo']-self.radius, plane_error=d['plane'])

    def export_case(self, index):
        value = lambda name: float(getattr(self, name)[index])
        name = self.names[int(self.body[index])]
        config = CaptureConfig(body_name=name, mu=value('mu'), radius=value('radius'),
            atmosphere_depth=value('atmosphere_depth'), sphere_of_influence=value('soi'),
            target_altitude=value('goal')-value('radius'), orbit_tolerance=value('tolerance'),
            mass=value('initial_mass'), dry_mass=value('dry_mass'), thrust=value('thrust'),
            isp=value('isp'), control_dt=value('control_dt'), physics_dt=value('physics_dt'),
            max_time=value('max_time'), pointing_rate_deg_s=value('pointing_rate')*180/math.pi,
            randomization=0, target_altitude_randomization=0)
        d=self._export_telemetry if hasattr(self,'_export_telemetry') else self.telemetry()
        incoming_speed=math.sqrt(max(0.,2*(float(d['energy'][index])+config.mu/float(d['peri'][index]))))
        reference_dv=abs(incoming_speed-math.sqrt(config.mu/float(d['peri'][index])))
        available_dv=config.isp*G0*math.log(config.mass/config.dry_mass)
        return dict(config=asdict(config), r=self.r[index].cpu().tolist(), v=self.v[index].cpu().tolist(),
                    goal_normal=self.goal_normal[index].cpu().tolist(), direction=self.direction[index].cpu().tolist(),
                    task='heliocentric_orbit_adjustment' if name == 'Sun' else 'local_inbound_capture',
                    sun_outer_guard_is_numerical=name == 'Sun',
                    reference_budget=dict(available_delta_v=available_dv,
                        impulsive_periapsis_circularization_delta_v=reference_dv,
                        shortfall=available_dv<reference_dv,feasibility_proof=False))
