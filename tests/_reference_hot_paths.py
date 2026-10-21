"""Frozen pre-optimization hot paths (commit 1b04fee), for exactness tests.

Generated verbatim from the original sources; only method names were
changed where a subclass needed to call the original parent body. Tests
compare these against the optimized implementations bit for bit.
"""
from __future__ import annotations

import hashlib
import math
import time

import torch
from torch import nn

from orbiter.batched_capture import BatchedCapture, G0, unit
from orbiter.generalized_env import BatchedAscent, MAX_STAGES
from orbiter.train_architectures import AscentBenchmark, check_stop, make_env
from orbiter.train_generalized import (
    _PHASE_PROGRESS,
    _check_stop,
    clip_ppo_gradients,
    ppo_optimizer_configuration,
    squashed_normal_entropy,
)


class ReferenceCapture(BatchedCapture):
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

    def telemetry(self):
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

    def derivative(self, state, mass, force):
        r, v = state[:, :3], state[:, 3:]
        return torch.cat((v, -self.mu[:, None]*r/r.norm(dim=-1)[:, None].pow(3)
                          +force/mass[:, None]), -1)

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
        for _ in range(8):
            continuing = active & (reason == 0) & (remaining > 1e-10)
            mdot = torch.where(self.mass > self.dry_mass+1e-9, throttle*self.thrust/(self.isp*G0), 0)
            dt = torch.minimum(self.physics_dt, remaining)
            dt = torch.where(mdot > 0, torch.minimum(dt, (self.mass-self.dry_mass)/mdot.clamp(min=1e-30)), dt)
            dt = torch.where(continuing, dt, 0)
            dot = (self.direction*target).sum(-1).clamp(-1, 1)
            angle = dot.acos()
            turn = torch.minimum(angle, self.pointing_rate*dt)
            axis = torch.linalg.cross(self.direction, target)
            fallback = torch.nn.functional.one_hot(self.direction.abs().argmin(-1), 3).to(self.dtype)
            axis = torch.where((axis.norm(dim=-1) < 1e-10)[:, None],
                               torch.linalg.cross(self.direction, fallback), axis)
            rotated = (turn.cos()[:, None]*self.direction
                       +turn.sin()[:, None]*torch.linalg.cross(unit(axis), self.direction))
            desired = torch.where((angle < 1e-10)[:, None], target, unit(rotated))
            self.direction = torch.where(((throttle > 0)&continuing)[:, None], desired, self.direction)
            force = torch.where((mdot > 0)[:, None], (throttle*self.thrust)[:, None]*self.direction, 0)
            mass = torch.maximum(self.dry_mass, self.mass-.5*mdot*dt)
            z = torch.cat((self.r, self.v), -1); half = dt[:, None]/2
            k1 = self.derivative(z, mass, force)
            k2 = self.derivative(z+half*k1, mass, force)
            k3 = self.derivative(z+half*k2, mass, force)
            k4 = self.derivative(z+dt[:, None]*k3, mass, force)
            z += dt[:, None]*(k1+2*k2+2*k3+k4)/6
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


class ReferenceAscent(AscentBenchmark):
    @torch.no_grad()
    def reset(self, mask=None):
        if self.drag_alpha_area is None:
            self.drag_alpha_area, self.lift_area = (
                self.zeros(self.count),
                self.zeros(self.count),
            )
        if mask is None:
            mask = torch.ones(self.count, dtype=torch.bool, device=self.device)
        self._reference_base_reset(mask)
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
        return torch.cat((self._reference_base_observation(), aero), -1)

    def derivative(self, state, mass, pitch, thrust):
        # Preserve the parent's gravity, constant drag and thrust exactly; add
        # only the two force terms also present in the independent scalar model.
        derivative = self._reference_base_derivative(state, mass, pitch, thrust)
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

    @torch.no_grad()
    def _reference_base_reset(self, mask=None):
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
        self.episode_task_return[selected] = 0
        self.discounted_task_return[selected] = 0
        self.episode_discount[selected] = 1
        self.steps[selected] = 0
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

    def potential(self):
        d = self.telemetry()
        target_r = self.radius + self.target
        return -8 * (
            (d["energy"] / (-self.mu / (2 * target_r)) - 1).abs()
            + (d["momentum"] / (self.mu * target_r).sqrt() - 1).abs()
        )

    def mass(self):
        future = self.stage_ids > self.stage[:, None]
        current_dry = self.stages[self.ids, self.stage, 0]
        return (
            self.payload
            + current_dry
            + self.fuel
            + (self.stages[:, :, :2].sum(-1) * future).sum(-1)
        )

    def _reference_base_observation(self):
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

    def _reference_base_derivative(self, state, mass, pitch, thrust):
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
    def _reference_base_step(self, action):
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
        deadline = ~terminated & (self.time >= self.max_time - 1e-8)
        terminal_reward = torch.where(terminated, torch.where(success, 100, -30), 0)
        if self.deadline_mode == "terminal":
            terminated = terminated | deadline
            truncated = torch.zeros_like(deadline)
            terminal_reward = torch.where(
                deadline, self.timeout_penalty, terminal_reward.to(self.dtype)
            )
        else:
            truncated = deadline
        phi = self.potential()
        shaping = self.gamma * torch.where(terminated, 0, phi) - self.previous_potential
        # Keep the historical operation order for exact v3/v4 default rewards.
        time_reward = -0.003 * elapsed
        reward = shaping + terminal_reward - 0.003 * elapsed
        task_reward = terminal_reward + time_reward
        self.previous_potential = phi
        self.episode_return += reward
        self.episode_task_return += task_reward
        # Gamma follows control decisions, as in the shaping term and PPO;
        # actual elapsed seconds determine only the existing time cost.
        self.discounted_task_return += self.episode_discount * task_reward
        self.episode_discount *= self.gamma
        self.steps += 1
        # Reason codes: success, impact, escape, fuel, time. Success has priority.
        reason = torch.where(
            success,
            1,
            torch.where(
                impact,
                2,
                torch.where(
                    escape, 3, torch.where(fuel_empty, 4, torch.where(deadline, 5, 0))
                ),
            ),
        )
        info = dict(
            success=success,
            reason=reason,
            episode_return=self.episode_return.clone(),
            terminal_reward=terminal_reward.to(self.dtype),
            time_reward=time_reward,
            task_reward=task_reward,
            shaping_reward=shaping,
            elapsed=elapsed,
            discounted_task_return=self.discounted_task_return.clone(),
            episode_task_return=self.episode_task_return.clone(),
            steps=self.steps.clone(),
            time=self.time.clone(),
            apoapsis=d["apoapsis"],
            periapsis=d["periapsis"],
        )
        return self.observation(), reward.float(), terminated, truncated, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self._reference_base_step(action)
        return observation, reward.float(), terminated | truncated, info


class ReferenceHistory:
    def __init__(self, count, length, observations, actions, device):
        self.observations, self.actions = observations, actions
        self.data = torch.zeros(count, length, observations + actions + 1, device=device)

    @torch.no_grad()
    def append(self, observation, previous_action=None, reset=None):
        if observation.shape != (self.data.shape[0], self.observations):
            raise ValueError("history observation shape mismatch")
        if reset is not None:
            self.data[reset] = 0
        self.data[:, :-1] = self.data[:, 1:].clone()
        self.data[:, -1] = 0
        self.data[:, -1, :self.observations] = observation
        if previous_action is not None:
            self.data[:, -1, self.observations:-1] = previous_action
            if reset is not None:
                self.data[reset, -1, self.observations:-1] = 0
        self.data[:, -1, -1] = 1
        return self.data


def reference_ppo_update(model, data, actor_optimizer, critic_optimizer, args, *, anchor=None,
               anchor_coef=0., train_actor=True):
    histories,latents,old_logp,advantages,returns=data
    actor_steps=critic_steps=0;last_kl=0.;stopped=False;anchor_loss=None
    for _ in range(args.epochs if train_actor else 0):
        for indices in torch.randperm(len(histories),device=args.device).split(args.batch_size):
            check_stop()
            x=histories[indices];latent=latents[indices];target=returns[indices]
            dist=model.distribution(x);logp=model.log_probability(dist,latent)
            log_ratio=logp-old_logp[indices]
            kl=(log_ratio.exp()-1-log_ratio).mean()
            last_kl=float(kl.detach())
            if last_kl>args.target_kl:stopped=True;break
            ratio=(logp-old_logp[indices]).exp();adv=advantages[indices]
            policy_loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
            sample=dist.rsample();entropy=-model.log_probability(dist,sample).mean()
            actor_loss=policy_loss-args.entropy*entropy
            if anchor is not None and anchor_coef>0:
                bank_x,bank_y=anchor
                chosen=torch.randint(len(bank_x),(len(indices),),device=args.device)
                anchor_loss=(model.actor(bank_x[chosen]).tanh()-bank_y[chosen]).square().mean()
                actor_loss=actor_loss+anchor_coef*anchor_loss
            actor_optimizer.zero_grad();actor_loss.backward()
            torch.nn.utils.clip_grad_norm_([*model.actor.parameters(),model.log_std],.5)
            actor_optimizer.step();actor_steps+=1
            value_loss=(model.critic(x).squeeze(-1)-target).square().mean()
            critic_optimizer.zero_grad();value_loss.backward();torch.nn.utils.clip_grad_norm_(model.critic.parameters(),.5)
            critic_optimizer.step();critic_steps+=1
        if stopped:break
    for _ in range(args.critic_extra if train_actor else args.critic_extra+args.epochs*math.ceil(len(histories)/args.batch_size)):
        check_stop()
        indices=torch.randint(len(histories),(min(args.batch_size,len(histories)),),device=args.device)
        loss=(model.critic(histories[indices]).squeeze(-1)-returns[indices]).square().mean()
        critic_optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.critic.parameters(),.5)
        critic_optimizer.step();critic_steps+=1
    return dict(actor_steps=actor_steps,critic_steps=critic_steps,last_kl=last_kl,kl_early_stop=stopped,
                anchor_coef=anchor_coef,anchor_mse=None if anchor_loss is None else float(anchor_loss.detach()),
                critic_warmup=not train_actor)


def reference_rollout(model, env, history, args):
    """Verbatim rollout/GAE body of the original train() loop."""
    xs=[];zs=[];lps=[];values=[];rewards=[];dones=[]
    rollout_successes=rollout_episodes=0
    with torch.no_grad():
        for _ in range(args.horizon):
            check_stop();xs.append(history.data.clone())
            action,z,lp=model.sample(history.data);value=model.critic(history.data).squeeze(-1)
            _,reward,done,info=env.step(action)
            zs.append(z);lps.append(lp);values.append(value);rewards.append(reward);dones.append(done)
            rollout_successes+=int(info['success'].sum());rollout_episodes+=int(done.sum())
            env.reset(done);history.append(env.observation(),action,done)
        next_value=model.critic(history.data).squeeze(-1)
        advantage=torch.zeros(args.worlds,device=args.device);advantages=[];returns=[]
        for t in reversed(range(args.horizon)):
            alive=(~dones[t]).float();delta=rewards[t]+.999*next_value*alive-values[t]
            advantage=delta+.999*.95*alive*advantage
            advantages.append(advantage);returns.append(advantage+values[t]);next_value=values[t]
        advantages=torch.stack(advantages[::-1]).flatten()
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        targets=torch.stack(returns[::-1]).flatten()
        latent_all=torch.cat(zs)
        saturation=float((latent_all.tanh().abs()>.99).float().mean())
        explained=1-float((targets-torch.cat(values)).var()/(targets.var()+1e-8))
    return ((torch.cat(xs),latent_all,torch.cat(lps),advantages,targets),
            dict(successes=rollout_successes,episodes=rollout_episodes,
                 latent_saturation=saturation,explained_variance=explained))


def reference_fit_teacher(model, task, parent, args, log):
    """Behavior-clone the teacher; return the label bank for the PPO anchor.

    Labels are fitted in tanh space through the policy's inverse action
    mapping. With ``scaled_tanh`` the targets lie strictly inside (-1, 1), so
    full-throttle/vertical labels no longer push latent means without bound.
    """
    env=make_env(task,args.label_worlds,args.device,args.seed+100000,
                 orbital_version=args.orbital_version)
    history=ReferenceHistory(env.count,model.spec['context'],env.observations,env.actions,args.device)
    history.append(env.observation())
    observations,labels=[],[]
    started=time.monotonic()
    for _ in range(args.label_steps):
        check_stop()
        with torch.no_grad():
            action=parent(env.observation()) if parent is not None else env.teacher()
            observations.append(history.data.clone());labels.append(action)
            obs,_,done,_=env.step(action)
            env.reset(done);obs=env.observation()
            history.append(obs,action,done)
    x=torch.cat(observations);y=model.squashed_target(torch.cat(labels))
    fingerprint=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
    observations_sha=hashlib.sha256(x[:,-1,:env.observations].contiguous().cpu().numpy().tobytes()).hexdigest()
    labels_sha=hashlib.sha256(y.contiguous().cpu().numpy().tobytes()).hexdigest()
    log('teacher_bank',samples=len(x),history_sha256=fingerprint,
        current_observation_sha256=observations_sha,labels_sha256=labels_sha,
        elapsed_s=time.monotonic()-started)
    optimizer=torch.optim.Adam(model.actor.parameters(),lr=3e-4)
    rng=torch.Generator(device=args.device).manual_seed(args.seed+200000)
    for update in range(args.bc_updates):
        check_stop()
        ids=torch.randint(len(x),(min(args.batch_size,len(x)),),generator=rng,device=args.device)
        loss=(model.actor(x[ids]).tanh()-y[ids]).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.actor.parameters(),.5);optimizer.step()
        if (update+1)%100==0:log('bc',update=update+1,action_mse=float(loss.detach()))
    with torch.no_grad():
        latent=torch.cat([model.actor(chunk) for chunk in x.split(8192)])
        log('bc_latent',mean_abs=float(latent.abs().mean()),max_abs=float(latent.abs().max()),
            saturated_fraction=float((latent.tanh().abs()>.99).float().mean()))
    del observations,labels,env
    return x,y


def reference_ppo_train(
    model,
    env,
    *,
    updates,
    horizon,
    batch_size,
    learning_rate,
    log,
    selected_callback,
    target_kl=None,
    validation_interval=5,
    ppo_optimizer="joint",
    actor_learning_rate=None,
    critic_learning_rate=None,
    actor_max_grad_norm=None,
    critic_max_grad_norm=None,
    entropy_coefficient=0.0,
    critic_extra_updates=0,
):
    """On-device PPO with proper terminal and time-limit bootstrap separation."""
    configuration = ppo_optimizer_configuration(
        optimizer=ppo_optimizer, learning_rate=learning_rate,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        actor_max_grad_norm=actor_max_grad_norm,
        critic_max_grad_norm=critic_max_grad_norm,
        entropy_coefficient=entropy_coefficient,
        critic_extra_updates=critic_extra_updates,
    )
    actor_parameters = [*model.actor.parameters(), model.log_std]
    critic_parameters = list(model.critic.parameters())
    optimizers = (
        [torch.optim.Adam(model.parameters(), lr=learning_rate)]
        if ppo_optimizer == "joint" else [
            torch.optim.Adam(actor_parameters, lr=configuration["actor_learning_rate"]),
            torch.optim.Adam(critic_parameters, lr=configuration["critic_learning_rate"]),
        ]
    )
    observation = env.observation()
    transitions = 0
    optimizer_steps = 0
    critic_optimizer_steps = 0
    for update in range(updates):
        _PHASE_PROGRESS.update(
            phase="ppo", completed_updates=update, ppo_transitions=transitions
        )
        _check_stop()
        buffers = {
            key: []
            for key in (
                "obs",
                "latent",
                "logp",
                "values",
                "rewards",
                "done",
                "next_values",
            )
        }
        episode_count = torch.zeros((), dtype=torch.long, device=env.device)
        successes = torch.zeros_like(episode_count)
        start = time.monotonic()
        for _ in range(horizon):
            with torch.no_grad():
                action, latent, logp, value = model.sample(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_value = model.critic(next_observation).squeeze(-1)
                next_value = torch.where(terminated, 0, next_value)
                for key, val in (
                    ("obs", observation),
                    ("latent", latent),
                    ("logp", logp),
                    ("values", value),
                    ("rewards", reward),
                    ("done", terminated | truncated),
                    ("next_values", next_value),
                ):
                    buffers[key].append(val)
                done = terminated | truncated
                episode_count += done.sum()
                successes += (done & info["success"]).sum()
                observation = env.reset(done) if bool(done.any()) else next_observation
        b = {key: torch.stack(val) for key, val in buffers.items()}
        advantage = torch.zeros_like(b["rewards"])
        gae = torch.zeros(env.count, device=env.device)
        for step in reversed(range(horizon)):
            delta = (
                b["rewards"][step]
                + env.gamma * b["next_values"][step]
                - b["values"][step]
            )
            gae = delta + env.gamma * 0.95 * (~b["done"][step]).float() * gae
            advantage[step] = gae
        returns = advantage + b["values"]
        td_error = b["rewards"] + env.gamma * b["next_values"] - b["values"]
        return_variance = returns.var(unbiased=False)
        explained_variance = (
            1 - (returns - b["values"]).var(unbiased=False) / return_variance
            if bool(return_variance > 0)
            else torch.full((), float("nan"), device=env.device)
        )
        action_saturation = (b["latent"].tanh().abs() >= 0.98).float().mean()
        flat = {key: val.flatten(0, 1) for key, val in b.items()}
        advantage = advantage.flatten()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        returns = returns.flatten()
        total_loss = torch.zeros((), device=env.device)
        total_kl = torch.zeros_like(total_loss)
        total_clipped = torch.zeros_like(total_loss)
        maximum_kl = torch.zeros_like(total_loss)
        total_actor_loss = torch.zeros_like(total_loss)
        total_critic_loss = torch.zeros_like(total_loss)
        total_grad_norm = torch.zeros_like(total_loss)
        total_actor_grad_norm = torch.zeros_like(total_loss)
        total_critic_grad_norm = torch.zeros_like(total_loss)
        total_entropy = torch.zeros_like(total_loss)
        checked_batches = 0
        loss_count = 0
        early_stopped = False
        for epoch in range(4):
            permutation = torch.randperm(len(returns), device=env.device)
            for selected in permutation.split(batch_size):
                distribution = model.distribution(flat["obs"][selected])
                action = flat["latent"][selected].tanh()
                logp = (
                    distribution.log_prob(flat["latent"][selected])
                    - (1 - action.square() + 1e-6).log()
                ).sum(-1)
                log_ratio = logp - flat["logp"][selected]
                ratio = log_ratio.exp()
                approximate_kl = ((ratio - 1) - log_ratio).detach().mean()
                total_kl += approximate_kl
                maximum_kl = torch.maximum(maximum_kl, approximate_kl)
                total_clipped += ((ratio - 1).abs() > 0.2).float().mean().detach()
                checked_batches += 1
                # Check before applying another optimizer step. Waiting for a
                # complete epoch permits 64 updates at the production batch size
                # and allowed the higher-rate pilot to overshoot substantially.
                if target_kl is not None:
                    measured_kl = float(approximate_kl)
                    if not math.isfinite(measured_kl):
                        raise FloatingPointError(
                            "PPO update produced a nonfinite KL divergence"
                        )
                    if measured_kl > target_kl:
                        early_stopped = True
                        break
                actor_loss = -torch.minimum(
                    ratio * advantage[selected],
                    ratio.clamp(0.8, 1.2) * advantage[selected],
                ).mean()
                value = model.critic(flat["obs"][selected]).squeeze(-1)
                value_loss = nn.functional.mse_loss(value, returns[selected])
                loss = actor_loss + 0.5 * value_loss
                entropy = None
                if entropy_coefficient:
                    entropy = squashed_normal_entropy(distribution)
                    loss = loss - entropy_coefficient * entropy
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm, actor_grad_norm, critic_grad_norm = clip_ppo_gradients(
                    actor_parameters, critic_parameters, optimizer=ppo_optimizer,
                    actor_max_grad_norm=configuration["gradient_clipping"].get(
                        "actor_max_norm", 0.5
                    ),
                    critic_max_grad_norm=configuration["gradient_clipping"].get(
                        "critic_max_norm", 0.5
                    ),
                    joint_parameters=model.parameters(),
                )
                for optimizer in optimizers:
                    optimizer.step()
                total_loss += loss.detach()
                total_actor_loss += actor_loss.detach()
                total_critic_loss += value_loss.detach()
                total_grad_norm += grad_norm.detach()
                total_actor_grad_norm += actor_grad_norm
                total_critic_grad_norm += critic_grad_norm
                if entropy is not None:
                    total_entropy += entropy.detach()
                loss_count += 1
                optimizer_steps += 1
                critic_optimizer_steps += 1
            if not math.isfinite(float(maximum_kl)):
                raise FloatingPointError(
                    "PPO update produced a nonfinite KL divergence"
                )
            if early_stopped:
                break
        if not loss_count:
            raise FloatingPointError(
                "PPO rejected the unchanged rollout policy before its first step"
            )
        extra_critic_loss = torch.zeros((), device=env.device)
        extra_critic_grad_norm = torch.zeros_like(extra_critic_loss)
        for _ in range(critic_extra_updates):
            # Fixed, separately budgeted fits on the frozen rollout targets.
            # The actor KL budget cannot stop these critic-only updates.
            selected = torch.randint(
                len(returns), (min(batch_size, len(returns)),), device=env.device
            )
            value = model.critic(flat["obs"][selected]).squeeze(-1)
            value_loss = nn.functional.mse_loss(value, returns[selected])
            optimizers[1].zero_grad(set_to_none=True)
            (0.5 * value_loss).backward()
            critic_norm = nn.utils.clip_grad_norm_(
                critic_parameters,
                configuration["gradient_clipping"]["critic_max_norm"],
            )
            optimizers[1].step()
            extra_critic_loss += value_loss.detach()
            extra_critic_grad_norm += critic_norm.detach()
            critic_optimizer_steps += 1
        transitions += horizon * env.count
        _PHASE_PROGRESS.update(
            phase="ppo",
            completed_updates=update + 1,
            ppo_transitions=transitions,
            cumulative_optimizer_steps=optimizer_steps,
            cumulative_critic_optimizer_steps=critic_optimizer_steps,
        )
        _check_stop()
        log(
            "ppo",
            update=update + 1,
            transitions=transitions,
            completed_episodes=int(episode_count),
            completed_successes=int(successes),
            loss=float(total_loss / loss_count),
            actor_loss=float(total_actor_loss / loss_count),
            critic_loss=float(total_critic_loss / loss_count),
            gradient_norm_before_clip=float(total_grad_norm / loss_count),
            actor_gradient_norm_before_clip=float(total_actor_grad_norm / loss_count),
            critic_gradient_norm_before_clip=float(total_critic_grad_norm / loss_count),
            explained_variance=float(explained_variance),
            mean_td_error=float(td_error.mean()),
            mean_absolute_td_error=float(td_error.abs().mean()),
            action_tanh_saturation_fraction=float(action_saturation),
            cumulative_optimizer_steps=optimizer_steps,
            cumulative_actor_optimizer_steps=optimizer_steps,
            cumulative_critic_optimizer_steps=critic_optimizer_steps,
            critic_extra_updates=critic_extra_updates,
            extra_critic_loss=(
                float(extra_critic_loss / critic_extra_updates)
                if critic_extra_updates else None
            ),
            extra_critic_gradient_norm_before_clip=(
                float(extra_critic_grad_norm / critic_extra_updates)
                if critic_extra_updates else None
            ),
            approximate_kl=float(total_kl / checked_batches),
            maximum_minibatch_kl=float(maximum_kl),
            clip_fraction=float(total_clipped / checked_batches),
            checked_minibatches=checked_batches,
            optimization_minibatches=loss_count,
            optimization_epochs=epoch + 1,
            kl_early_stop=early_stopped,
            kl_check="before_minibatch_optimizer_step",
            action_std=model.log_std.detach().exp().cpu().tolist(),
            ppo_optimizer=ppo_optimizer,
            entropy_coefficient=entropy_coefficient,
            mean_squashed_entropy=(
                float(total_entropy / loss_count) if entropy_coefficient else None
            ),
            elapsed_s=time.monotonic() - start,
        )
        if (update + 1) % validation_interval == 0 or update + 1 == updates:
            selected_callback(transitions)
    return transitions

