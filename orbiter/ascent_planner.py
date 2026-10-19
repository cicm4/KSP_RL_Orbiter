"""Original shooting-based ascent teacher for the planar rocket surrogate.

The cross-entropy method searches a small feedback guidance family, using the
full known stage stack. Every craft keeps its incumbent, including the existing
heuristic. A failed search says nothing about physical infeasibility. This is
an experimental simulator teacher, not KSP guidance or KSP validation.

Run ``python -m orbiter.ascent_planner --help`` for a paired frozen-bank
benchmark. Successful witnesses are replayed through the independent scalar
simulator, then coasted unpowered for a complete orbital period.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .env import EnvConfig, G0, HeadlessOrbiterEnv, orbital_elements
from .generalized_aero_env import BatchedAeroAscent


SCHEMA = "stage_aware_shooting_teacher_v1"
PARAMETER_NAMES = (
    "turn_start_altitude_m", "turn_span_m", "turn_power", "minimum_turn_pitch_deg",
    "apoapsis_offset_m", "circularization_lead", "altitude_gain",
    "radial_velocity_gain", "maximum_acceleration_m_s2", "q_target_pa",
)
LOWER = (300., 18000., .55, 0., -4000., .25, .025, .1, 18., 20000.)
UPPER = (4000., 70000., 1.8, 18., 10000., 1.1, .2, .6, 65., 90000.)
DEFAULT = (1500., 39000., 1., 8., 1500., .6, .12, .22, 45., 50000.)
REASONS = {1: "orbit_achieved", 2: "impact", 3: "escape", 4: "out_of_fuel", 5: "time_limit"}


@dataclass(frozen=True)
class SearchBudget:
    population: int = 32
    iterations: int = 4
    elite_fraction: float = .2
    seed: int = 0
    # None preserves each frozen config's integrator step. A changed search
    # step is explicitly recorded; scalar qualification uses the original.
    physics_dt: float | None = None
    max_dynamic_pressure: float | None = None

    def __post_init__(self):
        for name in ("population", "iterations", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.population < 3 or self.iterations < 1 or self.seed < 0:
            raise ValueError("population >=3, iterations >=1 and seed >=0 required")
        if isinstance(self.elite_fraction, bool) or not math.isfinite(self.elite_fraction) or not 0 < self.elite_fraction <= 1:
            raise ValueError("elite_fraction must be in (0,1]")
        if self.physics_dt is not None and (
            isinstance(self.physics_dt, bool) or not math.isfinite(self.physics_dt) or not 0 < self.physics_dt <= 2
        ):
            raise ValueError("physics_dt must be in (0,2]")
        if self.max_dynamic_pressure is not None and (
            isinstance(self.max_dynamic_pressure, bool) or not math.isfinite(self.max_dynamic_pressure) or self.max_dynamic_pressure <= 0
        ):
            raise ValueError("max_dynamic_pressure must be positive and finite")


def config_fingerprint(config: EnvConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _validate_configs(configs):
    configs = tuple(configs)
    if not configs:
        raise ValueError("at least one frozen craft is required")
    # load_configs validates all remaining physical and shared-global rules.
    for config in configs:
        if not isinstance(config, EnvConfig) or config.randomization or config.residual or config.curriculum:
            raise ValueError("provide explicit EnvConfigs without randomization/residual/curriculum")
    return configs


@torch.no_grad()
def remaining_burn_time(env, delta_v):
    """Rocket-equation burn time, crossing actual remaining stage boundaries.

    Vacuum thrust/Isp are used for this estimate; search rollouts still use the
    complete atmosphere/drag/lift/actuator model. Unavailable delta-v is exposed
    separately, rather than silently treating an exhausted stack as sufficient.
    """
    remaining = delta_v.clamp(min=0).to(env.dtype)
    elapsed = torch.zeros_like(remaining)
    mass = env.mass()
    for index in range(env.stages.shape[1]):
        dry, fuel, thrust, isp, _ = env.stages[:, index].unbind(-1)
        present = (index >= env.stage) & (index < env.stage_count)
        fuel = torch.where(index == env.stage, env.fuel, fuel)
        fuel = torch.where(present, fuel, 0)
        exhaust = (isp * G0).clamp(min=1e-12)
        capacity = exhaust * (mass / (mass - fuel).clamp(min=1e-9)).log()
        used_dv = torch.minimum(remaining, capacity.clamp(min=0))
        used_fuel = mass * (-torch.expm1(-used_dv / exhaust))
        elapsed += torch.where(present, used_fuel * exhaust / thrust.clamp(min=1e-12), 0)
        remaining -= used_dv
        # Future burn starts after the current stage has been discarded. Only
        # matters for worlds that still need delta-v from the following stage.
        mass = torch.where(present, mass - fuel - dry, mass).clamp(min=1e-9)
    return elapsed, remaining.clamp(min=0)


@torch.no_grad()
def guidance_action(env, parameters):
    """Stage-aware feedback profile; parameters have [worlds,10] shape.

    The ascent follows a searched pitch/altitude curve. A finite-burn lead
    estimate uses every remaining stage before coasting to circularization.
    Orbital feedback then damps radial motion while supplying horizontal speed.
    """
    if parameters.shape != (env.count, len(PARAMETER_NAMES)) or not bool(torch.isfinite(parameters).all()):
        raise ValueError("parameters must be finite [worlds,10] values")
    p = parameters.to(device=env.device, dtype=env.dtype)
    start, span, power, floor, offset, lead, kp, kd, accel_limit, q_target = p.unbind(-1)
    d = env.telemetry()
    alt, vr, vt, apo = (d[name] for name in ("altitude", "vertical_speed", "horizontal_speed", "apoapsis"))
    target = env.target + offset
    acceleration = env.available_thrust() / env.mass()
    circular = (env.mu / (env.radius + alt).clamp(min=1)).sqrt()
    deficit = (circular - vt).clamp(min=0)
    burn_time, _ = remaining_burn_time(env, torch.hypot(deficit, vr))
    # Ballistic radial deceleration includes centrifugal support. The simple
    # time-to-apo estimate is only used while still appreciably suborbital.
    radial_gravity = (env.mu / (env.radius + alt).square() - vt.square() / (env.radius + alt)).clamp(min=.2)
    time_to_apo = vr.clamp(min=0) / radial_gravity
    turn = ((alt - start) / span).clamp(0, 1).pow(power)
    pitch = 90 - (90 - floor) * turn
    high_pitch = (kp * (target - alt) / vt.clamp(min=100) - kd * vr).clamp(-10, 30)
    high_pitch = torch.where(apo < target - 1000, high_pitch.clamp(min=5), high_pitch)
    pitch = torch.where(alt >= start + span, high_pitch, pitch)
    throttle = (accel_limit / acceleration.clamp(min=.01)).clamp(0, 1)
    # Throttling at high q is a soft guidance preference, not structural proof.
    throttle = torch.minimum(throttle, (q_target / d["dynamic_pressure"].clamp(min=1)).clamp(.25, 1))
    reached_apo = apo >= target - 1000
    coast = reached_apo & (alt < target - 3000) & (vr > 20) & (time_to_apo > lead * burn_time + env.control_dt)
    circularize = reached_apo & ~coast
    throttle = torch.where(coast, 0, throttle)
    # The response time scales with actual acceleration, keeping low-thrust
    # upper stages burning rather than relying on a fixed velocity threshold.
    circular_throttle = (deficit / (acceleration * (2 * env.control_dt)).clamp(min=1)).clamp(0, 1)
    circular_pitch = (-kd * vr + kp * (target - alt) / vt.clamp(min=100)).clamp(-10, 20)
    pitch = torch.where(circularize, circular_pitch, pitch)
    throttle = torch.where(circularize, circular_throttle, throttle)
    pitch = torch.where(alt < start, 90, pitch)
    # The launch itself must have enough thrust to leave the pad.
    throttle = torch.where(alt < start, 1, throttle)
    return torch.stack((2 * throttle - 1, (pitch + 10) / 50 - 1), -1).clamp(-1, 1).float()


def _terminal_cost(env, success, max_q, budget):
    d = env.telemetry()
    tolerance = max(env.orbit_tolerance, 1.)
    apo_error = (d["apoapsis"] - env.target).abs().nan_to_num(posinf=1e9) / tolerance
    peri_error = (d["periapsis"] - env.target).abs().nan_to_num(posinf=1e9) / tolerance
    # Every feasible witness beats every non-witness. Search losses and RL
    # rewards are deliberately independent: no discounted deadline bootstrap.
    feasible = success.clone()
    pressure_penalty = torch.zeros_like(max_q)
    if budget.max_dynamic_pressure is not None:
        feasible &= max_q <= budget.max_dynamic_pressure
        pressure_penalty = (max_q / budget.max_dynamic_pressure - 1).clamp(min=0)
    distance = (apo_error + peri_error).clamp(max=1000) + 10 * pressure_penalty
    cost = torch.where(feasible, env.time / env.max_time, 100 + distance)
    return cost, feasible


@torch.no_grad()
def _rollout(configs, parameters, heuristic, *, device, budget, record_actions=False):
    env = BatchedAeroAscent(len(configs), device=device, seed=0)
    env.load_configs(configs)
    parameters = torch.as_tensor(parameters, device=device, dtype=env.dtype)
    heuristic = torch.as_tensor(heuristic, device=device, dtype=torch.bool)
    done = torch.zeros(env.count, device=device, dtype=torch.bool)
    success = done.clone()
    reason = torch.zeros(env.count, device=device, dtype=torch.long)
    max_q = torch.zeros(env.count, device=device, dtype=env.dtype)
    final_time, costs = env.time.clone(), torch.full_like(env.time, torch.inf)
    final_state = env.state.clone()
    final_stage, final_fuel = env.stage.clone(), env.fuel.clone()
    final_apo, final_peri = env.time.clone(), env.time.clone()
    feasible = done.clone()
    actions = [[] for _ in configs] if record_actions else None
    for _ in range(math.ceil(env.max_time / env.control_dt) + 1):
        action = guidance_action(env, parameters)
        if bool(heuristic.any()):
            action = torch.where(heuristic[:, None], env.teacher(), action)
        if actions is not None:
            for index in torch.nonzero(~done).flatten().cpu().tolist():
                actions[index].append(action[index].cpu().tolist())
        max_q = torch.maximum(max_q, torch.where(done, 0, env.telemetry()["dynamic_pressure"]))
        _, _, terminated, truncated, info = env.step(action)
        max_q = torch.maximum(max_q, torch.where(done, 0, env.telemetry()["dynamic_pressure"]))
        finished = (terminated | truncated) & ~done
        if bool(finished.any()):
            cost, valid = _terminal_cost(env, info["success"], max_q, budget)
            costs[finished], feasible[finished] = cost[finished], valid[finished]
            success[finished], reason[finished] = info["success"][finished], info["reason"][finished]
            final_time[finished], final_state[finished] = env.time[finished], env.state[finished]
            final_stage[finished], final_fuel[finished] = env.stage[finished], env.fuel[finished]
            final_apo[finished], final_peri[finished] = info["apoapsis"][finished], info["periapsis"][finished]
        done |= finished
        if bool(done.all()):
            break
        env.time[done] = env.max_time
    if not bool(done.all()):
        raise RuntimeError("search rollout did not finish within its finite horizon")
    return dict(cost=costs, success=success, feasible=feasible, reason=reason,
                time=final_time, state=final_state, stage=final_stage, fuel=final_fuel,
                apoapsis=final_apo, periapsis=final_peri, max_q_control_samples=max_q,
                actions=actions)


def _row(result, index):
    row = {}
    for key, value in result.items():
        if key == "actions":
            if value is not None:
                row[key] = value[index]
        else:
            value = value[index].detach().cpu()
            row[key] = value.tolist() if value.ndim else value.item()
    row["termination_reason"] = REASONS.get(row["reason"], "unknown")
    return row


def _coast_scalar(env, *, max_step=2.):
    """Independent scalar RK4, zero thrust, for one osculating orbit period."""
    c = env.config
    state = np.asarray([env.x, env.y, env.vx, env.vy], dtype=np.float64)
    apo, peri, energy, momentum = orbital_elements(*state, c.mu, c.radius)
    if energy >= 0 or not math.isfinite(apo):
        return dict(success=False, reason="not_bound", duration_s=0.)
    period = 2 * math.pi * (-c.mu / (2 * energy)) ** 1.5 / math.sqrt(c.mu)
    minimum = max(c.atmosphere_depth + 1000, c.target_altitude - c.orbit_tolerance)
    maximum = c.target_altitude + c.orbit_tolerance
    min_peri, max_apo = peri, apo
    start = state.copy()
    elapsed = 0.
    while elapsed < period - 1e-9:
        dt = min(max_step, period - elapsed)
        f = lambda z: env._derivative(z, env.mass, env.pitch, 0.)
        k1 = f(state)
        k2 = f(state + dt * k1 / 2)
        k3 = f(state + dt * k2 / 2)
        k4 = f(state + dt * k3)
        state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        elapsed += dt
        a, p, e, _ = orbital_elements(*state, c.mu, c.radius)
        min_peri, max_apo = min(min_peri, p), max(max_apo, a)
        if e >= 0 or p < minimum - .01 or a > maximum + .01:
            return dict(success=False, reason="orbit_left_goal_band", duration_s=elapsed,
                        requested_period_s=period, min_periapsis_m=min_peri, max_apoapsis_m=max_apo)
    return dict(success=True, duration_s=elapsed, requested_period_s=period,
                min_periapsis_m=min_peri, max_apoapsis_m=max_apo,
                return_position_error_m=float(np.linalg.norm(state[:2] - start[:2])),
                relative_energy_drift=abs((np.dot(state[2:], state[2:]) / 2 - c.mu / np.linalg.norm(state[:2]) - energy) / energy),
                initial_angular_momentum=momentum, simulator_only=True)


def qualify_witness(config, actions, *, expected=None, max_dynamic_pressure=None):
    """Replay fixed actions using original-step scalar physics, then coast.

    A search using coarser integration cannot certify itself. This replay never
    changes the frozen craft, timestep, target, or deadline to rescue a plan.
    """
    _validate_configs([config])
    if max_dynamic_pressure is not None and (
        isinstance(max_dynamic_pressure, bool) or not math.isfinite(max_dynamic_pressure) or max_dynamic_pressure <= 0
    ):
        raise ValueError("max_dynamic_pressure must be positive and finite")
    env = HeadlessOrbiterEnv(config)
    info = env.info("")
    consumed = 0
    for action in actions:
        _, _, terminated, truncated, info = env.step(action)
        consumed += 1
        if terminated or truncated:
            break
    orbital_success = bool(info["success"])
    pressure_ok = max_dynamic_pressure is None or env.max_q <= max_dynamic_pressure
    coast = _coast_scalar(env) if orbital_success and pressure_ok else None
    state = [env.x, env.y, env.vx, env.vy]
    return dict(
        qualified=bool(orbital_success and pressure_ok and coast and coast["success"]),
        scalar_success=orbital_success, pressure_constraint_satisfied=pressure_ok,
        scalar_termination_reason=info["termination_reason"] or "actions_exhausted",
        actions_consumed=consumed, total_actions=len(actions), time_s=env.t,
        stage=env.stage, max_q_pa=env.max_q, apoapsis_m=info["apoapsis"],
        periapsis_m=info["periapsis"], state=state,
        search_state_max_abs_difference=(float(np.max(np.abs(np.asarray(state) - expected["state"]))) if expected is not None else None),
        coast=coast, config_sha256=config_fingerprint(config),
        proof_scope="independent scalar planar simulator; not actual KSP",
    )


@torch.no_grad()
def plan_ascent(configs, *, budget=None, device="cpu", qualify=True):
    """Optimize each known craft independently, with a monotone incumbent.

    Population includes the previous best and exact heuristic every iteration.
    ``success`` below always means a rollout success; ``qualified`` additionally
    requires independent scalar replay and a complete unpowered orbit.
    """
    configs = _validate_configs(configs)
    budget = budget or SearchBudget()
    count, population = len(configs), budget.population
    start_time = time.monotonic()
    search_configs = tuple(replace(c, physics_dt=budget.physics_dt) if budget.physics_dt is not None else c for c in configs)
    # One validation call before allocating the larger population.
    check = BatchedAeroAscent(count, device=device, seed=budget.seed)
    check.load_configs(search_configs)
    del check
    lower = torch.tensor(LOWER, dtype=torch.float64, device=device)
    upper = torch.tensor(UPPER, dtype=torch.float64, device=device)
    default = torch.tensor(DEFAULT, dtype=torch.float64, device=device)
    mean = default.expand(count, -1).clone()
    std = ((upper - lower) / 3).expand(count, -1).clone()
    generator = torch.Generator(device=device).manual_seed(budget.seed)
    baseline = _rollout(search_configs, mean, [True] * count, device=device, budget=budget, record_actions=True)
    best_cost = baseline["cost"].clone()
    best_parameters = mean.clone()
    best_heuristic = torch.ones(count, device=device, dtype=torch.bool)
    history = [best_cost.cpu().tolist()]
    # Retain every improving incumbent, so an original-step scalar rejection
    # of a coarse search winner can fall back to an earlier qualified witness.
    archive = [[(float(best_cost[index]), mean[index].clone(), True)] for index in range(count)]
    elites = max(1, min(population - 1, math.ceil(population * budget.elite_fraction)))
    duplicated = tuple(c for c in search_configs for _ in range(population))
    for _ in range(budget.iterations):
        samples = mean[:, None] + std[:, None] * torch.randn(
            count, population, len(PARAMETER_NAMES), generator=generator, device=device, dtype=torch.float64
        )
        samples = torch.maximum(lower, torch.minimum(upper, samples))
        samples[:, 0] = best_parameters
        samples[:, 1] = default
        heuristic = torch.zeros(count, population, device=device, dtype=torch.bool)
        heuristic[:, 0] = best_heuristic
        # Reserve one exact baseline so no search initialization can lose it.
        heuristic[:, 2] = True
        result = _rollout(duplicated, samples.flatten(0, 1), heuristic.flatten(), device=device, budget=budget)
        costs = result["cost"].reshape(count, population)
        costs_min, indices = costs.min(-1)
        improved = costs_min < best_cost
        rows = torch.arange(count, device=device)
        best_parameters[improved] = samples[rows, indices][improved]
        best_heuristic[improved] = heuristic[rows, indices][improved]
        best_cost = torch.minimum(best_cost, costs_min)
        for index in torch.nonzero(improved).flatten().cpu().tolist():
            archive[index].append((float(best_cost[index]), best_parameters[index].clone(), bool(best_heuristic[index])))
        # Do not use the discontinuous heuristic candidate to fit a misleading
        # continuous parameter distribution.
        continuous_cost = costs.masked_fill(heuristic, torch.inf)
        # There is always at least one continuous candidate (the default).
        # The incumbent may itself be heuristic, leaving two excluded slots.
        elite_count = min(elites, int((~heuristic).sum(-1).min()))
        elite_indices = continuous_cost.argsort(-1)[:, :elite_count]
        elite = samples.gather(1, elite_indices[:, :, None].expand(-1, -1, len(PARAMETER_NAMES)))
        mean = elite.mean(1)
        std = torch.maximum(elite.std(1, unbiased=False), (upper - lower) * .04)
        history.append(best_cost.cpu().tolist())
    archived = [(world, entry) for world, entries in enumerate(archive) for entry in sorted(entries, key=lambda entry: entry[0])]
    witness = _rollout(
        tuple(search_configs[world] for world, _ in archived),
        torch.stack([entry[1] for _, entry in archived]),
        [entry[2] for _, entry in archived], device=device, budget=budget, record_actions=True,
    )
    expected_costs = torch.tensor([entry[0] for _, entry in archived], device=device, dtype=torch.float64)
    if not torch.allclose(witness["cost"], expected_costs, rtol=1e-10, atol=1e-10):
        raise RuntimeError("replayed incumbent differs from the searched trajectory")
    rows = []
    for index, config in enumerate(configs):
        baseline_row = _row(baseline, index)
        baseline_qualification = qualify_witness(config, baseline_row["actions"], expected=baseline_row, max_dynamic_pressure=budget.max_dynamic_pressure) if qualify and baseline_row["success"] else None
        baseline_row["qualification"] = baseline_qualification
        baseline_row["qualified"] = bool(baseline_qualification and baseline_qualification["qualified"])
        choices = [(slot, entry) for slot, (world, entry) in enumerate(archived) if world == index]
        slot, selected = choices[0]
        row = _row(witness, slot)
        qualification = None
        rejected = []
        if qualify:
            for candidate_slot, entry in choices:
                candidate = _row(witness, candidate_slot)
                if not candidate["feasible"]:
                    continue
                checked = baseline_qualification if entry[2] else qualify_witness(config, candidate["actions"], expected=candidate, max_dynamic_pressure=budget.max_dynamic_pressure)
                if checked and checked["qualified"]:
                    slot, selected, row, qualification = candidate_slot, entry, candidate, checked
                    break
                rejected.append(dict(cost=candidate["cost"], qualification=checked))
            if qualification is None and row["success"]:
                qualification = baseline_qualification if selected[2] else qualify_witness(config, row["actions"], expected=row, max_dynamic_pressure=budget.max_dynamic_pressure)
        qualified = bool(qualification and qualification["qualified"])
        row.update(world=index, config=asdict(config), config_sha256=config_fingerprint(config),
                   selected_plan="heuristic" if selected[2] else "stage_aware_profile",
                   parameters=dict(zip(PARAMETER_NAMES, selected[1].cpu().tolist())),
                   baseline=baseline_row, search_best_cost=float(best_cost[index]),
                   qualification_fallback=slot != choices[0][0], rejected_incumbents=rejected,
                   reachability=("feasible_independent_scalar_witness" if qualified else
                                 "unqualified_search_candidate" if row["feasible"] else "unknown_search_failed"))
        row["qualification"], row["qualified"] = qualification, qualified
        rows.append(row)
    return dict(schema=SCHEMA, budget=asdict(budget), craft_count=count,
                candidate_rollouts=count * (1 + population * budget.iterations) + len(archived),
                baseline_successes=sum(r["baseline"]["success"] for r in rows),
                baseline_qualified_successes=sum(r["baseline"]["qualified"] for r in rows) if qualify else None,
                search_successes=sum(r["success"] for r in rows),
                qualified_successes=sum(r["qualified"] for r in rows),
                incumbent_cost_history=history, elapsed_s=time.monotonic() - start_time,
                search_physics_dt_s=[c.physics_dt for c in search_configs],
                qualification_requested=qualify, selection_bank_only=True,
                scope="known-craft planar 2-4 liquid stage shooting teacher; not a generalized learned policy or KSP proof",
                results=rows)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def generate_bank(count, *, seed, device="cpu", curriculum="stock"):
    """Use the trainer's actual envelope, rather than a similar local copy."""
    if curriculum not in ("efficient", "stock"):
        raise ValueError("curriculum must be efficient or stock")
    # Runtime import keeps this module usable by future trainer integrations.
    from .train_generalized import experiment_contract

    envelope, _, contract = experiment_contract(curriculum, "aero")
    env = BatchedAeroAscent(count, device=device, seed=seed, envelope=envelope,
                           physics_dt=contract["physics_dt"],
                           max_time=contract["max_time"],
                           orbit_tolerance=contract["orbit_tolerance"])
    return [env.export_config(index) for index in range(count)], contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, help="JSON list of frozen EnvConfig dictionaries; generated if omitted")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--curriculum", choices=("efficient", "stock"), default="stock",
                        help="trainer envelope for generated banks (default: stock)")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--physics-dt", type=float, help="optional search-only integration step")
    parser.add_argument("--max-dynamic-pressure", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="CPU tensor threads; one avoids small-rollout oversubscription")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-qualify", action="store_true", help="omit independent scalar replay/coast; report remains unqualified")
    args = parser.parse_args(argv)
    if args.count < 1 or args.cpu_threads < 1:
        parser.error("count and cpu-threads must be positive")
    if torch.device(args.device).type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    if args.bank:
        bank = json.loads(args.bank.read_text())
        configs = [EnvConfig.from_dict(row) for row in bank]
        generation_contract = None
    else:
        configs, generation_contract = generate_bank(
            args.count, device=args.device, seed=args.seed, curriculum=args.curriculum,
        )
    result = plan_ascent(configs, device=args.device, qualify=not args.no_qualify,
                         budget=SearchBudget(population=args.population, iterations=args.iterations,
                                             seed=args.seed, physics_dt=args.physics_dt,
                                             max_dynamic_pressure=args.max_dynamic_pressure))
    result["bank_source"] = str(args.bank) if args.bank else "generated_frozen_bank"
    result["generation_contract"] = generation_contract
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("craft_count", "baseline_successes", "search_successes", "qualified_successes", "candidate_rollouts", "elapsed_s")}))


if __name__ == "__main__":
    main()
