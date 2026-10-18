"""Inspect KSP, or explicitly fly a liquid-fuel launch with the shared policy API.

Default invocation is read-only. Flight requires --fly, a controller, and an
explicit activation-stage plan. No quicksave/quickload or GUI interaction occurs.
Examples (stage numbers must come from inspection of the actual craft):
    python -m orbiter.live
    python -m orbiter.live --fly --scripted --stage-plan 3,2,1 --fuel-stages 2,-1
    python -m orbiter.live --fly --model runs/ppo/best.zip --stage-plan 3,2,1 --fuel-stages 2,-1

The simulator is planar. This adapter commands heading 90 degrees, reports
out-of-plane velocity, and uses actual KSP apses for the success decision.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np

from .env import G0, EnvConfig, build_observation
from .checkpoints import load_controller
from .guidance import expert_action
from .ownership import FlightOwnership


@contextmanager
def rpc_deadline(seconds):
    """Bound blocking kRPC handshakes/reads on the macOS/Linux main thread."""
    if seconds <= 0:
        raise TimeoutError('kRPC deadline expired')
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)
    timer = signal.getitimer(signal.ITIMER_REAL)
    parent_first = timer[0] > 0 and timer[0] <= seconds
    parent_fired = False
    def expired(signum, frame):
        nonlocal parent_fired
        if parent_first:
            parent_fired = True
            if callable(previous):
                previous(signum, frame)
        raise TimeoutError(f'kRPC operation exceeded {seconds:g} wall-clock seconds')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, timer[0]) if timer[0] > 0 else seconds)
    try:
        yield
    finally:
        # A nested read must neither extend the parent's deadline nor restart a
        # parent alarm already delivered while inside this scope.
        remaining = max(1e-6, timer[0]-(time.monotonic()-started)) if timer[0] > 0 and not parent_fired else 0.
        signal.signal(signal.SIGALRM, previous)
        signal.setitimer(signal.ITIMER_REAL, remaining, timer[1])


def wait_for_ut(space_center, target_ut, wall_timeout=15.0, *, clock=time.monotonic,
                sleeper=time.sleep, rpc_timeout=5.0):
    """Wait for game time, with a finite wall bound even when KSP is paused."""
    deadline = clock() + wall_timeout
    previous_ut = None
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError('KSP universal time did not advance before the wall-clock deadline')
        with rpc_deadline(min(rpc_timeout, remaining)):
            current_ut = float(space_center.ut)
        if previous_ut is not None and current_ut < previous_ut - 1e-6:
            raise RuntimeError('KSP universal time moved backwards; flight was reverted or loaded')
        if current_ut >= target_ut - 1e-8:
            return current_ut
        previous_ut = current_ut
        sleeper(min(0.02, max(0, deadline-clock())))


def jsonable(value):
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class FlightLog:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A flight record must never silently overwrite an earlier attempt.
        self.handle = path.open('x', buffering=1)

    def write(self, event, **data):
        self.handle.write(json.dumps(jsonable(dict(event=event, wall_time=time.time(), **data)),
                                     allow_nan=False) + '\n')

    def close(self):
        self.handle.close()


def propellant_mass(resources, names):
    present = set(resources.names)
    return sum(float(resources.amount(name)) * float(resources.density(name))
               for name in names if name in present)


class FuelMonitor:
    """Measure actual stage propellant; never infer mass from a generic vehicle."""
    def __init__(self, vessel, fuel_stages):
        self.fuel_stages = tuple(fuel_stages)
        self.names = sorted({name for engine in vessel.parts.engines for name in engine.propellant_names})
        self.initial_mass = float(vessel.mass)
        self.initial_fuel = tuple(self.mass(vessel, i) for i in range(len(self.fuel_stages)))
        if any(value <= 0 for value in self.initial_fuel):
            raise ValueError('Every --fuel-stages entry must contain loaded engine propellant')

    def mass(self, vessel, index):
        resources = vessel.resources_in_decouple_stage(self.fuel_stages[index], cumulative=False)
        return propellant_mass(resources, self.names)

    def fraction(self, vessel, index):
        return self.mass(vessel, index) / self.initial_fuel[index]


def validate_vehicle_mass(config, fuels, tolerance=0.1):
    """Reject an obviously different craft before applying a learned policy.

    Matching these masses is necessary, not sufficient: engine/aerodynamic
    calibration and held-out flights are still required for transfer.
    """
    expected = config.payload_mass + sum(s.dry_mass+s.fuel_mass for s in config.stages)
    if abs(fuels.initial_mass/expected-1) > tolerance:
        raise ValueError('Live wet mass differs from the trained craft by over 10%; calibrate the craft and retrain first')
    if len(fuels.initial_fuel) != len(config.stages):
        raise ValueError('Live fuel stage count differs from the trained craft')
    for i, (actual, stage) in enumerate(zip(fuels.initial_fuel, config.stages)):
        if abs(actual/stage.fuel_mass-1) > tolerance:
            raise ValueError(f'Live propellant mass in stage {i} differs from training by over 10%; calibrate and retrain first')


class LiveState:
    """Read-only telemetry adapter for expert_action and build_observation.

    East is transformed from kRPC's documented surface-frame z axis. Using an
    ordinary right-handed cross product on kRPC's left-handed axes reverses it.
    """
    def __init__(self, space_center, vessel, config, fuels, stage=0, start_ut=None):
        self.space_center, self.vessel, self.config, self.fuels = space_center, vessel, config, fuels
        self.stage = stage
        self.start_ut = float(space_center.ut) if start_ut is None else start_ut
        self.data = {}

    def sample(self):
        sc, vessel = self.space_center, self.vessel
        body = vessel.orbit.body
        begin = float(sc.ut)
        frame = body.non_rotating_reference_frame
        position = np.asarray(vessel.position(frame), dtype=float)
        velocity = np.asarray(vessel.velocity(frame), dtype=float)
        radius = float(np.linalg.norm(position))
        if radius <= 0 or not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise RuntimeError('Invalid inertial position/velocity from KSP')
        radial = position / radius
        east = np.asarray(sc.transform_direction((0., 0., 1.), vessel.surface_reference_frame, frame), dtype=float)
        north = np.asarray(sc.transform_direction((0., 1., 0.), vessel.surface_reference_frame, frame), dtype=float)
        east /= np.linalg.norm(east)
        north /= np.linalg.norm(north)
        flight = vessel.flight(body.reference_frame)
        # Flight's Euler angles use its requested frame. The planet-fixed
        # axes are not the local horizon, even on an equatorial launchpad.
        attitude = vessel.flight(vessel.surface_reference_frame)
        mass = float(vessel.mass)
        if mass <= 0:
            raise RuntimeError('Vessel has no positive mass')
        self.data = {
            'altitude': radius - float(body.equatorial_radius),
            'vertical_speed': float(velocity @ radial),
            'horizontal_speed': float(velocity @ east),
            'north_speed': float(velocity @ north),
            'apoapsis': float(vessel.orbit.apoapsis_altitude),
            'periapsis': float(vessel.orbit.periapsis_altitude),
            'energy': float(velocity @ velocity)/2 - float(body.gravitational_parameter)/radius,
            'momentum': radius * float(velocity @ east),
            'dynamic_pressure': float(flight.dynamic_pressure),
            'surface_altitude': float(flight.surface_altitude),
            'latitude': float(flight.latitude),
            'pitch': float(attitude.pitch),
            'heading': float(attitude.heading),
            'throttle': float(vessel.control.throttle),
            'mass': mass,
            'available_thrust': float(vessel.available_thrust),
            'situation': str(vessel.situation).lower(),
            'current_stage': int(vessel.control.current_stage),
            'stage': self.stage,
            'position_inertial': position.tolist(),
            'velocity_inertial': velocity.tolist(),
            'ut_begin': begin,
        }
        self.data['ut_end'] = float(sc.ut)
        self.data['time'] = self.data['ut_end'] - self.start_ut
        if self.data['time'] < -1e-6:
            raise RuntimeError('KSP universal time moved backwards; flight was reverted or loaded')
        if self.fuels is not None:
            self.data['fuel_fraction'] = self.fuels.fraction(vessel, self.stage)
        return self.data

    def telemetry(self):
        return self.data

    def observation(self):
        d = self.data
        if self.fuels is None:
            raise RuntimeError('Stage fuel mapping is required for policy observations')
        observation = build_observation(
            self.config, d, fuel_fraction=d['fuel_fraction'],
            mass_fraction=d['mass']/self.fuels.initial_mass, stage=self.stage,
            pitch=d['pitch'], throttle=d['throttle'], elapsed=d['time'],
            thrust_to_weight=d['available_thrust']/d['mass']/G0,
        )
        if not np.isfinite(observation).all():
            raise RuntimeError('Live policy observation contains a non-finite value')
        return observation


def engine_status(vessel):
    return [dict(name=engine.part.name, activation_stage=int(engine.part.stage),
                 decouple_stage=int(engine.part.decouple_stage), active=bool(engine.active),
                 has_fuel=bool(engine.has_fuel), available_thrust=float(engine.available_thrust),
                 can_shutdown=bool(engine.can_shutdown), propellants=list(engine.propellant_names))
            for engine in vessel.parts.engines]


class StageController:
    """A bounded, explicitly listed sequence for ordinary serial liquid stages."""
    def __init__(self, space_center, vessel, plan, *, max_transitions=12, on_event=None):
        self.space_center, self.vessel = space_center, vessel
        self.plan = tuple(plan)
        self.max_transitions = max_transitions
        self.transitions = 0
        self.propulsion_index = -1
        self.engine_groups = set()
        self.control_part = vessel.parts.controlling
        self.on_event = on_event or (lambda *args, **kwargs: None)
        current = int(vessel.control.current_stage)
        if not self.plan or self.plan != tuple(range(current-1, current-1-len(self.plan), -1)):
            raise ValueError('--stage-plan must list consecutive next activation stages, starting at current_stage-1')
        if min(self.plan) < 0 or len(self.plan) > max_transitions:
            raise ValueError('Stage plan exceeds the nonnegative stage numbers or transition bound')
        for engine in engine_status(vessel):
            if (engine['active'] or engine['activation_stage'] in self.plan) and (
                not engine['can_shutdown'] or {'SolidFuel', 'IntakeAir'} & set(engine['propellants'])
            ):
                raise ValueError('This adapter supports shutdown-capable liquid rocket engines; SRBs/jets need another controller')

    def ready_engines(self):
        return [engine for engine in self.vessel.parts.engines
                if engine.active and engine.has_fuel and float(engine.available_thrust) > 1e-3]

    def register_ready(self):
        engines = self.ready_engines()
        if not engines:
            return False
        group = frozenset(int(engine.part.stage) for engine in engines)
        if len(group) != 1:
            raise RuntimeError('Simultaneous engine activation stages need a calibrated parallel/asparagus controller')
        if group not in self.engine_groups:
            self.engine_groups.add(group)
            self.propulsion_index += 1
            self.on_event('propulsion_stage', index=self.propulsion_index, activation_stage=list(group))
        return True

    def activate(self):
        if self.transitions >= len(self.plan) or self.transitions >= self.max_transitions:
            return False
        before = int(self.vessel.control.current_stage)
        expected = self.plan[self.transitions]
        if expected != before - 1:
            raise RuntimeError('KSP stage changed outside this controller; stopping instead of guessing')
        self.vessel.control.activate_next_stage()
        # Preserve the original controlling part when a decoupler changes which
        # vessel KSP marks active. Never follow a discarded booster by accident.
        self.vessel = self.control_part.vessel
        after = int(self.vessel.control.current_stage)
        if after != expected:
            raise RuntimeError(f'Expected stage {expected}, received {after}')
        if self.space_center.active_vessel != self.vessel:
            self.space_center.active_vessel = self.vessel
        self.transitions += 1
        self.on_event('activate_stage', before=before, after=after, ut=float(self.space_center.ut))
        return True

    def advance_until_ready(self, wait, *, initial=False):
        if initial:
            if not self.activate():
                return False
            wait()
        while not self.register_ready():
            if not self.activate():
                return False
            wait()
        return True

    def depleted(self):
        # Individual engines expose potential thrust even while inactive;
        # depletion must consider the currently ignited engines only.
        active = [engine for engine in self.vessel.parts.engines if engine.active]
        return bool(active) and all(not engine.has_fuel or float(engine.available_thrust) <= 1e-3
                                    for engine in active)


def terminal_reason(data, config, *, left_ground=False):
    success = (data['energy'] < 0 and math.isfinite(data['apoapsis'])
               and data['periapsis'] >= max(config.atmosphere_depth+1000, config.target_altitude-config.orbit_tolerance)
               and data['apoapsis'] <= config.target_altitude+config.orbit_tolerance)
    if success:
        return 'orbit_achieved'
    if data['energy'] >= 0:
        return 'escape'
    if data['surface_altitude'] < -0.1 or (left_ground and any(word in data['situation'] for word in ('landed', 'splashed'))):
        return 'impact'
    if data['time'] >= config.max_time-1e-8:
        return 'time_limit'
    return ''


def decode_action(action, state, *, scripted=False):
    value = np.asarray(action, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError('Controller must produce two finite actions')
    value = np.clip(value, -1, 1)
    if state.config.residual and not scripted:
        value = np.clip(expert_action(state) + value * np.array([0.25, 0.2]), -1, 1)
    return value, float((value[0]+1)/2), float((value[1]+1)*50-10)


def stop_controls(vessels, timeout=3):
    """Try both shutdown operations independently, even after an RPC failure."""
    errors = []
    unique = []
    for vessel in vessels:
        if vessel not in unique:
            unique.append(vessel)
    for vessel in unique:
        for operation, callback in (
            ('throttle_zero', lambda v=vessel: setattr(v.control, 'throttle', 0.0)),
            ('autopilot_disengage', lambda v=vessel: v.auto_pilot.disengage()),
        ):
            try:
                with rpc_deadline(timeout):
                    callback()
            except Exception as exc:
                errors.append(f'{operation}: {type(exc).__name__}: {exc}')
    return errors


def fly(connection, config, *, plan, fuel_stages, model=None, scripted=False,
        log, max_steps=None, rpc_timeout=10, wall_timeout=1800, step_wall_timeout=20,
        stage_settle=0.5, save_name=None):
    """One explicit launch attempt. Never resets/loads the user's game."""
    if len(fuel_stages) != len(config.stages):
        raise ValueError('--fuel-stages must map every propulsive stage in the trained configuration')
    with rpc_deadline(rpc_timeout):
        sc = connection.space_center
        vessel = sc.active_vessel
    touched = [vessel]
    deadline = time.monotonic() + wall_timeout
    max_steps = max_steps or math.ceil(config.max_time/config.control_dt)+2
    cleanup_errors = []
    controls_owned = False
    try:
        with rpc_deadline(rpc_timeout):
            if not any(word in str(vessel.situation).lower() for word in ('pre_launch', 'prelaunch', 'landed')):
                raise ValueError('--fly starts a launch from the pad; use a fresh craft in a dedicated save')
            body = vessel.orbit.body
            if abs(float(body.gravitational_parameter)/config.mu-1) > 0.01 or abs(float(body.equatorial_radius)/config.radius-1) > 0.01:
                raise ValueError('Live body differs from the trained environment; capture/calibrate and retrain first')
            if abs(float(vessel.flight(body.reference_frame).latitude)) > 1:
                raise ValueError('The planar heading-90 launch adapter requires an approximately equatorial pad')
            stager = StageController(sc, vessel, plan, on_event=log.write)
            fuels = FuelMonitor(vessel, fuel_stages)
            if model is not None:
                validate_vehicle_mass(config, fuels)
            start_ut = float(sc.ut)
            state = LiveState(sc, vessel, config, fuels, start_ut=start_ut)
            state.sample()
            log.write('flight_start', config=asdict(config), stage_plan=plan, fuel_stages=fuel_stages,
                      controller='scripted' if scripted else type(model).__name__, telemetry=state.data)
            if save_name:
                sc.save(save_name)
                log.write('named_save', name=save_name, ut=float(sc.ut))
            controls_owned = True
            vessel.control.throttle = 0.0
            vessel.control.sas = False
            vessel.control.pitch = vessel.control.yaw = vessel.control.roll = 0.0
            ap = vessel.auto_pilot
            ap.reference_frame = vessel.surface_reference_frame
            ap.target_pitch_and_heading(90.0, 90.0)
            ap.engage()
            autopilot_vessel = vessel
            vessel.control.throttle = 1.0
        def settle():
            if time.monotonic() >= deadline:
                raise TimeoutError('Flight wall-clock limit reached')
            with rpc_deadline(rpc_timeout):
                ut = float(sc.ut)
            wait_for_ut(sc, ut+stage_settle, min(step_wall_timeout, max(0.01, deadline-time.monotonic())), rpc_timeout=rpc_timeout)
        with rpc_deadline(rpc_timeout):
            ready = stager.activate()
        if ready:
            settle()
        while ready:
            with rpc_deadline(rpc_timeout):
                if stager.register_ready():
                    break
                ready = stager.activate()
            if ready:
                settle()
        if not ready:
            raise RuntimeError('Stage plan finished without an ignited fueled engine')
        left_ground = False
        reason = 'step_limit'
        for step in range(max_steps):
            if time.monotonic() >= deadline:
                raise TimeoutError('Flight wall-clock limit reached')
            with rpc_deadline(rpc_timeout):
                vessel = stager.vessel
                if vessel not in touched:
                    touched.append(vessel)
                if stager.propulsion_index >= len(config.stages):
                    raise RuntimeError('Live propulsive stage count exceeds the trained configuration')
                state.vessel, state.stage = vessel, stager.propulsion_index
                data = state.sample()
                left_ground |= ('flying' in data['situation'] or data['surface_altitude'] > 20)
                reason = terminal_reason(data, config, left_ground=left_ground)
                observation = state.observation()
                log.write('telemetry', step=step, observation=observation, **data)
            if reason:
                break
            with rpc_deadline(rpc_timeout):
                depleted = stager.depleted()
            if depleted:
                while True:
                    with rpc_deadline(rpc_timeout):
                        ready = stager.activate()
                    if not ready:
                        reason = 'out_of_fuel'
                        break
                    settle()
                    with rpc_deadline(rpc_timeout):
                        ready = stager.register_ready()
                    if ready:
                        break
                if reason:
                    break
                continue  # sample the new mass/fuel/stage before inference
            raw_action = expert_action(state) if scripted else model.predict(observation, deterministic=True)[0]
            action, throttle, pitch = decode_action(raw_action, state, scripted=scripted)
            with rpc_deadline(rpc_timeout):
                changed_vessel = vessel != autopilot_vessel
                if changed_vessel:
                    ap = vessel.auto_pilot
                    ap.reference_frame = vessel.surface_reference_frame
                ap.target_pitch_and_heading(pitch, 90.0)
                # kRPC Engage restarts its PID controller. Preserve its state
                # between decisions; a separated vessel needs a fresh autopilot.
                if changed_vessel:
                    ap.engage()
                    autopilot_vessel = vessel
                vessel.control.throttle = throttle
                command_ut = float(sc.ut)
            log.write('action', step=step, ut=command_ut, policy_action=raw_action,
                      applied_action=action, throttle=throttle, pitch_setpoint=pitch, heading_setpoint=90.0)
            wait_for_ut(sc, command_ut+config.control_dt,
                        min(step_wall_timeout, max(0.01, deadline-time.monotonic())), rpc_timeout=rpc_timeout)
        else:
            # The last held action can reach orbit. Sample it before declaring
            # the decision bound, rather than returning the previous telemetry.
            with rpc_deadline(rpc_timeout):
                data = state.sample()
                reason = terminal_reason(data, config, left_ground=left_ground) or 'step_limit'
                log.write('telemetry', step=max_steps, observation=state.observation(), **data)
        result = dict(success=reason=='orbit_achieved', termination_reason=reason,
                      steps=step+1, final_telemetry=state.data)
        log.write('flight_end', **result)
        return result
    except BaseException as exc:
        log.write('flight_error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if controls_owned:
            if 'stager' in locals() and stager.vessel not in touched:
                touched.append(stager.vessel)
            cleanup_errors = stop_controls(touched, min(rpc_timeout, 3))
            log.write('controls_released', errors=cleanup_errors)
            if cleanup_errors:
                print('Control release encountered RPC errors: ' + '; '.join(cleanup_errors), flush=True)


def inspect(connection, config):
    sc, vessel = connection.space_center, connection.space_center.active_vessel
    current = int(vessel.control.current_stage)
    engines = engine_status(vessel)
    state = LiveState(sc, vessel, config, None)
    telemetry = state.sample()
    activation_stages = sorted({engine['activation_stage'] for engine in engines if engine['activation_stage'] >= 0}, reverse=True)
    lowest_engine = min(activation_stages) if activation_stages else current
    return dict(mode='inspect_only', vessel=vessel.name, telemetry=telemetry, engines=engines,
                suggested_activation_plan=list(range(current-1, lowest_engine-1, -1)),
                suggested_fuel_stages=[next(e['decouple_stage'] for e in engines if e['activation_stage']==stage)
                                       for stage in activation_stages],
                notes=['No controls, staging, saves or loads were changed.',
                       'Inspect the suggested plan: clamp/decoupler-only stages may separate engine ignition.',
                       'Fuel stages are tank decouple groups in propulsive-stage order; crossfeed/asparagus and SRBs are unsupported.',
                       'Policy transfer requires actual craft calibration; generic simulator success is insufficient.'])


def stage_numbers(value):
    try:
        result = tuple(int(item) for item in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated integer stages, such as 3,2,1') from exc
    if not result:
        raise argparse.ArgumentTypeError('Stage list cannot be empty')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fly', action='store_true', help='Explicitly operate the current launch craft')
    controller = parser.add_mutually_exclusive_group()
    controller.add_argument('--scripted', action='store_true')
    controller.add_argument('--model', type=Path)
    parser.add_argument('--config', type=Path, help='Scripted/inspection configuration; model metadata owns the learned contract')
    parser.add_argument('--stage-plan', type=stage_numbers)
    parser.add_argument('--fuel-stages', type=stage_numbers)
    parser.add_argument('--address', default='127.0.0.1')
    parser.add_argument('--rpc-port', type=int, default=50000)
    parser.add_argument('--rpc-timeout', type=float, default=15)
    parser.add_argument('--step-wall-timeout', type=float, default=30)
    parser.add_argument('--wall-timeout', type=float, default=1800)
    parser.add_argument('--stage-settle', type=float, default=0.5, help='Game seconds after each stage activation')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--save-name', help='Optional named dedicated save made before this flight; never quicksave/load')
    parser.add_argument('--output', type=Path, default=Path('runs/live')/f'flight_{time.time_ns()}.jsonl')
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or value <= 0 for value in (
            args.rpc_timeout, args.step_wall_timeout, args.wall_timeout, args.stage_settle)):
        parser.error('Timeouts and stage-settle must be finite and positive')
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error('--max-steps must be positive')
    if args.fly and not (args.scripted or args.model):
        parser.error('--fly requires --scripted or --model')
    if args.fly and (args.stage_plan is None or args.fuel_stages is None):
        parser.error('--fly requires an inspected --stage-plan and --fuel-stages mapping')
    if args.save_name and not args.fly:
        parser.error('--save-name requires --fly')
    if args.model and args.config:
        parser.error('Use model metadata for learned-policy configuration; --config is for scripted/inspect mode')
    return args


def main(argv=None):
    args = parse_args(argv)
    connection = None
    log = None
    ownership = None
    try:
        if args.fly:
            ownership = FlightOwnership(args.address, args.rpc_port)
            ownership.acquire()
        model, config = load_controller(args.model) if args.model else (None, EnvConfig.load(args.config) if args.config else EnvConfig())
        import krpc
        with rpc_deadline(args.rpc_timeout):
            connection = krpc.connect(name='Orbiter flight' if args.fly else 'Read-only orbiter inspection',
                                      address=args.address, rpc_port=args.rpc_port, stream_port=None,
                                      use_pregenerated_stubs=False)
        if not args.fly:
            with rpc_deadline(args.rpc_timeout):
                result = inspect(connection, config)
        else:
            log = FlightLog(args.output)
            if args.model:
                log.write('controller_checkpoint', path=str(args.model.resolve()),
                          sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                          algorithm=type(model).__name__, num_timesteps=int(model.num_timesteps))
            result = fly(connection, config, plan=args.stage_plan, fuel_stages=args.fuel_stages,
                         model=model, scripted=args.scripted, log=log, max_steps=args.max_steps,
                         rpc_timeout=args.rpc_timeout, wall_timeout=args.wall_timeout,
                         step_wall_timeout=args.step_wall_timeout, stage_settle=args.stage_settle,
                         save_name=args.save_name)
            result['recording'] = str(args.output)
        print(json.dumps(jsonable(result), indent=2, allow_nan=False))
        return 0 if not args.fly or result.get('success', False) else 1
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        print(f'Live operation stopped: {type(exc).__name__}: {exc}', flush=True)
        return 1
    finally:
        try:
            try:
                if log is not None:
                    log.close()
            finally:
                if connection is not None:
                    try:
                        with rpc_deadline(min(args.rpc_timeout, 3)):
                            connection.close()
                    except Exception as exc:
                        print(f'Connection close: {type(exc).__name__}: {exc}', flush=True)
        finally:
            if ownership is not None:
                ownership.release()


if __name__ == '__main__':
    raise SystemExit(main())
