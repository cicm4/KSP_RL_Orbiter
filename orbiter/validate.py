"""Validate a learned checkpoint in actual KSP, including a bounded coast check.

Normal mode flies the current pad craft. --pad-save explicitly reloads a named
snapshot before each trial. --verify-current only reads the current orbit; it
cannot establish which controller reached it. No headless fallback exists.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from .live import (FlightLog, fly, jsonable, load_controller, rpc_deadline,
                   stage_numbers, wait_for_ut)
from .ownership import FlightOwnership


def orbit_failures(sample, config):
    required = ('ut_begin', 'ut_end', 'apoapsis', 'periapsis', 'eccentricity', 'throttle')
    if any(not isinstance(sample.get(key), (int, float)) or not math.isfinite(sample[key])
           for key in required):
        return ['nonfinite_or_missing_telemetry']
    failures = []
    if sample['ut_end'] < sample['ut_begin'] - 1e-6:
        failures.append('universal_time_moved_backwards')
    if not 0 <= sample['eccentricity'] < 1 or sample['apoapsis'] < sample['periapsis']:
        failures.append('not_a_bound_orbit')
    minimum = max(config.atmosphere_depth + 1000,
                  config.target_altitude - config.orbit_tolerance)
    if sample['periapsis'] < minimum:
        failures.append('periapsis_below_target')
    if sample['apoapsis'] > config.target_altitude + config.orbit_tolerance:
        failures.append('apoapsis_above_target')
    if not 0 <= sample['throttle'] <= 1e-5:
        failures.append('throttle_not_zero')
    if 'orbiting' not in sample.get('situation', '').lower():
        failures.append('vessel_not_orbiting')
    return failures


def read_orbit(space_center, vessel):
    """Independent of LiveState projections: read KSP's actual orbit directly."""
    begin = float(space_center.ut)
    orbit = vessel.orbit
    return dict(ut_begin=begin, vessel=vessel.name,
                apoapsis=float(orbit.apoapsis_altitude),
                periapsis=float(orbit.periapsis_altitude),
                eccentricity=float(orbit.eccentricity),
                throttle=float(vessel.control.throttle),
                situation=str(vessel.situation), ut_end=float(space_center.ut))


def verify_coast(connection, config, *, seconds=30., wall_timeout=180., rpc_timeout=15., log=None):
    """Read-only post-cutoff probes over the requested elapsed simulation time."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Coast seconds must be positive and finite')
    sc = connection.space_center
    deadline = time.monotonic() + wall_timeout
    with rpc_deadline(rpc_timeout):
        vessel = sc.active_vessel
    samples = []
    failures = []
    start = None
    previous_end = None
    while True:
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Coast wall-clock deadline expired')
        with rpc_deadline(min(rpc_timeout, remaining)):
            if sc.active_vessel != vessel:
                raise RuntimeError('Active vessel changed during independent coast verification')
            sample = read_orbit(sc, vessel)
        if previous_end is not None and sample['ut_begin'] < previous_end-1e-6:
            raise RuntimeError('KSP universal time moved backwards during coast verification')
        failures = orbit_failures(sample, config)
        samples.append(sample)
        if log is not None:
            log.write('independent_coast_probe', failures=failures, **sample)
        if failures:
            break
        if start is None:
            start = sample['ut_end']
        previous_end = sample['ut_end']
        if sample['ut_begin'] >= start+seconds-1e-8:
            break
        wait_for_ut(sc, min(start+seconds, sample['ut_end']+1.),
                    max(.001, deadline-time.monotonic()), rpc_timeout=rpc_timeout)
    elapsed = samples[-1]['ut_begin']-samples[0]['ut_end']
    return dict(success=not failures and elapsed >= seconds-1e-8,
                requested_coast_seconds=seconds, observed_coast_seconds=max(0., elapsed),
                failures=failures, samples=samples,
                limitation='This bounded coast check does not demonstrate a complete orbit or long-term stability.')


def wait_for_pad(connection, first_stage, *, timeout=60., rpc_timeout=15.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        with rpc_deadline(min(rpc_timeout, deadline-time.monotonic())):
            try:
                vessel = connection.space_center.active_vessel
                if ('pre_launch' in str(vessel.situation).lower()
                        or 'prelaunch' in str(vessel.situation).lower()):
                    if int(vessel.control.current_stage) == first_stage+1:
                        return
            except (RuntimeError, ValueError):
                pass
        time.sleep(.1)
    raise TimeoutError('Named snapshot did not become the expected pre-launch craft')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--stage-plan', type=stage_numbers)
    parser.add_argument('--fuel-stages', type=stage_numbers)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pad-save', help='Explicitly load this named dedicated pad snapshot before each flight')
    parser.add_argument('--verify-current', action='store_true', help='Read-only coast verification; never fly/load/stage')
    parser.add_argument('--coast-seconds', type=float, default=30.)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--address', default='127.0.0.1')
    parser.add_argument('--rpc-port', type=int, default=50000)
    parser.add_argument('--rpc-timeout', type=float, default=15.)
    parser.add_argument('--wall-timeout', type=float, default=1800.)
    parser.add_argument('--coast-wall-timeout', type=float, default=180.)
    args = parser.parse_args(argv)
    if args.trials < 1 or any(not math.isfinite(value) or value <= 0 for value in (
            args.coast_seconds, args.rpc_timeout, args.wall_timeout, args.coast_wall_timeout)):
        parser.error('Trials must be positive; timeouts/coast seconds must be finite and positive')
    if args.verify_current:
        if args.pad_save or args.trials != 1 or args.stage_plan or args.fuel_stages:
            parser.error('--verify-current accepts no load/staging options and exactly one trial')
    else:
        if not args.stage_plan or not args.fuel_stages:
            parser.error('Flight validation requires --stage-plan and --fuel-stages')
        if (min(args.stage_plan) < 0 or args.stage_plan != tuple(
                range(args.stage_plan[0], args.stage_plan[0]-len(args.stage_plan), -1))):
            parser.error('--stage-plan must contain consecutive descending nonnegative stages')
        if min(args.fuel_stages) < -1 or len(set(args.fuel_stages)) != len(args.fuel_stages):
            parser.error('--fuel-stages must contain unique decouple groups, each at least -1')
        if args.trials > 1 and not args.pad_save:
            parser.error('Multiple flight trials require an explicit --pad-save')
        if args.pad_save and (args.pad_save in {'quicksave', 'persistent'}
                              or any(char in args.pad_save for char in '/\\')):
            parser.error('--pad-save must name a dedicated snapshot, excluding quicksave/persistent')
    return args


def main(argv=None):
    args = parse_args(argv)
    summary = dict(schema_version=1, source='actual_ksp_krpc', success=False,
                   mode='verify_current_read_only' if args.verify_current else 'learned_policy_flight',
                   trials=[], controller_flight_executed=False,
                   limitation='A current-orbit probe does not establish controller attribution; a bounded coast is not a full-orbit test.')
    connection, ownership = None, None
    owns_summary = False
    print_summary = True
    path = args.output_dir/'validation.json'
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Reserve the summary before connecting or operating the game.
        with path.open('x') as output:
            json.dump(summary, output)
        owns_summary = True
        # Read-only verification also needs a quiet endpoint: its independent
        # coast samples must not race a separate flight worker's controls.
        ownership = FlightOwnership(args.address, args.rpc_port)
        ownership.acquire()
        checkpoint_sha = hashlib.sha256(args.model.read_bytes()).hexdigest()
        model, config = load_controller(args.model)
        if hashlib.sha256(args.model.read_bytes()).hexdigest() != checkpoint_sha:
            raise RuntimeError('Checkpoint changed while loading; validate an immutable checkpoint copy')
        summary.update(model_path=str(args.model.resolve()),
                       model_sha256=checkpoint_sha,
                       algorithm=type(model).__name__, num_timesteps=int(model.num_timesteps),
                       environment=asdict(config), stage_plan=args.stage_plan,
                       fuel_stages=args.fuel_stages, pad_save=args.pad_save)
        if not args.verify_current and len(args.fuel_stages) != len(config.stages):
            raise ValueError('--fuel-stages must map every model propulsion stage')
        import krpc
        with rpc_deadline(args.rpc_timeout):
            connection = krpc.connect(name='Orbiter actual KSP validation', address=args.address,
                                      rpc_port=args.rpc_port, stream_port=None,
                                      use_pregenerated_stubs=False)
        for index in range(args.trials):
            trial = dict(trial=index+1, success=False)
            summary['trials'].append(trial)
            log = FlightLog(args.output_dir/f'trial_{index+1:03d}.jsonl')
            try:
                log.write('validation_checkpoint', model_sha256=summary['model_sha256'],
                          num_timesteps=summary['num_timesteps'], mode=summary['mode'])
                if not args.verify_current:
                    if args.pad_save:
                        with rpc_deadline(args.rpc_timeout):
                            connection.space_center.load(args.pad_save)
                        log.write('explicit_named_snapshot_load', name=args.pad_save)
                        wait_for_pad(connection, args.stage_plan[0], rpc_timeout=args.rpc_timeout)
                    summary['controller_flight_executed'] = True
                    trial['flight'] = fly(connection, config, plan=args.stage_plan,
                                          fuel_stages=args.fuel_stages, model=model, log=log,
                                          rpc_timeout=args.rpc_timeout, wall_timeout=args.wall_timeout)
                # fly has returned and released its controls before this separate probe.
                trial['coast'] = verify_coast(connection, config, seconds=args.coast_seconds,
                                              wall_timeout=args.coast_wall_timeout,
                                              rpc_timeout=args.rpc_timeout, log=log)
                trial['success'] = trial['coast']['success'] and (
                    args.verify_current or trial['flight']['success'])
                log.write('validation_trial_end', success=trial['success'])
            finally:
                log.close()
        summary['success'] = all(trial['success'] for trial in summary['trials'])
    except FileExistsError as error:
        # Never overwrite an existing record, including the summary itself.
        print(f'Validation stopped: {error}', flush=True)
        print_summary = False
        return 1
    except Exception as error:
        summary['error'] = f'{type(error).__name__}: {error}'
        summary['success'] = False
    finally:
        try:
            if connection is not None:
                try:
                    with rpc_deadline(min(args.rpc_timeout, 3)):
                        connection.close()
                except Exception as error:
                    summary['connection_close_error'] = str(error)
            if owns_summary:
                path.write_text(json.dumps(jsonable(summary), indent=2, allow_nan=False)+'\n')
            if print_summary:
                print(json.dumps(jsonable(summary), indent=2, allow_nan=False), flush=True)
        finally:
            if ownership is not None:
                ownership.release()
    return 0 if summary['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
