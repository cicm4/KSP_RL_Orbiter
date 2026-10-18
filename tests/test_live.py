"""Exercise live flight boundaries with fake kRPC objects, never the game."""
from dataclasses import asdict
import json
import math
import signal
import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from orbiter.env import EnvConfig, HeadlessOrbiterEnv
from orbiter import live


@pytest.mark.parametrize("outer,inner", [(.02, .08), (.08, .02)],
                         ids=["parent_deadline_first", "inner_deadline_first"])
def test_nested_rpc_arms_and_dispatches_the_earliest_deadline(monkeypatch, outer, inner):
    # Signal delivery depends on the host scheduler. Check the armed deadline
    # and dispatched handler directly, retaining real-signal integration below.
    now = [100.0]
    original_handler = lambda signum, frame: None
    handler = [original_handler]
    timer = [(0.0, 0.0)]

    def set_timer(which, seconds, interval=0.0):
        previous = timer[0]
        timer[0] = (seconds, interval)
        return previous

    monkeypatch.setattr(live.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(live.signal, "getsignal", lambda number: handler[0])
    monkeypatch.setattr(live.signal, "getitimer", lambda which: timer[0])
    monkeypatch.setattr(live.signal, "signal", lambda number, value: handler.__setitem__(0, value))
    monkeypatch.setattr(live.signal, "setitimer", set_timer)
    with pytest.raises(TimeoutError, match="0.02 wall-clock seconds"):
        with live.rpc_deadline(outer):
            with live.rpc_deadline(inner):
                assert timer[0] == pytest.approx((.02, 0.0))
                now[0] += .02
                handler[0](signal.SIGALRM, None)
    assert timer[0] == (0.0, 0.0)
    assert handler[0] is original_handler


def test_nested_rpc_restores_elapsed_parent_timer_and_handler(monkeypatch):
    now = [100.]
    parent_handler = lambda signum, frame: None
    current_handler = [parent_handler]
    timer = [(.08,0.)]
    def set_timer(which,seconds,interval=0.):
        previous = timer[0]
        timer[0] = (seconds,interval)
        return previous
    monkeypatch.setattr(live.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(live.signal,'getsignal',lambda number:current_handler[0])
    monkeypatch.setattr(live.signal,'getitimer',lambda which:timer[0])
    monkeypatch.setattr(live.signal,'signal',lambda number,handler:current_handler.__setitem__(0,handler))
    monkeypatch.setattr(live.signal,'setitimer',set_timer)
    with live.rpc_deadline(.02):
        now[0] += .01
    assert timer[0] == pytest.approx((.07,0.))
    assert current_handler[0] is parent_handler


def test_parent_deadline_is_not_restarted_when_it_expires_inside_inner_scope():
    original_handler = signal.getsignal(signal.SIGALRM)
    with live.rpc_deadline(.02):
        with pytest.raises(TimeoutError):
            with live.rpc_deadline(.08):
                time.sleep(.2)
        # The already-delivered parent alarm must not be restored for another
        # interval if this caller deliberately catches its timeout.
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0.
    assert signal.getsignal(signal.SIGALRM) == original_handler


class Resources:
    names = ['LiquidFuel', 'Oxidizer']

    def __init__(self, amount=100):
        self.quantity = amount

    def amount(self, name):
        return self.quantity

    def density(self, name):
        return 5.0


class Engine:
    def __init__(self, activation_stage, decouple_stage):
        self.part = SimpleNamespace(name=f'engine{activation_stage}', stage=activation_stage,
                                    decouple_stage=decouple_stage)
        self.active = False
        self.has_fuel = True
        self.can_shutdown = True
        self.propellant_names = ['LiquidFuel', 'Oxidizer']

    @property
    def available_thrust(self):
        # kRPC exposes the potential thrust of inactive engines too.
        return 100_000.0 if self.has_fuel else 0.0


class Autopilot:
    def __init__(self):
        self.operations = []
        self._reference_frame = None

    @property
    def reference_frame(self):
        return self._reference_frame

    @reference_frame.setter
    def reference_frame(self, value):
        self._reference_frame = value
        self.operations.append(('reference_frame', value))

    def target_pitch_and_heading(self, pitch, heading):
        self.operations.append(('target', pitch, heading))

    def engage(self):
        self.operations.append(('engage',))

    def disengage(self):
        self.operations.append(('disengage',))


class Control:
    def __init__(self, vessel):
        self.vessel = vessel
        self.current_stage = 4
        self.throttle = 0.0
        self.sas = True
        self.staged = []

    def activate_next_stage(self):
        self.current_stage -= 1
        self.staged.append(self.current_stage)
        if self.current_stage == 3:
            self.vessel.parts.engines[0].active = True
        elif self.current_stage == 2:
            self.vessel.parts.engines[0].active = False
        elif self.current_stage == 1:
            self.vessel.parts.engines[1].active = True


class Vessel:
    def __init__(self):
        c = EnvConfig()
        self.name = 'Fake liquid rocket'
        self.situation = 'VesselSituation.pre_launch'
        self.mass = 34_000.0
        body = SimpleNamespace(gravitational_parameter=c.mu, equatorial_radius=c.radius,
                               non_rotating_reference_frame='inertial', reference_frame='rotating')
        self.orbit = SimpleNamespace(body=body, apoapsis_altitude=100, periapsis_altitude=-599_000)
        self.surface_reference_frame = 'surface'
        self.parts = SimpleNamespace(engines=[Engine(3, 2), Engine(1, -1)])
        self.parts.controlling = SimpleNamespace(vessel=self)
        self.control = Control(self)
        self.auto_pilot = Autopilot()
        self.fuel = {2: Resources(200), -1: Resources(100)}
        self._flight = SimpleNamespace(dynamic_pressure=0.0, surface_altitude=1.0,
                                       latitude=0.0, pitch=90.0, heading=90.0)
        self._position = (c.radius+80, 0, 0)
        self._velocity = (0, 0, self._position[0]*2*math.pi/c.rotation_period)

    def flight(self, reference_frame):
        return self._flight

    def position(self, reference_frame):
        return self._position

    def velocity(self, reference_frame):
        return self._velocity

    @property
    def available_thrust(self):
        return sum(engine.available_thrust for engine in self.parts.engines if engine.active)

    def resources_in_decouple_stage(self, stage, cumulative=False):
        assert cumulative is False
        return self.fuel[stage]


class SpaceCenter:
    def __init__(self, vessel=None):
        self.active_vessel = vessel or Vessel()
        self._ut = 0.0
        self.saved = []

    @property
    def ut(self):
        return self._ut

    def transform_direction(self, vector, source, target):
        assert source == 'surface' and target == 'inertial'
        return vector

    def save(self, name):
        self.saved.append(name)


class Log:
    def __init__(self):
        self.events = []

    def write(self, event, **values):
        self.events.append(dict(event=event, **values))


def test_default_inspection_reads_without_staging_or_autopilot_writes():
    sc = SpaceCenter()
    connection = SimpleNamespace(space_center=sc)
    result = live.inspect(connection, EnvConfig())
    assert result['mode'] == 'inspect_only'
    assert result['suggested_activation_plan'] == [3, 2, 1]
    assert result['suggested_fuel_stages'] == [2, -1]
    assert sc.active_vessel.control.staged == []
    assert sc.active_vessel.control.throttle == 0
    assert sc.active_vessel.control.sas is True
    assert sc.active_vessel.auto_pilot.operations == []
    assert sc.saved == []


def test_live_observations_match_simulator_contract_with_actual_mass():
    c = EnvConfig(randomization=0)
    sim = HeadlessOrbiterEnv(c)
    vessel = Vessel()
    vessel.mass = sim.mass
    vessel.orbit.apoapsis_altitude = sim.telemetry()['apoapsis']
    vessel.orbit.periapsis_altitude = sim.telemetry()['periapsis']
    vessel._flight.dynamic_pressure = sim.telemetry()['dynamic_pressure']
    # Match available thrust while the engine is ignited but throttle is zero.
    vessel.parts.engines[0].active = True
    vessel.parts.engines[0].has_fuel = True
    fuels = live.FuelMonitor(vessel, [2, -1])
    sc = SpaceCenter(vessel)
    state = live.LiveState(sc, vessel, c, fuels)
    state.sample()
    expected = sim.observation()
    # The actual craft's engine can differ from the generic simulator engine.
    expected[13] = vessel.available_thrust/vessel.mass/live.G0
    np.testing.assert_allclose(state.observation(), expected, atol=1e-6)
    vessel.mass *= 0.7
    vessel.fuel[2].quantity *= 0.25
    state.sample()
    assert state.observation()[6] == pytest.approx(0.7)
    assert state.observation()[5] == pytest.approx(0.25)


def test_live_projection_distinguishes_inertial_east_from_north():
    sc = SpaceCenter()
    vessel = sc.active_vessel
    vessel._velocity = (12.0, -8.0, 2400.0)
    state = live.LiveState(sc, vessel, EnvConfig(), None)
    data = state.sample()
    assert data['vertical_speed'] == 12
    assert data['horizontal_speed'] == 2400
    assert data['north_speed'] == -8
    assert data['energy'] == pytest.approx((12**2+8**2+2400**2)/2 - EnvConfig().mu/(EnvConfig().radius+80))


def test_live_attitude_uses_local_horizon_instead_of_body_fixed_axes():
    sc = SpaceCenter()
    vessel = sc.active_vessel
    vessel._flight.pitch, vessel._flight.heading = 15.389, 269.888
    surface = SimpleNamespace(pitch=89.948, heading=180.438)
    vessel.flight = lambda frame: surface if frame == 'surface' else vessel._flight
    fuels = live.FuelMonitor(vessel, [2, -1])
    state = live.LiveState(sc, vessel, EnvConfig(), fuels)
    data = state.sample()
    assert data['pitch'] == 89.948
    assert data['heading'] == 180.438
    assert state.observation()[8] == pytest.approx(math.sin(math.radians(89.948)))


def test_capture_uses_surface_attitude_and_body_fixed_force_vectors():
    from orbiter.capture import flight_snapshot

    sc = SpaceCenter()
    vessel = sc.active_vessel
    vessel.thrust = 0.0
    vessel.control.pitch = vessel.control.yaw = vessel.control.roll = 0.0
    vessel.orbit.time_to_apoapsis = 0.0
    vessel.direction = lambda frame: (1., 0., 0.)
    vessel.angular_velocity = lambda frame: (0., 0., 0.)
    body_fixed = SimpleNamespace(
        pitch=15.389, heading=269.888, roll=-0.414,
        mean_altitude=80., surface_altitude=6., latitude=0., longitude=-74.,
        vertical_speed=0., horizontal_speed=0., dynamic_pressure=123.,
        static_pressure=101000., atmosphere_density=1.2, mach=0.,
        angle_of_attack=0., drag=(1., 2., 3.), lift=(4., 5., 6.))
    surface = SimpleNamespace(pitch=89.948, heading=180.438, roll=-90.)
    vessel.flight = lambda frame: surface if frame == 'surface' else body_fixed
    warnings = []
    sample = flight_snapshot(sc, vessel, vessel.orbit.body, warnings)
    assert not warnings
    assert sample['pitch_deg'] == 89.948
    assert sample['heading_deg'] == 180.438
    assert sample['roll_deg'] == -90.
    assert sample['drag_body_fixed_n'] == [1., 2., 3.]
    assert sample['lift_body_fixed_n'] == [4., 5., 6.]
    assert sample['dynamic_pressure_pa'] == 123.


def test_staging_uses_ignited_engines_and_skips_decoupler_only_stage():
    sc = SpaceCenter()
    vessel = sc.active_vessel
    stager = live.StageController(sc, vessel, [3, 2, 1])
    assert not stager.depleted()  # pre-ignition zero vessel thrust is not depletion
    assert stager.activate() and stager.register_ready()
    assert stager.propulsion_index == 0
    vessel.parts.engines[0].has_fuel = False
    assert vessel.parts.engines[1].available_thrust > 0  # inactive upper stage
    assert stager.depleted()
    assert stager.activate() and not stager.register_ready()  # decouple only
    assert stager.activate() and stager.register_ready()  # ignite upper engine
    assert stager.propulsion_index == 1
    assert not stager.activate()  # never stage beyond the supplied plan
    assert vessel.control.staged == [3, 2, 1]


def test_staging_stops_if_an_external_action_changes_stage():
    sc = SpaceCenter()
    stager = live.StageController(sc, sc.active_vessel, [3, 2, 1])
    sc.active_vessel.control.current_stage = 2
    with pytest.raises(RuntimeError, match='outside this controller'):
        stager.activate()
    assert sc.active_vessel.control.staged == []


def test_staging_follows_original_control_part_when_active_vessel_changes():
    sc = SpaceCenter()
    vessel = sc.active_vessel
    stager = live.StageController(sc, vessel, [3, 2, 1])
    assert stager.activate()
    controlled = Vessel()
    controlled.control.current_stage = 2
    vessel.parts.controlling.vessel = controlled
    assert stager.activate()
    assert stager.vessel is controlled and sc.active_vessel is controlled


def test_solid_rocket_is_rejected_before_controls_change():
    sc = SpaceCenter()
    engine = sc.active_vessel.parts.engines[0]
    engine.can_shutdown = False
    engine.propellant_names = ['SolidFuel']
    with pytest.raises(ValueError, match='SRBs'):
        live.StageController(sc, sc.active_vessel, [3, 2, 1])
    assert sc.active_vessel.control.staged == []


def test_paused_ut_hits_wall_timeout_without_sleeping_in_real_time():
    sc = SpaceCenter()
    now = [0.0]
    with pytest.raises(TimeoutError, match='did not advance'):
        live.wait_for_ut(sc, 2, 0.05, clock=lambda: now[0],
                         sleeper=lambda duration: now.__setitem__(0, now[0]+duration))
    assert now[0] == pytest.approx(0.05)


def test_ut_revert_is_detected():
    values = iter([10.0, 9.0])
    class Reverted:
        @property
        def ut(self):
            return next(values)
    with pytest.raises(RuntimeError, match='backwards'):
        live.wait_for_ut(Reverted(), 11, sleeper=lambda duration: None)


def test_cleanup_attempts_autopilot_even_when_throttle_rpc_fails():
    class BrokenControl:
        @property
        def throttle(self):
            return 1

        @throttle.setter
        def throttle(self, value):
            raise OSError('disconnected throttle RPC')
    ap = Autopilot()
    errors = live.stop_controls([SimpleNamespace(control=BrokenControl(), auto_pilot=ap)])
    assert len(errors) == 1 and 'throttle_zero' in errors[0]
    assert ap.operations == [('disengage',)]


def test_flight_releases_controls_on_paused_game_timeout(monkeypatch):
    sc = SpaceCenter()
    log = Log()
    def paused(*args, **kwargs):
        raise TimeoutError('paused game')
    monkeypatch.setattr(live, 'wait_for_ut', paused)
    with pytest.raises(TimeoutError, match='paused game'):
        live.fly(SimpleNamespace(space_center=sc), EnvConfig(), plan=[3, 2, 1],
                 fuel_stages=[2, -1], scripted=True, log=log)
    assert sc.active_vessel.control.throttle == 0
    assert sc.active_vessel.auto_pilot.operations[-1] == ('disengage',)
    assert sc.saved == []
    assert any(item['event'] == 'controls_released' for item in log.events)


def test_bounded_scripted_flight_records_actual_commands_and_optional_named_save(monkeypatch):
    sc = SpaceCenter()
    log = Log()
    monkeypatch.setattr(live, 'wait_for_ut', lambda center, target, *args, **kwargs: setattr(center, '_ut', target))
    result = live.fly(SimpleNamespace(space_center=sc), EnvConfig(), plan=[3, 2, 1],
                      fuel_stages=[2, -1], scripted=True, log=log, max_steps=2,
                      save_name='Dedicated Orbiter Test')
    assert result['termination_reason'] == 'step_limit' and not result['success']
    assert sc.saved == ['Dedicated Orbiter Test']
    assert sc.active_vessel.control.throttle == 0
    actions = [item for item in log.events if item['event'] == 'action']
    assert len(actions) == 2
    assert all(item['heading_setpoint'] == 90 and item['pitch_setpoint'] == 90 for item in actions)
    assert actions[1]['ut'] > actions[0]['ut']
    # kRPC Engage resets its PID integrators. Repeated target changes must
    # preserve the existing controller instead of restarting it every decision.
    operations = sc.active_vessel.auto_pilot.operations
    assert operations.count(('engage',)) == 1
    assert operations.count(('reference_frame', 'surface')) == 1


def test_flight_engages_new_vessel_autopilot_once_after_separation(monkeypatch):
    sc = SpaceCenter()
    booster_stack = sc.active_vessel
    upper = Vessel()
    upper.parts.engines[0].has_fuel = False
    control_part = booster_stack.parts.controlling
    stage_stack = booster_stack.control.activate_next_stage

    def separate():
        stage_stack()
        if booster_stack.control.current_stage == 2:
            control_part.vessel = upper
            upper.control.current_stage = 2

    booster_stack.control.activate_next_stage = separate

    def advance(center, target, *args, **kwargs):
        center._ut = target
        if target >= 2.5:
            booster_stack.parts.engines[0].has_fuel = False

    monkeypatch.setattr(live, 'wait_for_ut', advance)
    log = Log()
    live.fly(SimpleNamespace(space_center=sc), EnvConfig(), plan=[3, 2, 1],
             fuel_stages=[2, -1], scripted=True, log=log, max_steps=4)
    assert sc.active_vessel is upper
    for vessel in (booster_stack, upper):
        assert vessel.auto_pilot.operations.count(('engage',)) == 1
        assert vessel.auto_pilot.operations.count(('reference_frame', 'surface')) == 1
        assert vessel.auto_pilot.operations[-1] == ('disengage',)
    assert len([entry for entry in upper.auto_pilot.operations if entry[0] == 'target']) == 2


def test_residual_action_decoding_matches_environment():
    env = HeadlessOrbiterEnv(EnvConfig(randomization=0, residual=True))
    action, throttle, pitch = live.decode_action([0, 0], env)
    np.testing.assert_allclose(action, live.expert_action(env))
    assert throttle == 1 and pitch == 90


@pytest.mark.parametrize('algorithm', ['ppo', 'sac', 'tqc'])
def test_model_loader_uses_algorithm_and_environment_metadata(tmp_path, monkeypatch, algorithm):
    from stable_baselines3 import PPO, SAC
    from sb3_contrib import TQC
    config = EnvConfig(target_altitude=85_000, residual=True)
    path = tmp_path/'best.zip'
    path.with_suffix('.metadata.json').write_text(json.dumps(dict(algorithm=algorithm, environment=asdict(config))))
    environment = HeadlessOrbiterEnv(config)
    expected = SimpleNamespace(observation_space=environment.observation_space,
                               action_space=environment.action_space, gamma=config.gamma,
                               orbiter_config=asdict(config), orbiter_algorithm=algorithm)
    loaded = []
    def load(model_path, device):
        loaded.append((model_path, device))
        return expected
    monkeypatch.setattr({'ppo': PPO, 'sac': SAC, 'tqc': TQC}[algorithm], 'load', load)
    model, restored = live.load_controller(path)
    assert model is expected and loaded == [(path, 'cpu')]
    assert restored.target_altitude == 85_000 and restored.residual is True


def test_flight_requires_explicit_activation_and_fuel_mapping():
    assert not live.parse_args([]).fly
    with pytest.raises(SystemExit):
        live.parse_args(['--fly', '--scripted'])


@pytest.mark.parametrize('option', ['--rpc-timeout', '--step-wall-timeout', '--wall-timeout', '--stage-settle'])
@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_live_cli_rejects_nonfinite_wait_limits(option, value):
    with pytest.raises(SystemExit) as error:
        live.parse_args([f'{option}={value}'])
    assert error.value.code == 2


def test_live_success_and_escape_use_bound_orbit_criteria():
    config = EnvConfig()
    data = dict(energy=-1, apoapsis=85_000, periapsis=72_000,
                surface_altitude=80_000, situation='orbiting', time=200)
    assert live.terminal_reason(data, config) == 'orbit_achieved'
    data.update(energy=1, apoapsis=math.inf)
    assert live.terminal_reason(data, config) == 'escape'
@pytest.mark.parametrize('mass_factor,fuel_factor', [(1.25,1.), (1.,1.25)])
def test_learned_policy_rejects_a_different_vehicle(mass_factor, fuel_factor):
    from types import SimpleNamespace
    from orbiter.live import validate_vehicle_mass
    config = EnvConfig()
    expected = config.payload_mass + sum(s.dry_mass+s.fuel_mass for s in config.stages)
    fuels = SimpleNamespace(initial_mass=expected*mass_factor,
        initial_fuel=[s.fuel_mass*fuel_factor for s in config.stages])
    with pytest.raises(ValueError, match='differs'):
        validate_vehicle_mass(config, fuels)


def test_matching_vehicle_mass_is_accepted():
    from types import SimpleNamespace
    from orbiter.live import validate_vehicle_mass
    config = EnvConfig()
    fuels = SimpleNamespace(initial_mass=config.payload_mass+sum(s.dry_mass+s.fuel_mass for s in config.stages),
        initial_fuel=[s.fuel_mass for s in config.stages])
    validate_vehicle_mass(config, fuels)


@pytest.mark.parametrize('reason,expected_exit', [
    ('orbit_achieved', 0), ('impact', 1), ('out_of_fuel', 1),
    ('time_limit', 1), ('step_limit', 1),
])
def test_live_cli_reports_unsuccessful_flight_as_nonzero_exit(tmp_path, monkeypatch, capsys, reason, expected_exit):
    import krpc

    closed = []
    connection = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(krpc, 'connect', lambda **kwargs: connection)
    monkeypatch.setattr(live, 'fly', lambda *args, **kwargs: dict(
        success=reason == 'orbit_achieved', termination_reason=reason))
    output = tmp_path / 'flight.jsonl'
    status = live.main(['--fly', '--scripted', '--stage-plan', '3,2',
                        '--fuel-stages', '2,1', '--output', str(output),
                        '--address', f'legacy-flight-test-{uuid4().hex}.invalid'])
    assert status == expected_exit
    assert json.loads(capsys.readouterr().out)['termination_reason'] == reason
    assert closed == [True]
    assert output.exists()


def test_live_cli_inspection_succeeds_without_flight_success_field(monkeypatch, capsys):
    import krpc

    monkeypatch.setattr(krpc, 'connect', lambda **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(live, 'inspect', lambda *args: dict(mode='inspect_only'))
    assert live.main(['--address', f'legacy-inspection-test-{uuid4().hex}.invalid']) == 0
    assert json.loads(capsys.readouterr().out)['mode'] == 'inspect_only'


def test_live_cli_busy_worker_is_rejected_before_model_or_rpc(tmp_path, monkeypatch, capsys):
    import krpc
    from uuid import uuid4
    from orbiter.ownership import FlightOwnership

    address = f'legacy-flight-test-{uuid4().hex}.invalid'
    monkeypatch.setattr(live, 'load_controller', lambda *args: pytest.fail('Busy CLI loaded model'))
    monkeypatch.setattr(krpc, 'connect', lambda **kwargs: pytest.fail('Busy CLI contacted RPC'))
    with FlightOwnership(address):
        status = live.main(['--fly', '--model', str(tmp_path/'unused.zip'),
                            '--stage-plan', '3,2', '--fuel-stages', '2,1',
                            '--address', address])
    assert status == 1 and 'FlightOwnershipBusy' in capsys.readouterr().out


@pytest.mark.parametrize('outcome', ['success', 'interrupt', 'close_error', 'log_close_error'])
def test_live_cli_holds_ownership_through_models_and_cleanup(tmp_path, monkeypatch, outcome):
    import krpc
    from uuid import uuid4
    from orbiter.ownership import flight_busy

    address = f'legacy-flight-test-{uuid4().hex}.invalid'
    observed = []
    model_path = tmp_path/'model.zip'
    model_path.write_bytes(b'fake')

    def load(*args):
        observed.append(('model', flight_busy(address)))
        return SimpleNamespace(num_timesteps=1), EnvConfig()

    def close():
        observed.append(('close', flight_busy(address)))
        if outcome == 'close_error':
            raise RuntimeError('test connection close')

    def fly(*args, **kwargs):
        observed.append(('fly', flight_busy(address)))
        if outcome == 'interrupt':
            raise KeyboardInterrupt
        return dict(success=True)

    class Recording:
        def __init__(self, path):
            pass
        def write(self, *args, **kwargs):
            pass
        def close(self):
            observed.append(('log_close', flight_busy(address)))
            if outcome == 'log_close_error':
                raise OSError('test log close')

    monkeypatch.setattr(live, 'load_controller', load)
    monkeypatch.setattr(live, 'FlightLog', Recording)
    monkeypatch.setattr(live, 'fly', fly)
    monkeypatch.setattr(krpc, 'connect', lambda **kwargs: SimpleNamespace(close=close))
    args = ['--fly', '--model', str(model_path), '--stage-plan', '3,2',
            '--fuel-stages', '2,1', '--address', address]
    if outcome == 'interrupt':
        with pytest.raises(KeyboardInterrupt):
            live.main(args)
    elif outcome == 'log_close_error':
        with pytest.raises(OSError, match='log close'):
            live.main(args)
    else:
        assert live.main(args) == 0
    assert observed == [('model', True), ('fly', True), ('log_close', True), ('close', True)]
    assert not flight_busy(address)

def test_initial_flight_vessel_rpc_has_a_deadline_before_controls_are_owned(monkeypatch):
    from contextlib import contextmanager

    depth = [0]
    @contextmanager
    def deadline(seconds):
        assert seconds == 7.
        depth[0] += 1
        try:
            yield
        finally:
            depth[0] -= 1

    class UnresponsiveSpaceCenter:
        @property
        def active_vessel(self):
            assert depth[0] > 0, 'Initial vessel read is unbounded'
            raise TimeoutError('initial vessel RPC timed out')

    monkeypatch.setattr(live, 'rpc_deadline', deadline)
    log = Log()
    with pytest.raises(TimeoutError, match='initial vessel RPC timed out'):
        live.fly(SimpleNamespace(space_center=UnresponsiveSpaceCenter()), EnvConfig(),
                 plan=[3, 2], fuel_stages=[2, 1], scripted=True, log=log, rpc_timeout=7.)
    assert not log.events  # no flight or control operation began
    assert depth[0] == 0
