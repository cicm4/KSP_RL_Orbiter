import math
import socket
import time

import krpc
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Unbounded, Composite
from torchrl.envs import EnvBase


class KSPState(EnvBase):
    def __init__(self, target_orbit_altitude=80000, step_interval=0.5, max_steps=2000, connection_name='Training', device=torch.device('cpu')):
        # Keep the environment on CPU; kRPC calls dominate the cost here.
        super().__init__(device=device)
        self.target_orbit_altitude = target_orbit_altitude
        self.step_interval = step_interval
        self.max_steps = max_steps
        self.connection_name = connection_name
        self._step_count = 0
        self._prev_obs = None
        self._orbit_achieved = False
        self._seed = None
        self._episode_start_ut = None
        self._episode_max_altitude = 0.0
        self._episode_max_speed = 0.0
        self._last_step_info = {}
        self._last_termination_reason = ""

        n_observations = 10

        self.observation_spec = Composite(
            observation=Bounded(
                low=-2.0,
                high=3.0,
                shape=(n_observations,),
                dtype=torch.float32
            )
        )

        self.action_spec = Composite(
            action=Bounded(
                low=torch.tensor([-1.0, -1.0, -1.0]),
                high=torch.tensor([1.0, 1.0, 1.0]),
                shape=(3,),
                dtype=torch.float32
            )
        )

        self.reward_spec = Composite(
            reward=Unbounded(shape=(1,), dtype=torch.float32)
        )

        self.done_spec = Composite(
            done=Bounded(
                low=0,
                high=1,
                shape=(1,),
                dtype=torch.bool
            ),
            terminated=Bounded(
                low=0,
                high=1,
                shape=(1,),
                dtype=torch.bool
            ),
            truncated=Bounded(
                low=0,
                high=1,
                shape=(1,),
                dtype=torch.bool
            )
        )

    def _connect(self):
        # Save a single reset point and quickload back to it each episode.
        try:
            self.conn = krpc.connect(name=self.connection_name)
        except ConnectionRefusedError as exc:
            raise RuntimeError("Could not connect to kRPC. Start KSP and enable the server.") from exc
        except socket.error as exc:
            raise RuntimeError("kRPC connection failed.") from exc
        self.space_center = self.conn.space_center
        self.vessel = self.space_center.active_vessel
        self.body = self.vessel.orbit.body
        self.target_v = math.sqrt(
            self.body.gravitational_parameter /
            (self.body.equatorial_radius + self.target_orbit_altitude)
        )
        self.atmo_depth = self.body.atmosphere_depth if self.body.has_atmosphere else self.target_orbit_altitude
        self._initial_fuel = self._get_fuel_max()
        self.space_center.quicksave()
        time.sleep(1.0)

    def _get_fuel_max(self):
        try:
            return self.vessel.resources.max("LiquidFuel")
        except Exception:
            return 1.0

    def _get_fuel_frac(self):
        try:
            initial_fuel = self._initial_fuel
            if initial_fuel <= 0:
                return 1.0
            current_fuel = self.vessel.resources.amount("LiquidFuel")
            return current_fuel / initial_fuel
        except Exception:
            return 1.0

    def _get_obs(self) -> torch.Tensor:
        flight = self.vessel.flight(self.body.reference_frame)
        orbit = self.vessel.orbit

        # Normalize telemetry around the target orbit to keep scales stable.
        return torch.tensor([
            flight.mean_altitude / self.target_orbit_altitude,
            orbit.apoapsis_altitude / self.target_orbit_altitude,
            orbit.periapsis_altitude / self.target_orbit_altitude,
            orbit.speed / self.target_v,
            flight.vertical_speed / self.target_v,
            flight.pitch / 90.0,
            flight.heading / 180.0,
            self._get_fuel_frac(),
            self.vessel.control.throttle,
            min(orbit.time_to_apoapsis / 120.0, 2.0),
        ], dtype=torch.float32)

    def _get_vehicle_metrics(self) -> dict:
        metrics = {
            "fuel_remaining": math.nan,
            "fuel_remaining_frac": math.nan,
            "apoapsis": math.nan,
            "periapsis": math.nan,
            "altitude": math.nan,
            "surface_altitude": math.nan,
            "speed": math.nan,
            "vertical_speed": math.nan,
            "situation": "",
        }
        try:
            flight = self.vessel.flight(self.body.reference_frame)
            orbit = self.vessel.orbit
            metrics.update(
                {
                    "fuel_remaining": float(self.vessel.resources.amount("LiquidFuel")),
                    "fuel_remaining_frac": float(self._get_fuel_frac()),
                    "apoapsis": float(orbit.apoapsis_altitude),
                    "periapsis": float(orbit.periapsis_altitude),
                    "altitude": float(flight.mean_altitude),
                    "surface_altitude": float(flight.surface_altitude),
                    "speed": float(orbit.speed),
                    "vertical_speed": float(flight.vertical_speed),
                    "situation": str(self.vessel.situation).lower(),
                }
            )
        except Exception:
            pass
        return metrics

    def _vessel_intact(self) -> bool:
        try:
            # Part count is a coarse crash proxy, but it is cheap to query.
            return len(self.vessel.parts.all) > 5
        except Exception:
            return False
        
    def _wait_for_vessel(self, timeout=15.0):
        start = time.time()
        while time.time() - start < timeout:
            try:
                v = self.space_center.active_vessel
                _ = v.name  # Force an RPC read to confirm the handle is valid.
                return v
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("KSP did not return a valid vessel within timeout")

    def _reset(self, tensor_dict=None) -> TensorDict:
        if not hasattr(self, 'conn') or self.conn is None:
            self._connect()
        else:
            # Reload the same launch state so every episode starts from one setup.
            self.space_center.quickload()
            time.sleep(4.0)
            self.vessel = self._wait_for_vessel()
            self.body = self.vessel.orbit.body
            self._initial_fuel = self._get_fuel_max()

        self.vessel.control.activate_next_stage()
        time.sleep(1.0)
        self._step_count = 0
        self._orbit_achieved = False
        self._episode_start_ut = self.space_center.ut
        self._episode_max_altitude = 0.0
        self._episode_max_speed = 0.0
        self._last_termination_reason = ""
        self._last_step_info = {}
        self._prev_obs = self._get_obs()

        return TensorDict(
            {
                'observation': self._prev_obs,
                'done': torch.tensor([False], dtype=torch.bool),
                'terminated': torch.tensor([False], dtype=torch.bool),
                'truncated': torch.tensor([False], dtype=torch.bool),
            },
            batch_size=[],
        )

    def _reward_breakdown(self, prev_obs, current_obs, intact) -> tuple[float, dict]:
        components = {
            "reward_potential": 0.0,
            "reward_alive": 0.0,
            "reward_overshoot": 0.0,
            "reward_orbit_bonus": 0.0,
            "reward_failure_penalty": 0.0,
            "reward_ground_penalty": 0.0,
            "reward_descent_thrust_penalty": 0.0,
        }

        if not intact:
            components["reward_failure_penalty"] = -40.0
            return -40.0, components

        components["reward_potential"] = (
            self._potential(current_obs) - self._potential(prev_obs)
        )
        # Keep this tiny so the agent prefers progress, not mere survival.
        components["reward_alive"] = 0.001

        if current_obs[1].item() >= 1.6:
            components["reward_overshoot"] = -0.05

        # Penalize burning low in the atmosphere while descending, since it can
        # inflate apoapsis without representing useful orbital progress.
        current_altitude_m = max(current_obs[0].item(), 0.0) * self.target_orbit_altitude
        if (
            self._step_count > 10
            and current_altitude_m < 10_000.0
            and current_obs[4].item() < 0.0
            and current_obs[8].item() > 0.1
        ):
            descent_ratio = min(abs(current_obs[4].item()), 1.0)
            throttle_ratio = min(max(current_obs[8].item(), 0.0), 1.0)
            altitude_ratio = 1.0 - (current_altitude_m / 10_000.0)
            components["reward_descent_thrust_penalty"] = -2.0 * (
                0.5 + altitude_ratio
            ) * descent_ratio * throttle_ratio

        if self._orbit_achieved:
            components["reward_orbit_bonus"] = 100.0

        reward = sum(components.values())
        return reward, components

    def _build_step_info(
        self,
        reward: float,
        reward_components: dict,
        metrics: dict,
        terminated: bool,
        truncated: bool,
        termination_reason: str,
    ) -> dict:
        orbit_time_s = math.nan
        if self._orbit_achieved and self._episode_start_ut is not None:
            orbit_time_s = float(self.space_center.ut - self._episode_start_ut)

        step_info = {
            "step_in_episode": int(self._step_count),
            "success": bool(self._orbit_achieved),
            "orbit_time_s": orbit_time_s,
            "reward_total": float(reward),
            "termination_reason": termination_reason,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "max_altitude": float(self._episode_max_altitude),
            "max_speed": float(self._episode_max_speed),
        }
        step_info.update(reward_components)
        step_info.update(metrics)
        return step_info

    def _ground_contact_detected(self, metrics: dict) -> bool:
        if self._step_count <= 10:
            return False

        situation = metrics.get("situation", "")
        if "landed" in situation or "splashed" in situation:
            return True

        surface_altitude = metrics.get("surface_altitude", math.nan)
        vertical_speed = metrics.get("vertical_speed", math.nan)
        if math.isfinite(surface_altitude) and surface_altitude <= 1.0:
            if not math.isfinite(vertical_speed) or vertical_speed <= 0.0:
                return True

        return False

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict['action']

        throttle = (action[0].item() + 1.0) / 2.0
        pitch_input = action[1].item()
        heading_input = action[2].item()

        self.vessel.control.throttle = max(min(throttle, 1.0), 0.0)
        self.vessel.control.pitch = pitch_input
        self.vessel.control.yaw = heading_input

        self.vessel.auto_pilot.disengage()
        self.vessel.control.sas = True

        # Hold the action for a fixed control interval before reading the next state.
        t_start = self.space_center.ut
        while self.space_center.ut < t_start + self.step_interval:
            time.sleep(0.01)

        self._step_count += 1

        intact = self._vessel_intact()
        termination_reason = ""

        if intact:
            current_obs = self._get_obs()
            metrics = self._get_vehicle_metrics()
        else:
            current_obs = torch.zeros(10, dtype=torch.float32)
            metrics = self._get_vehicle_metrics()

        altitude = metrics.get("altitude", math.nan)
        speed = metrics.get("speed", math.nan)
        if math.isfinite(altitude):
            self._episode_max_altitude = max(self._episode_max_altitude, altitude)
        if math.isfinite(speed):
            self._episode_max_speed = max(self._episode_max_speed, speed)

        reward, reward_components = self._reward_breakdown(
            self._prev_obs, current_obs, intact
        )

        if intact and current_obs[2].item() >= 1.0 and current_obs[1].item() >= 1.0:
            self._orbit_achieved = True
            reward, reward_components = self._reward_breakdown(
                self._prev_obs, current_obs, intact
            )

        terminated = False
        truncated = False
        if not intact:
            terminated = True
            termination_reason = "vessel_destroyed"
        elif self._orbit_achieved:
            terminated = True
            termination_reason = "orbit_achieved"
        elif self._ground_contact_detected(metrics):
            terminated = True
            termination_reason = "below_ground"
            reward_components["reward_ground_penalty"] = -40.0 - reward
            reward = -40.0
        elif self._step_count >= self.max_steps:
            truncated = True
            termination_reason = "max_steps"

        self._prev_obs = current_obs
        self._last_termination_reason = termination_reason
        self._last_step_info = self._build_step_info(
            reward=reward,
            reward_components=reward_components,
            metrics=metrics,
            terminated=terminated,
            truncated=truncated,
            termination_reason=termination_reason,
        )

        return TensorDict(
            {
                'observation': current_obs,
                'reward': torch.tensor([reward], dtype=torch.float32),
                'done': torch.tensor([terminated or truncated], dtype=torch.bool),
                'terminated': torch.tensor([terminated], dtype=torch.bool),
                'truncated': torch.tensor([truncated], dtype=torch.bool),
            },
            batch_size=[],
        )
    
    def _potential(self, obs):
        altitude = min(max(obs[0].item(), 0.0), 0.35)
        apo = min(max(obs[1].item(), 0.0), 1.1)
        peri = min(max(obs[2].item(), 0.0), 1.0)

        return (
            2.0 * altitude
            + 8.0 * apo
            + 28.0 * peri
        )

    def _reward_function(self, prev_obs, current_obs, intact) -> float:
        reward, _ = self._reward_breakdown(prev_obs, current_obs, intact)
        return reward

    def _set_seed(self, seed: int):
        self._seed = seed
        torch.manual_seed(seed)

    def get_step_info(self) -> dict:
        return dict(self._last_step_info)

    def close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.conn = None
