from asyncio import timeout

import torch
import math
import time
import krpc
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Unbounded, Composite


class KSPState(EnvBase):
    def __init__(self, target_orbit_altitude=80000, step_interval=0.5, max_steps=2000, connection_name='Training', device=torch.device('cpu')):
        # The reason it uses a CPU is that KRPC creates a size 10 tensor which would be too slow to move to the GPU
        super().__init__(device=device)
        self.target_orbit_altitude = target_orbit_altitude
        self.step_interval = step_interval
        self.max_steps = max_steps
        self.connection_name = connection_name
        self._step_count = 0
        self._prev_obs = None
        self._orbit_achieved = False

        n_observations = 10

        # This is the observation space.
        self.observation_spec = Composite(
            observation=Bounded(
                low=-2.0,
                high=3.0,
                shape=(n_observations,),
                dtype=torch.float32
            )
        )

        # Continuous action space.
        self.action_spec = Composite(
            action=Bounded(
                low=torch.tensor([-1.0, -1.0, -1.0]),
                high=torch.tensor([1.0, 1.0, 1.0]),
                shape=(3,),
                dtype=torch.float32
            )
        )

        # reward
        self.reward_spec = Composite(
            reward=Unbounded(shape=(1,), dtype=torch.float32)
        )

        # done, terminated, truncated (end of episode)
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
        """Connect to KSP, cache body parameters, quicksave initial state."""
        self.conn = krpc.connect(name=self.connection_name)
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

    def _vessel_intact(self) -> bool:
      try:
          return len(self.vessel.parts.all) > 5
          if (self._step_count > 10
              and obs[0].item() < 0.01
              and abs(obs[4].item()) < 0.01):
            return False
      except Exception:
        return False
        
    def _wait_for_vessel(self, timeout=15.0):
      start = time.time()
      while time.time() - start < timeout:
          try:
              v = self.space_center.active_vessel
              _ = v.name  # force a real RPC call to confirm it's valid
              return v
          except Exception:
              time.sleep(0.5)
      raise RuntimeError("KSP did not return a valid vessel within timeout")

    def _reset(self, tensor_dict=None) -> TensorDict:
        if not hasattr(self, 'conn') or self.conn is None:

            self._connect()
        else:

            self.space_center.quickload()
            time.sleep(4.0)
            self.vessel = self._wait_for_vessel()
            self.vessel = self.space_center.active_vessel
            self.body = self.vessel.orbit.body
            self._initial_fuel = self._get_fuel_max()

        self.vessel.control.activate_next_stage()
        time.sleep(1.0)
        self._step_count = 0
        self._orbit_achieved = False
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

        t_start = self.space_center.ut
        while self.space_center.ut < t_start + self.step_interval:
            time.sleep(0.01)

        self._step_count += 1

        intact = self._vessel_intact()

        if intact:
            current_obs = self._get_obs()
        else:
            current_obs = torch.zeros(10, dtype=torch.float32)

        reward = self._reward_function(self._prev_obs, current_obs, intact)

        terminated = not intact
        truncated = self._step_count >= self.max_steps

        if intact and current_obs[2].item() >= 1.0 and current_obs[1].item() >= 1.0:
            terminated = True
            self._orbit_achieved = True

        if intact and current_obs[0].item() < 0.0 and self._step_count > 10:
            terminated = True
            reward = -10.0

        self._prev_obs = current_obs

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

    def _reward_function(self, prev_obs, current_obs, intact) -> float:
        if not intact:
            return -10.0

        d_apoapsis = current_obs[1].item() - prev_obs[1].item()
        d_periapsis = current_obs[2].item() - prev_obs[2].item()

        reward = d_apoapsis + (d_periapsis * 2.0) - 0.01

        # Milestone: apoapsis reached target
        if current_obs[1].item() >= 1.0 and prev_obs[1].item() < 1.0:
            reward += 5.0

        # Milestone: periapsis reached target (orbit!)
        if current_obs[2].item() >= 1.0 and prev_obs[2].item() < 1.0:
            reward += 20.0

        return reward

    def _set_seed(self, seed: int):
        pass

    def close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.conn = None