import torch
import math
import time
import krpc
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Unbounded, Composite


class KSPState(EnvBase):
    def __init__(self, target_orbit_altitude=80000, step_interval=0.5, max_steps=2000, connection_name='Training', device=torch.device('cpu')):
        # The reason it uses a CPU is that KRPC creates a size 10 tensor which wouldb be too slow to move to the GPU
        super().__init__(device=device)
        self.target_orbit_altitude = target_orbit_altitude
        self.step_interval = step_interval
        self.max_steps = max_steps
        self.connection_name = connection_name
        self._step_count = 0

        n_observations = 10

        #This is the observation space.
        self.observation_spec = Composite(
            observation=Bounded(
                low=-1.0,
                high=2.0,
                shape=(n_observations,),
                dtype=torch.float32
            )
        )

        #Continuous action space.
        self.action_spec = Composite(
            action=Bounded(
                low=torch.tensor([-1.0, -1.0, -1.0]),
                high=torch.tensor([1.0, 1.0, 1.0]),
                shape=(3,),
                dtype=torch.float32
            )
        )

        #reward
        self.reward_spec = Composite(
            reward=Unbounded(shape=(1,), dtype=torch.float32)
        )

        #done, terminated, truncated (end of episode)
        self.done_spec = Composite(
            done=Bounded(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=torch.bool
            ),
            terminated=Bounded(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=torch.bool
            ),
            truncated=Bounded(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=torch.bool
            )
        )

    def _connect(self): #connect to KSP and get the initial state, quicksaves and waits for 1 second
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
            return self.vessel.resources.max("LiquidFuel") #we should also check for oxidizer but that is not implemented yet
        except Exception as e:
            return 1.0

    def _get_fuel_frac(self):
        try:
            initial_fuel = self._initial_fuel
            current_fuel = self.vessel.resources.amount("LiquidFuel")
            return current_fuel / initial_fuel
        except Exception as e:
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
            return len(self.vessel.parts) > 1
        except Exception as e:
            return False
    
    def _reset(self, tensor_dict=None) -> TensorDict:
        if not hasattr(self, 'conn') or self.conn is None:
            self._connect()
        
        self.quickload()
        time.sleep(2.0)

        self.vessel = self.space_center.active_vessel
        self.body = self.vessel.orbit.body
        self._initial_fuel = self._get_fuel_max()
        self._step_count = 0
        self._prev_obs = self._get_obs()

        return TensorDict(
            {
                'observation': self._prev_obs,
                'done': torch.tensor([False], dtype=torch.bool),
                'terminated': torch.tensor([False], dtype=torch.bool),
                'truncated': torch.tensor([False], dtype=torch.bool)
            }
        )
    
    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict['action']

        throttle = (action[0].item() + 1.0 ) / 2.0
        pitch_rate = (action[1].item() * 5.0 )
        heading_rate = (action[2].item() * 5.0 )

        self.vessel.control.throttle = max(min(throttle, 1.0), 0.0)

        ap = self.vessel.auto_pilot
        ap.engage()
        current_pitch = self.vessel.flight(self.body.reference_frame).pitch
        current_heading = self.vessel.flight(self.body.reference_frame).heading
        ap.target_pitch = max(-90, min(90, current_pitch + pitch_rate))
        ap.target_heading = (current_heading + heading_rate) % 360

        t_start = self.space_center.ut
        while self.space_center.ut < t_start + self.step_interval:
            time.sleep(0.01)
        
        self._step_count += 1

        current_obs = self._get_obs()
        intact = self._vessel_intact()
        reward = self._reward_function(self._prev_obs, current_obs, intact)

        terminated = not intact
        truncated = self._step_count >= self.max_steps
        
        if current_obs[2].item() >= 1.0 and current_obs[1].item() >= 1.0:
            terminated = True
            reward += 20.0
        
        self._prev_obs = current_obs

        return TensorDict(
            {
                'observation': current_obs,
                'reward': torch.tensor([reward], dtype=torch.float32),
                'done': torch.tensor([terminated or truncated], dtype=torch.bool),
                'terminated': torch.tensor([terminated], dtype=torch.bool),
                'truncated': torch.tensor([truncated], dtype=torch.bool)
            },
            batch_size=[]
        )

    def _reward_function(self, prev_obs, current_obs, intact) -> float:
        if not intact:
            return -10.0
        else:
            d_apoapsis = current_obs[1].item() - prev_obs[1].item()
            d_periapsis = current_obs[2].item() - prev_obs[2].item()
            return d_apoapsis + ( d_periapsis * 2.0 ) - 0.01 #time penalty is the 0.01 decrease per step
        
    def _set_seed(self, seed: int):
         pass

    def close(self):
         if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.conn = None        
