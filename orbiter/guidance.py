"""Transparent feedback guidance used as a feasibility baseline and BC teacher."""
import math
import numpy as np


def expert_action(env):
    d, c = env.telemetry(), env.config
    alt, vr, vt = d['altitude'], d['vertical_speed'], d['horizontal_speed']
    apo = d['apoapsis']
    target = c.target_altitude + 1500
    # Build horizontal speed throughout the gravity turn, then coast and
    # circularize. The policy learns throttle and an attitude setpoint.
    if alt < 1500:
        pitch, throttle = 90., 1.
    elif alt < 40000 and apo < target:
        pitch = max(8., 88. - (alt-1500)/450.)
        throttle = 1.
    else:
        circular = math.sqrt(c.mu/(c.radius+alt))
        pitch = float(np.clip(0.12*(target-alt) / max(vt, 100.) - 0.22*vr, -10., 25.))
        if apo < target-1000:
            pitch = max(pitch, 10.)
            throttle = 1.
        elif alt < c.target_altitude-c.circularization_margin and vr > 30:
            throttle = 0.
        else:
            # radial cancellation plus tangential acceleration near apoapsis
            pitch = float(np.clip(-0.3*vr, -10., 15.))
            throttle = float(np.clip((circular-vt)/100., 0., 1.))
    return np.array([2*throttle-1, (pitch+10)/50-1], dtype=np.float32)
