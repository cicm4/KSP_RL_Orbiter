// Thrust-axis pointing and throttle through OnFlyByWire, replacing kRPC's
// AutoPilot. Roll is left free and only damped (kRPC target_roll = NaN).
//
// KSP torque providers apply world torque R * (-input (x) maxTorque) with
// input (pitch, roll, yaw) on the ReferenceTransform axes (x, y, z), so a
// positive control input accelerates rotation about the *negative* local axis.
using System;
using UnityEngine;

namespace OrbiterRL
{
    public sealed class Attitude
    {
        private Vessel vessel;
        private bool pointing;
        private Vector3d target = Vector3d.zero;
        private bool attached;

        public double Throttle;
        /// <summary>Maximum commanded turn rate, rad/s.</summary>
        public double MaxRate = 0.25;
        /// <summary>Time constant of the angle-to-rate profile, s.</summary>
        public double AngleTime = 1.2;
        /// <summary>Time constant of the rate loop, s.</summary>
        public double RateTime = 0.25;
        /// <summary>Integral time of the rate loop, s; holds against steady aerodynamic torque.</summary>
        public double IntegralTime = 1.5;
        private Vector3d integral = Vector3d.zero;

        public Vessel Vessel { get { return vessel; } }

        public void Attach(Vessel next)
        {
            if (vessel == next && attached) return;
            Detach();
            vessel = next;
            vessel.OnFlyByWire += Fly;
            attached = true;
            vessel.ActionGroups.SetGroup(KSPActionGroup.SAS, false);
        }

        /// <summary>Point the thrust axis along a world-frame direction.</summary>
        public void Point(Vector3d worldDirection)
        {
            if (worldDirection.magnitude < 1e-9 || double.IsNaN(worldDirection.x))
                throw new ArgumentException("Pointing direction must be finite and nonzero");
            target = worldDirection.normalized;
            pointing = true;
        }

        public void Release() { pointing = false; integral = Vector3d.zero; }

        /// <summary>Angle between the thrust axis and the target, degrees.</summary>
        public double Error
        {
            get
            {
                if (vessel == null || !pointing) return double.NaN;
                return Vector3d.Angle(Frames.Nose(vessel), target);
            }
        }

        public void Detach()
        {
            if (vessel != null && attached)
            {
                vessel.OnFlyByWire -= Fly;
                vessel.ctrlState.mainThrottle = 0f;
                if (FlightGlobals.ActiveVessel == vessel) FlightInputHandler.state.mainThrottle = 0f;
            }
            attached = false;
            pointing = false;
            integral = Vector3d.zero;
            Throttle = 0;
        }

        private void Fly(FlightCtrlState state)
        {
            float throttle = (float)Math.Max(0.0, Math.Min(1.0, Throttle));
            state.mainThrottle = throttle;
            if (FlightGlobals.ActiveVessel == vessel) FlightInputHandler.state.mainThrottle = throttle;
            if (!pointing || vessel == null || vessel.packed) return;
            Transform reference = vessel.ReferenceTransform;
            Quaternion inverse = Quaternion.Inverse(reference.rotation);
            Vector3 local = inverse * (Vector3)target;
            Vector3 nose = Vector3.up;
            Vector3 axis = Vector3.Cross(nose, local);
            double sine = axis.magnitude, cosine = Vector3.Dot(nose, local);
            double angle = Math.Atan2(sine, cosine);
            Vector3 error;
            if (sine > 1e-6) error = axis / (float)sine * (float)angle;
            else error = cosine < 0 ? new Vector3((float)Math.PI, 0, 0) : Vector3.zero;
            Rigidbody body = vessel.rootPart != null ? vessel.rootPart.rb : null;
            Vector3 omega = body != null ? inverse * body.angularVelocity : Vector3.zero;
            Vector3 torque = AvailableTorque(vessel);
            Vector3 moment = vessel.MOI;
            double dt = TimeWarp.fixedDeltaTime;
            state.pitch = Axis(0, error.x, omega.x, torque.x, moment.x, true, dt);
            state.roll = Axis(1, 0f, omega.y, torque.y, moment.y, false, dt);
            state.yaw = Axis(2, error.z, omega.z, torque.z, moment.z, true, dt);
        }

        private float Axis(int index, double error, double omega, double torque, double moment, bool steer, double dt)
        {
            double authority = moment > 1e-6 ? torque / moment : 0.0;
            if (authority < 1e-4) { integral[index] = 0; return 0f; }
            double desired = 0.0;
            if (steer)
            {
                double magnitude = Math.Abs(error);
                // Linear near the target, braking-limited far from it.
                desired = Math.Sign(error) * Math.Min(MaxRate, Math.Min(magnitude / AngleTime,
                                                                       Math.Sqrt(authority * magnitude)));
            }
            double rateError = desired - omega;
            // PI rate loop; the integral is clamped to the available authority (anti-windup).
            double limit = authority * RateTime * IntegralTime;
            integral[index] = Math.Max(-limit, Math.Min(limit, integral[index] + rateError * dt));
            double acceleration = (rateError + integral[index] / IntegralTime) / RateTime;
            return (float)Math.Max(-1.0, Math.Min(1.0, -acceleration / authority));
        }

        private static Vector3 AvailableTorque(Vessel vessel)
        {
            Vector3 total = Vector3.zero;
            foreach (Part part in vessel.parts)
                foreach (PartModule module in part.Modules)
                {
                    ITorqueProvider provider = module as ITorqueProvider;
                    if (provider == null || !module.isEnabled) continue;
                    Vector3 positive, negative;
                    provider.GetPotentialTorque(out positive, out negative);
                    total += new Vector3((Math.Abs(positive.x) + Math.Abs(negative.x)) / 2,
                                         (Math.Abs(positive.y) + Math.Abs(negative.y)) / 2,
                                         (Math.Abs(positive.z) + Math.Abs(negative.z)) / 2);
                }
            return total;
        }
    }
}
