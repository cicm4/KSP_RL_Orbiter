// Policy observation/action contracts, mirroring orbiter/env.py
// (build_observation) and orbiter/capture_env.py (build_capture_observation,
// decode_capture_action). Inputs are SI units in a right-handed frame.
using System;
using System.Collections.Generic;

namespace OrbiterRL.Core
{
    public static class Units
    {
        public const double G0 = 9.80665;

        public static float Clip(double value, double limit)
        {
            return (float)Math.Max(-limit, Math.Min(limit, value));
        }
    }

    public sealed class AscentTelemetry
    {
        public double Altitude, VerticalSpeed, HorizontalSpeed, Apoapsis, Periapsis, Energy;
        public double DynamicPressure, FuelFraction, MassFraction, Pitch, Throttle, Elapsed, ThrustToWeight;
        public int Stage;
    }

    public sealed class AscentContract
    {
        public readonly double Mu, Radius, AtmosphereDepth, TargetAltitude, OrbitTolerance, MaxTime, ControlDt, WetMass;
        public readonly double[] FuelMass;
        public readonly int StageCount;

        public AscentContract(Policy policy)
        {
            if (policy.Kind != "ascent" || policy.Observations != 16 || policy.Actions != 2)
                throw new ArgumentException("Not a 16-input ascent policy");
            Mu = policy.Constant("mu"); Radius = policy.Constant("radius");
            AtmosphereDepth = policy.Constant("atmosphere_depth");
            TargetAltitude = policy.Constant("target_altitude");
            OrbitTolerance = policy.Constant("orbit_tolerance");
            MaxTime = policy.Constant("max_time"); ControlDt = policy.Constant("control_dt");
            StageCount = (int)policy.Constant("stage_count");
            WetMass = policy.Constant("wet_mass");
            FuelMass = new double[StageCount];
            for (int i = 0; i < StageCount; i++) FuelMass[i] = policy.Constant("fuel_mass." + i);
        }

        /// <summary>live.terminal_reason; empty while the ascent continues.</summary>
        public string TerminalReason(double energy, double apoapsis, double periapsis, double surfaceAltitude,
                                     bool landedOrSplashed, bool leftGround, double elapsed)
        {
            bool finiteApo = !(double.IsNaN(apoapsis) || double.IsInfinity(apoapsis));
            if (energy < 0 && finiteApo
                && periapsis >= Math.Max(AtmosphereDepth + 1000, TargetAltitude - OrbitTolerance)
                && apoapsis <= TargetAltitude + OrbitTolerance)
                return "orbit_achieved";
            if (energy >= 0) return "escape";
            if (surfaceAltitude < -0.1 || (leftGround && landedOrSplashed)) return "impact";
            if (elapsed >= MaxTime - 1e-8) return "time_limit";
            return "";
        }

        public float[] Observation(AscentTelemetry d)
        {
            double v = Math.Sqrt(Mu / (Radius + TargetAltitude));
            double apo = double.IsInfinity(d.Apoapsis) || double.IsNaN(d.Apoapsis)
                ? 10.0 : Math.Max(-10, Math.Min(10, d.Apoapsis / TargetAltitude));
            double pitch = d.Pitch * Math.PI / 180.0;
            double[] values =
            {
                d.Altitude / TargetAltitude, d.VerticalSpeed / v, d.HorizontalSpeed / v, apo,
                d.Periapsis / Radius, d.FuelFraction, d.MassFraction,
                d.Stage / (double)Math.Max(StageCount - 1, 1),
                Math.Sin(pitch), Math.Cos(pitch), d.Throttle, d.Elapsed / MaxTime,
                d.DynamicPressure / 40000.0, d.ThrustToWeight,
                (TargetAltitude - d.Altitude) / TargetAltitude,
                d.Energy / (-Mu / (2 * (Radius + TargetAltitude))),
            };
            float[] result = new float[values.Length];
            // Python builds float32 first, then clips.
            for (int i = 0; i < values.Length; i++) result[i] = Units.Clip((float)values[i], 10);
            return result;
        }

        /// <summary>live.decode_action: throttle in [0,1], pitch in [-10, 90] degrees.</summary>
        public static void Decode(double[] action, out double throttle, out double pitch)
        {
            double a0 = Math.Max(-1, Math.Min(1, action[0])), a1 = Math.Max(-1, Math.Min(1, action[1]));
            throttle = (a0 + 1) / 2;
            pitch = (a1 + 1) * 50 - 10;
        }
    }

    public struct OrbitState
    {
        public double Radius, Energy, Momentum, Eccentricity, PeriapsisRadius, ApoapsisRadius;
        public V3 H;

        public OrbitState(V3 r, V3 v, double mu)
        {
            Radius = r.Norm;
            Energy = V3.Dot(v, v) / 2 - mu / Radius;
            H = V3.Cross(r, v);
            Momentum = H.Norm;
            Eccentricity = (V3.Cross(v, H) / mu - r / Radius).Norm;
            double p = Momentum * Momentum / mu;
            PeriapsisRadius = p / (1 + Eccentricity);
            ApoapsisRadius = Energy < 0 && Eccentricity < 1 ? p / (1 - Eccentricity) : double.PositiveInfinity;
        }
    }

    public sealed class CaptureConfig
    {
        public string BodyName;
        public double Mu, Radius, AtmosphereDepth, SphereOfInfluence, TargetAltitude;
        public double OrbitTolerance, SurfaceClearance, AtmosphereClearance, PlaneToleranceDeg;
        public double AccelerationGravityLimit, ControlDt, MaxTime, ArrivalRadiusRatio;
        public int SuccessHoldSteps;

        public double GoalRadius { get { return Radius + TargetAltitude; } }
        public double VelocityScale { get { return Math.Sqrt(Mu / GoalRadius); } }
        public double TimeScale { get { return GoalRadius / VelocityScale; } }
        public double SafeRadius { get { return Radius + Math.Max(SurfaceClearance, AtmosphereDepth + AtmosphereClearance); } }

        /// <summary>Live override set used by mission_live.capture_config.</summary>
        public static CaptureConfig ForBody(Policy policy, string body, double mu, double radius,
                                            double atmosphereDepth, double soi, double targetAltitude)
        {
            if (policy.Kind != "capture" || policy.Observations != 27 || policy.Actions != 3)
                throw new ArgumentException("Not a 27-input capture policy");
            return new CaptureConfig
            {
                BodyName = body, Mu = mu, Radius = radius, AtmosphereDepth = atmosphereDepth,
                SphereOfInfluence = soi, TargetAltitude = targetAltitude, OrbitTolerance = 5000.0,
                SurfaceClearance = policy.Constant("surface_clearance"),
                AtmosphereClearance = policy.Constant("atmosphere_clearance"),
                PlaneToleranceDeg = policy.Constant("plane_tolerance_deg"),
                AccelerationGravityLimit = policy.Constant("acceleration_gravity_limit"),
                ControlDt = policy.Constant("control_dt"), MaxTime = policy.Constant("max_time"),
                ArrivalRadiusRatio = policy.Constant("arrival_radius_ratio"),
                SuccessHoldSteps = (int)policy.Constant("success_hold_steps"),
            };
        }
    }

    public sealed class CaptureTelemetry
    {
        public V3 R, V, Direction, GoalNormal;
        public double Mass, InitialMass, DryMass, Thrust, Isp, Time;
    }

    public static class CaptureContract
    {
        public static double ControlAcceleration(CaptureConfig c, CaptureTelemetry d)
        {
            if (double.IsNaN(d.Mass) || double.IsNaN(d.Thrust) || double.IsNaN(d.DryMass) ||
                d.Mass <= 0 || d.DryMass <= 0 || d.Thrust < 0 || d.Mass < d.DryMass)
                throw new ArgumentException("invalid capture actuator telemetry");
            double physical = d.Mass > d.DryMass + 1e-9 ? d.Thrust / d.Mass : 0.0;
            return Math.Min(physical, c.AccelerationGravityLimit * c.Mu / (c.GoalRadius * c.GoalRadius));
        }

        public static float[] Observation(CaptureConfig c, CaptureTelemetry d)
        {
            OrbitState state = new OrbitState(d.R, d.V, c.Mu);
            Rtn basis = new Rtn(d.R, d.V);
            V3 localV = basis.ToLocal(d.V);
            V3 goal = basis.ToLocal(d.GoalNormal.Unit());
            V3 direction = basis.ToLocal(d.Direction / Math.Max(d.Direction.Norm, 1e-12));
            double L = c.GoalRadius, V = c.VelocityScale, T = c.TimeScale;
            double fuel = Math.Max(0.0, d.Mass - d.DryMass);
            double initialFuel = Math.Max(1e-12, d.InitialMass - d.DryMass);
            double availableDv = d.Isp * Units.G0 * Math.Log(Math.Max(d.Mass / d.DryMass, 1.0));
            double approaching = Math.Max(0.0, -localV.X);
            double tta = Math.Max(0.0, state.Radius - state.PeriapsisRadius) / Math.Max(approaching, 0.02 * V);
            double[] values =
            {
                state.Radius / L, localV.X / V, localV.Y / V, state.Energy / (V * V),
                state.Eccentricity, state.PeriapsisRadius / L,
                double.IsInfinity(state.ApoapsisRadius) ? 10.0 : state.ApoapsisRadius / L,
                goal.X, goal.Y, goal.Z, direction.X, direction.Y, direction.Z,
                fuel / initialFuel, d.Mass / d.InitialMass, availableDv / V,
                Log1p(ControlAcceleration(c, d) * T / V), Log1p(d.Isp * Units.G0 / V),
                c.Radius / L, c.SafeRadius / L, Log1p(c.SphereOfInfluence / L),
                Math.Min(tta / T, 10.0), d.Time / c.MaxTime, (state.Radius - L) / L,
                state.Energy < 0 ? 1.0 : -1.0, state.Momentum / (L * V), c.ControlDt / T,
            };
            float[] result = new float[values.Length];
            for (int i = 0; i < values.Length; i++) result[i] = Units.Clip((float)values[i], 10);
            return result;
        }

        /// <summary>Throttle and RTN unit direction; zero action gives zero throttle.</summary>
        public static double Decode(CaptureConfig c, CaptureTelemetry d, double[] action, out V3 directionRtn)
        {
            V3 command = new V3(action[0], action[1], action[2]);
            double magnitude = command.Norm;
            if (!command.Finite) throw new ArgumentException("capture action must be finite");
            if (magnitude == 0) { directionRtn = V3.Zero; return 0.0; }
            double authority = ControlAcceleration(c, d);
            double physical = d.Thrust / d.Mass;
            directionRtn = command / magnitude;
            return physical > 0 ? Math.Min(1.0, magnitude) * authority / physical : 0.0;
        }

        /// <summary>mission_live.orbit_failures on KSP's apse altitudes; empty when in band.</summary>
        public static List<string> OrbitFailures(CaptureConfig c, string body, double apoapsis, double periapsis,
                                                 double eccentricity, double semiMajorAxis)
        {
            List<string> failures = new List<string>();
            if (body != c.BodyName) { failures.Add("wrong_body"); return failures; }
            double energy = semiMajorAxis != 0 ? -c.Mu / (2 * semiMajorAxis) : double.NaN;
            foreach (double value in new[] { apoapsis, periapsis, eccentricity, energy })
                if (double.IsNaN(value) || double.IsInfinity(value)) { failures.Add("invalid_telemetry"); return failures; }
            if (energy >= 0 || !(eccentricity >= 0 && eccentricity < 1)) failures.Add("not_bound");
            if (!(c.TargetAltitude - c.OrbitTolerance <= periapsis && periapsis <= apoapsis
                  && apoapsis <= c.TargetAltitude + c.OrbitTolerance))
                failures.Add("outside_target_band");
            if (periapsis + c.Radius <= c.SafeRadius) failures.Add("unsafe_periapsis");
            if (apoapsis + c.Radius >= c.SphereOfInfluence) failures.Add("outside_soi");
            return failures;
        }

        /// <summary>Angle between the orbit normal and the goal plane normal, degrees.</summary>
        public static double PlaneError(V3 r, V3 v, V3 goalNormal)
        {
            V3 h = V3.Cross(r, v);
            double cosine = V3.Dot(h / Math.Max(h.Norm, 1e-12), goalNormal.Unit());
            return Math.Acos(Math.Max(-1.0, Math.Min(1.0, cosine))) * 180.0 / Math.PI;
        }

        /// <summary>log1p with full precision for small arguments (Mono 4.x lacks Math.Log1p).</summary>
        public static double Log1p(double x)
        {
            if (Math.Abs(x) > 1e-4) return Math.Log(1.0 + x);
            return x - x * x / 2 + x * x * x / 3;
        }
    }
}
