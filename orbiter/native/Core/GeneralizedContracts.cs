// Observation contracts of the generalized policies, mirroring
// orbiter/generalized_env.py + generalized_aero_env.py (56 floats, any 1-4
// stage sequential liquid rocket) and orbiter/batched_capture.py version 2
// (35 floats, any stock body). Inputs are SI units.
using System;
using System.Collections.Generic;

namespace OrbiterRL.Core
{
    public sealed class StageSpec
    {
        public double DryMass, FuelMass, ThrustVac, IspVac, IspSl;
    }

    /// <summary>Static craft description the ascent policy is conditioned on.</summary>
    public sealed class CraftSpec
    {
        public StageSpec[] Stages;
        public double Payload, DragArea, DragAlphaArea, LiftArea, AttitudeRate, AttitudeTau, Target;

        public double InitialMass
        {
            get
            {
                double total = Payload;
                foreach (StageSpec s in Stages) total += s.DryMass + s.FuelMass;
                return total;
            }
        }

        /// <summary>Ideal vacuum delta-v of the whole stack (the training screen).</summary>
        public double IdealDeltaV
        {
            get
            {
                double total = 0;
                for (int i = 0; i < Stages.Length; i++)
                {
                    double ignition = Payload;
                    for (int j = i; j < Stages.Length; j++) ignition += Stages[j].DryMass + Stages[j].FuelMass;
                    total += Stages[i].IspVac * Units.G0 * Math.Log(ignition / (ignition - Stages[i].FuelMass));
                }
                return total;
            }
        }
    }

    public sealed class GeneralizedAscentTelemetry
    {
        public double Altitude, VerticalSpeed, HorizontalSpeed, Apoapsis, Periapsis, Energy, DynamicPressure;
        public double StageFuel, Mass, Pitch, Throttle, Time, AvailableThrust;
        public int Stage;
    }

    public sealed class GeneralizedAscentContract
    {
        public const int MaxStages = 4;
        public readonly double Mu, Radius, AtmosphereDepth, MaxTime, OrbitTolerance, ControlDt;
        private readonly SequencePolicy policy;

        public GeneralizedAscentContract(SequencePolicy policy)
        {
            if (policy.Kind != "generalized_ascent" || policy.Observations != 56 || policy.Actions != 2)
                throw new ArgumentException("Not a 56-input generalized ascent policy");
            this.policy = policy;
            Mu = policy.Constant("mu"); Radius = policy.Constant("radius");
            AtmosphereDepth = policy.Constant("atmosphere_depth"); MaxTime = policy.Constant("max_time");
            OrbitTolerance = policy.Constant("orbit_tolerance"); ControlDt = policy.Constant("control_dt");
            if ((int)policy.Constant("max_stages") != MaxStages) throw new ArgumentException("Stage capacity mismatch");
        }

        public double Range(string name, bool upper) { return policy.Constant(name + (upper ? "_max" : "_min")); }

        public double Clamp(string name, double value)
        {
            return Math.Max(Range(name, false), Math.Min(Range(name, true), value));
        }

        /// <summary>BatchedAscent._context: 8 floats per stage slot plus 6 globals.</summary>
        public double[] Context(CraftSpec craft)
        {
            double vc = Math.Sqrt(Mu / (Radius + craft.Target));
            double tc = (Radius + craft.Target) / vc;
            double m0 = craft.InitialMass;
            List<double> values = new List<double>();
            for (int index = 0; index < MaxStages; index++)
            {
                if (index >= craft.Stages.Length) { for (int k = 0; k < 8; k++) values.Add(0.0); continue; }
                StageSpec s = craft.Stages[index];
                double ignition = craft.Payload;
                for (int j = index; j < craft.Stages.Length; j++) ignition += craft.Stages[j].DryMass + craft.Stages[j].FuelMass;
                double dv = s.IspVac * Units.G0 * Math.Log(ignition / Math.Max(ignition - s.FuelMass, 1e-20));
                double burn = s.FuelMass * s.IspVac * Units.G0 / Math.Max(s.ThrustVac, 1e-20);
                values.Add(1.0);
                values.Add(s.DryMass / m0);
                values.Add(s.FuelMass / m0);
                values.Add(CaptureContract.Log1p(s.ThrustVac / m0 / Units.G0));
                values.Add(s.IspVac * Units.G0 / vc);
                values.Add(s.IspSl / Math.Max(s.IspVac, 1e-20));
                values.Add(dv / vc);
                values.Add(burn / tc);
            }
            values.Add(craft.Payload / m0);
            values.Add(craft.DragArea / m0 * 1000);
            values.Add(CaptureContract.Log1p(craft.AttitudeRate * tc / 360));
            values.Add(craft.AttitudeTau / tc);
            values.Add(craft.Target / Radius);
            values.Add(craft.Stages.Length / (double)MaxStages);
            return values.ToArray();
        }

        public float[] Observation(CraftSpec craft, GeneralizedAscentTelemetry d)
        {
            double vc = Math.Sqrt(Mu / (Radius + craft.Target));
            double m0 = craft.InitialMass;
            double stageFuel = craft.Stages[d.Stage].FuelMass;
            double apo = double.IsNaN(d.Apoapsis) || double.IsPositiveInfinity(d.Apoapsis) ? 10.0
                       : Math.Max(-10, Math.Min(10, d.Apoapsis / craft.Target));
            double pitch = d.Pitch * Math.PI / 180;
            List<double> values = new List<double>
            {
                d.Altitude / craft.Target, d.VerticalSpeed / vc, d.HorizontalSpeed / vc, apo,
                d.Periapsis / Radius, d.StageFuel / stageFuel, d.Mass / m0,
                d.Stage / (double)Math.Max(craft.Stages.Length - 1, 1),
                Math.Sin(pitch), Math.Cos(pitch), d.Throttle, d.Time / MaxTime,
                CaptureContract.Log1p(d.DynamicPressure / 40000), CaptureContract.Log1p(d.AvailableThrust / d.Mass / Units.G0),
                (craft.Target - d.Altitude) / craft.Target,
                d.Energy / (-Mu / (2 * (Radius + craft.Target))),
            };
            values.AddRange(Context(craft));
            float[] result = new float[values.Count + 2];
            for (int i = 0; i < values.Count; i++) result[i] = (float)Math.Max(-10, Math.Min(10, values[i]));
            result[values.Count] = (float)Math.Max(-10, Math.Min(10, craft.DragAlphaArea / m0 * 1000));
            result[values.Count + 1] = (float)Math.Max(-10, Math.Min(10, craft.LiftArea / m0 * 1000));
            return result;
        }

        /// <summary>BatchedAscent.step success test (periapsis floor 71 km).</summary>
        public bool Success(CraftSpec craft, double energy, double apoapsis, double periapsis)
        {
            return energy < 0 && periapsis >= Math.Max(71000, craft.Target - OrbitTolerance)
                   && apoapsis <= craft.Target + OrbitTolerance;
        }

        /// <summary>Executed action to (throttle, pitch setpoint in degrees).</summary>
        public static void Decode(double[] action, out double throttle, out double pitch)
        {
            AscentContract.Decode(action, out throttle, out pitch);
        }

        /// <summary>
        /// The simulator's attitude servo over dt: rate = clamp((target - pitch)/tau, +-rate).
        /// The KSP attitude controller then tracks this shaped setpoint.
        /// </summary>
        public static double Servo(double pitch, double target, double rate, double tau, double dt)
        {
            double change = Math.Max(-rate, Math.Min(rate, (target - pitch) / tau));
            return pitch + change * dt;
        }
    }

    public sealed class GeneralizedCaptureConfig
    {
        public string BodyName;
        public double Mu, Radius, AtmosphereDepth, SphereOfInfluence, Goal, Safe, Tolerance, V, T, ControlDt, MaxTime;

        /// <summary>BatchedCapture.reset rules for one body and a chosen altitude scale.</summary>
        public static GeneralizedCaptureConfig For(SequencePolicy policy, string body, double mu, double radius,
                                                   double atmosphereDepth, double soi, double altitude)
        {
            if (policy.Kind != "generalized_capture" || policy.Observations != 35 || policy.Actions != 3)
                throw new ArgumentException("Not a 35-input generalized capture policy");
            GeneralizedCaptureConfig c = new GeneralizedCaptureConfig
            {
                BodyName = body, Mu = mu, Radius = radius, AtmosphereDepth = atmosphereDepth, SphereOfInfluence = soi,
            };
            c.Goal = radius + altitude;
            c.Safe = radius + atmosphereDepth + policy.Constant("safe_margin");
            c.Tolerance = Math.Min(policy.Constant("tolerance_cap"), policy.Constant("tolerance_fraction") * (c.Goal - c.Safe));
            c.V = Math.Sqrt(mu / c.Goal);
            c.T = c.Goal / c.V;
            c.ControlDt = policy.Constant("control_dt_over_T") * c.T;
            c.MaxTime = policy.Constant("max_time_over_T") * c.T;
            return c;
        }

        /// <summary>Training altitude band for a body: base * [scale_min, scale_max], above the atmosphere.</summary>
        public static void AltitudeBand(SequencePolicy policy, double radius, double atmosphereDepth,
                                        out double low, out double high)
        {
            double floor = policy.Constant("altitude_floor");
            double baseAltitude = Math.Max(floor, Math.Max(atmosphereDepth + policy.Constant("altitude_atmosphere_margin"),
                                                           policy.Constant("altitude_radius_fraction") * radius));
            low = Math.Max(baseAltitude * policy.Constant("altitude_scale_min"), atmosphereDepth + 10000);
            high = Math.Max(baseAltitude * policy.Constant("altitude_scale_max"), atmosphereDepth + 10000);
        }
    }

    public sealed class GeneralizedCaptureTelemetry
    {
        public V3 R, V, Direction, GoalNormal;
        public double Mass, InitialMass, DryMass, Thrust, Isp, Time;
    }

    public static class GeneralizedCaptureContract
    {
        public const double GravityLimit = 32.0;

        public static double Authority(GeneralizedCaptureConfig c, GeneralizedCaptureTelemetry d)
        {
            if (!(d.Mass > d.DryMass + 1e-9)) return 0.0;
            return Math.Min(d.Thrust / d.Mass, GravityLimit * c.Mu / (c.Goal * c.Goal));
        }

        public static bool InBand(GeneralizedCaptureConfig c, V3 r, V3 v, V3 goalNormal, out double plane)
        {
            OrbitState s = new OrbitState(r, v, c.Mu);
            double cosine = Math.Max(-1, Math.Min(1, V3.Dot(s.H / Math.Max(s.Momentum, 1e-30), goalNormal)));
            plane = Math.Acos(cosine) * 180 / Math.PI;
            return s.Energy < 0 && s.PeriapsisRadius > c.Safe && Math.Abs(s.PeriapsisRadius - c.Goal) <= c.Tolerance
                   && Math.Abs(s.ApoapsisRadius - c.Goal) <= c.Tolerance && plane <= 5;
        }

        public static float[] Observation(GeneralizedCaptureConfig c, GeneralizedCaptureTelemetry d)
        {
            OrbitState s = new OrbitState(d.R, d.V, c.Mu);
            Rtn basis = new Rtn(d.R, d.V);
            V3 localV = basis.ToLocal(d.V);
            V3 normal = basis.ToLocal(d.GoalNormal);
            V3 direction = basis.ToLocal(d.Direction);
            double plane;
            InBand(c, d.R, d.V, d.GoalNormal, out plane);
            double fuel = (d.Mass - d.DryMass) / (d.InitialMass - d.DryMass);
            double dv = d.Isp * Units.G0 * Math.Log(d.Mass / d.DryMass);
            double tta = Math.Max(0.0, s.Radius - s.PeriapsisRadius) / Math.Max(-localV.X, 0.02 * c.V);
            double apo = double.IsInfinity(s.ApoapsisRadius) ? double.PositiveInfinity : s.ApoapsisRadius;
            double[] values =
            {
                s.Radius / c.Goal, localV.X / c.V, localV.Y / c.V, s.Energy / (c.V * c.V), s.Eccentricity,
                s.PeriapsisRadius / c.Goal, double.IsInfinity(apo) ? 10.0 : apo / c.Goal,
                normal.X, normal.Y, normal.Z, direction.X, direction.Y, direction.Z,
                fuel, d.Mass / d.InitialMass, CaptureContract.Log1p(dv / c.V),
                CaptureContract.Log1p(Authority(c, d) * c.T / c.V), CaptureContract.Log1p(d.Isp * Units.G0 / c.V),
                c.Radius / c.Goal, c.Safe / c.Goal, CaptureContract.Log1p(c.SphereOfInfluence / c.Goal),
                Math.Min(tta / c.T, 10.0), d.Time / c.MaxTime, (s.Radius - c.Goal) / c.Goal,
                s.Energy < 0 ? 1.0 : -1.0, s.Momentum / (c.Goal * c.V), c.ControlDt / c.T,
                Math.Log10(c.Mu / 1e10), Math.Log10(c.Radius / 600000), CaptureContract.Log1p(d.Thrust / d.Mass * c.T / c.V),
                d.DryMass / d.InitialMass, c.Tolerance / c.Goal,
                (s.PeriapsisRadius - c.Goal) / c.Tolerance,
                double.IsInfinity(apo) ? 10.0 : (apo - c.Goal) / c.Tolerance, plane / 5,
            };
            float[] result = new float[values.Length];
            for (int i = 0; i < values.Length; i++) result[i] = (float)Math.Max(-10, Math.Min(10, values[i]));
            return result;
        }

        /// <summary>Throttle (fraction of physical thrust) and RTN unit direction.</summary>
        public static double Decode(GeneralizedCaptureConfig c, GeneralizedCaptureTelemetry d, double[] action, out V3 directionRtn)
        {
            V3 command = new V3(action[0], action[1], action[2]);
            double magnitude = command.Norm;
            directionRtn = command / Math.Max(magnitude, 1e-30);
            double physical = d.Thrust / d.Mass;
            return physical > 0 ? Math.Min(1.0, magnitude) * Authority(c, d) / physical : 0.0;
        }
    }
}
