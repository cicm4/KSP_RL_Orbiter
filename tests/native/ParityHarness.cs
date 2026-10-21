// Reads one command per stdin line, writes one line of "R"-formatted numbers.
// Built only against orbiter/native/Core for tests/test_native_core.py.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using OrbiterRL.Core;

public static class ParityHarness
{
    private static int cursor;
    private static string[] words;

    private static double D() { return CraftBuilder.Parse(words[cursor++]); }
    private static V3 V() { return new V3(D(), D(), D()); }
    private static string S() { return words[cursor++]; }

    private static string Format(IEnumerable<double> values)
    {
        return string.Join(" ", values.Select(x => x.ToString("R", CultureInfo.InvariantCulture)).ToArray());
    }

    private static IEnumerable<double> Of(V3 v) { return new[] { v.X, v.Y, v.Z }; }

    private static readonly Dictionary<string, Policy> Policies = new Dictionary<string, Policy>();

    private static Policy LoadPolicy(string path)
    {
        Policy policy;
        if (!Policies.TryGetValue(path, out policy)) Policies[path] = policy = Policy.Load(path);
        return policy;
    }

    private static CaptureConfig Config(Policy policy)
    {
        string name = S();
        double mu = D(), radius = D(), atmosphere = D(), soi = D(), altitude = D();
        return CaptureConfig.ForBody(policy, name, mu, radius, atmosphere, soi, altitude);
    }

    private static CaptureTelemetry Telemetry()
    {
        return new CaptureTelemetry
        {
            R = V(), V = V(), Direction = V(), GoalNormal = V(), Mass = D(), InitialMass = D(),
            DryMass = D(), Thrust = D(), Isp = D(), Time = D(),
        };
    }

    private static Body ReadBody()
    {
        return new Body { Mu = D(), Radius = D(), AtmosphereDepth = D(), SphereOfInfluence = D(), SemiMajorAxis = D() };
    }

    public static int Main()
    {
        string line;
        while ((line = Console.ReadLine()) != null)
        {
            words = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            cursor = 1;
            try
            {
                Console.WriteLine(Run(words[0]));
            }
            catch (Exception error)
            {
                Console.WriteLine("error " + error.GetType().Name + ": " + error.Message.Replace('\n', ' '));
            }
        }
        return 0;
    }

    private static string Run(string command)
    {
        switch (command)
        {
            case "act":
            {
                Policy policy = LoadPolicy(S());
                float[] observation = new float[policy.Observations];
                for (int i = 0; i < observation.Length; i++) observation[i] = (float)D();
                return Format(policy.Act(observation));
            }
            case "ascent_obs":
            {
                AscentContract contract = new AscentContract(LoadPolicy(S()));
                AscentTelemetry t = new AscentTelemetry
                {
                    Altitude = D(), VerticalSpeed = D(), HorizontalSpeed = D(), Apoapsis = D(),
                    Periapsis = D(), Energy = D(), DynamicPressure = D(), FuelFraction = D(),
                    MassFraction = D(), Pitch = D(), Throttle = D(), Elapsed = D(), ThrustToWeight = D(),
                    Stage = (int)D(),
                };
                return Format(contract.Observation(t).Select(x => (double)x));
            }
            case "ascent_decode":
            {
                double throttle, pitch;
                AscentContract.Decode(new[] { D(), D() }, out throttle, out pitch);
                return Format(new[] { throttle, pitch });
            }
            case "capture_obs":
            {
                Policy policy = LoadPolicy(S());
                CaptureConfig config = Config(policy);
                return Format(CaptureContract.Observation(config, Telemetry()).Select(x => (double)x));
            }
            case "capture_decode":
            {
                Policy policy = LoadPolicy(S());
                CaptureConfig config = Config(policy);
                CaptureTelemetry telemetry = Telemetry();
                V3 direction;
                double throttle = CaptureContract.Decode(config, telemetry, new[] { D(), D(), D() }, out direction);
                return Format(new[] { throttle }.Concat(Of(direction)));
            }
            case "propagate":
            {
                V3 r = V(), v = V(), pr, pv;
                double dt = D(), mu = D();
                Kepler.Propagate(r, v, dt, mu, out pr, out pv);
                return Format(Of(pr).Concat(Of(pv)));
            }
            case "lambert":
            {
                V3 p1 = V(), p2 = V(), v1, v2;
                double dt = D(), mu = D();
                Kepler.Lambert(p1, p2, dt, mu, D() != 0, out v1, out v2);
                return Format(Of(v1).Concat(Of(v2)));
            }
            case "ephemeris":
            {
                KeplerEphemeris e = new KeplerEphemeris(D(), D(), D(), D(), D(), V(), V());
                V3 r, v;
                e.State(D(), out r, out v);
                return Format(Of(r).Concat(Of(v)));
            }
            case "plan":
            {
                Body source = ReadBody(), target = ReadBody();
                KeplerEphemeris e = new KeplerEphemeris(source.Mu, D(), D(), D(), D(), V(), V());
                V3 r = V(), v = V();
                double epoch = D(), altitude = D();
                TransferPlan plan = Planner.PlanChildTransfer(source, target, e, r, v, epoch, altitude);
                return Format(new[] { plan.DepartureUt, plan.ArrivalUt, plan.ArrivalVinf, plan.CaptureDeltaV,
                                      plan.LongWay ? 1.0 : 0.0 }
                              .Concat(Of(plan.DeltaV)).Concat(Of(plan.AimOffset)));
            }
            case "correction":
            {
                V3 r = V(), v = V();
                V3? delta = Planner.PeriapsisCorrection(r, v, D(), D(), D(), D());
                return delta.HasValue ? Format(Of(delta.Value)) : "none";
            }
            case "incoming":
            {
                V3 r = V(), v = V();
                return Format(new[] { Planner.IncomingRadiusTime(r, v, D(), D(), D()) });
            }
            case "craft":
            {
                string one = S(), science = S(), output = S();
                System.IO.File.WriteAllText(output, CraftBuilder.MakeTransfer(
                    System.IO.File.ReadAllText(one), System.IO.File.ReadAllText(science)), new System.Text.UTF8Encoding(false));
                return "0";
            }
            case "g10":
                return CraftBuilder.G10(D());
            default:
                throw new ArgumentException("unknown command " + command);
        }
    }
}
