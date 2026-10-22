// In-process port of orbiter/mission_live.py fly_mission: learned ascent from
// the pad, Lambert transfer verified against KSP's own patched conics, node
// burns, learned capture, and a full unpowered orbit check. No kRPC, no Python.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using KSP.UI.Screens;
using OrbiterRL.Core;
using UnityEngine;

namespace OrbiterRL
{
    public sealed class MissionFailure : Exception
    {
        public readonly string Reason;
        public MissionFailure(string reason, string message) : base(message) { Reason = reason; }
    }

    public sealed partial class Mission
    {
        private const int ParkingWindows = 8;
        private const double DeltaVReserve = 100.0, StageSettle = 0.5;
        private const double NodeTolerance = 0.1, AlignmentTolerance = 3.0, AlignmentTimeout = 60.0;
        private const double NodeTick = 0.1, WarpMargin = 10.0, PeriapsisTolerance = 1000.0;

        private readonly Policy ascentPolicy, capturePolicy;
        private readonly AscentContract ascent;
        private readonly CelestialBody target;
        private readonly double targetAltitude;
        private readonly MissionPlan plan;
        private readonly MissionPolicies policies;
        private readonly MissionLog log;
        private readonly Attitude attitude = new Attitude();
        private readonly List<ManeuverNode> ownedNodes = new List<ManeuverNode>();
        private Vessel vessel;
        private Part controlPart;

        public string Phase { get; private set; }
        public string Detail { get; private set; }
        public bool Succeeded { get; private set; }

        public Mission(MissionPolicies policies, MissionPlan plan, MissionLog log)
        {
            this.policies = policies;
            this.plan = plan;
            ascentPolicy = policies.Ascent;
            capturePolicy = policies.Capture;
            ascent = new AscentContract(ascentPolicy);
            generalAscent = new GeneralizedAscentContract(policies.GeneralAscent);
            target = plan.Target;
            targetAltitude = plan.TargetAltitude;
            this.log = log;
            Phase = "preflight";
            Detail = "Checking the craft";
        }

        private static double Now { get { return Planetarium.GetUniversalTime(); } }

        private void Write(string name, params object[] pairs)
        {
            if (log != null) log.Write(name, pairs);
        }

        /// <summary>Launchpad readiness without side effects: null when ready, else the reason.</summary>
        public string CheckPreflight(Vessel candidate)
        {
            vessel = candidate;
            try
            {
                if (vessel == null) return "No active vessel";
                Preflight();
                return null;
            }
            catch (MissionFailure failure) { return failure.Message; }
            catch (Exception error) { return "Preflight error: " + error.Message; }
            finally { vessel = null; }
        }

        /// <summary>Raised on every phase change (the autotest saves resumable checkpoints).</summary>
        public event Action<string> PhaseChanged;

        private void SetPhase(string phase, string detail)
        {
            Phase = phase;
            Detail = detail;
            if (PhaseChanged != null) PhaseChanged(phase);
            Write("phase", "phase", phase, "detail", detail, "ut", Now);
        }

        // ------------------------------------------------------------------ root

        public IEnumerator Run()
        {
            try
            {
                vessel = FlightGlobals.ActiveVessel;
                if (vessel == null) throw new MissionFailure("no_vessel", "No active vessel");
                Preflight();
                if (ResumeFrom != null)
                {
                    yield return GeneralizedResume();
                    yield break;
                }
                if (!Validated)
                {
                    yield return GeneralizedRun();
                    yield break;
                }
                yield return Ascent();
                yield return Transfer();
                yield return CoastToTarget();
                CaptureConfig config = CaptureConfig.ForBody(capturePolicy, target.bodyName, target.gravParameter,
                    target.Radius, target.atmosphere ? target.atmosphereDepth : 0.0, target.sphereOfInfluence,
                    targetAltitude);
                yield return CorrectPeriapsis(config.Mu, config.Radius, config.GoalRadius);
                yield return CoastToCapture(config.Mu, config.ArrivalRadiusRatio * config.GoalRadius);
                yield return Capture(config);
                yield return VerifyOrbit(o => CaptureContract.OrbitFailures(config, o.referenceBody.bodyName, o.ApA, o.PeA,
                                                                            o.eccentricity, o.semiMajorAxis));
                Succeeded = true;
                SetPhase("succeeded", string.Format(CultureInfo.InvariantCulture,
                    "Stable {0:0.0} x {1:0.0} km orbit of {2} verified for one full revolution",
                    vessel.orbit.PeA / 1000, vessel.orbit.ApA / 1000, target.bodyName));
            }
            finally
            {
                ReleaseAll();
            }
        }

        public void ReleaseAll()
        {
            attitude.Detach();
            foreach (ManeuverNode node in ownedNodes)
                if (node != null && node.solver != null && node.solver.maneuverNodes.Contains(node)) node.RemoveSelf();
            ownedNodes.Clear();
            if (TimeWarp.CurrentRateIndex != 0) TimeWarp.SetRate(0, true);
            Write("controls_released", "ut", Now);
        }

        // ------------------------------------------------------------- preflight

        private void Require(bool condition, string reason, string message)
        {
            if (!condition) throw new MissionFailure(reason, message);
        }

        /// <summary>True when the validated single-craft policies fly this mission.</summary>
        public bool Validated { get; private set; }

        private void Preflight()
        {
            if (log != null) SetPhase("preflight", "Checking the rocket");
            if (vessel.situation != Vessel.Situations.PRELAUNCH && target != null)
            {
                ResumePreflight();
                return;
            }
            ResumeFrom = null;
            Require(vessel.situation == Vessel.Situations.PRELAUNCH, "not_on_pad",
                    "Start from the launchpad: the learned ascent begins at PRELAUNCH");
            Require(vessel.IsControllable, "no_control", "The vessel has no crew or probe control");
            CelestialBody body = vessel.mainBody;
            Require(Math.Abs(body.gravParameter / ascent.Mu - 1) <= 0.01 && Math.Abs(body.Radius / ascent.Radius - 1) <= 0.01,
                    "wrong_body", "The learned ascent was trained for Kerbin");
            Require(Math.Abs(vessel.latitude) <= 1.0, "not_equatorial", "The heading-90 ascent needs an equatorial pad");
            if (target != null)
            {
                Require(target.referenceBody == body && target != body, "unsupported_target",
                        target.bodyName + " does not orbit " + body.bodyName);
                Require(vessel.patchedConicSolver != null, "no_patched_conics",
                        "Maneuver nodes are unavailable (upgrade the Tracking Station or Mission Control)");
                Require(vessel.patchedConicSolver.maneuverNodes.Count == 0, "existing_nodes",
                        "Remove existing maneuver nodes first");
            }
            // The validated pair flies only the trained rocket to the Mun or Minmus;
            // every other rocket or destination uses the generalized policies.
            Validated = target != null && TrainedCraft() == null;
            if (Validated) TrainedCraftPreflight();
            else GeneralizedPreflight();
        }

        /// <summary>Null when the vessel matches the trained craft's parts and engines, else why not.</summary>
        private string TrainedCraft()
        {
            string parts = VesselInfo.Describe(vessel.parts.Select(p => p.partInfo.name));
            if (parts != Header("vehicle.parts")) return "different parts";
            return null;
        }

        private void TrainedCraftPreflight()
        {
            string parts = VesselInfo.Describe(vessel.parts.Select(p => p.partInfo.name));
            Require(parts == Header("vehicle.parts"), "wrong_craft",
                    "This autopilot is trained for the '" + Header("vehicle.name") + "' craft only");
            string[] expected = Header("vehicle.engines").Split(';');
            List<ModuleEngines> engines = VesselInfo.Engines(vessel);
            Require(engines.Count == expected.Length, "wrong_craft", "Unexpected engine count");
            foreach (string signature in expected)
            {
                string[] fields = signature.Split(':');
                Require(engines.Any(e => e.part.partInfo.name == fields[0] && e.part.inverseStage == int.Parse(fields[1])
                                         && VesselInfo.DecoupleStage(e.part) == int.Parse(fields[2])
                                         && !e.EngineIgnited && e.allowShutdown && e.thrustPercentage >= 99.9f),
                        "wrong_craft", "Engine " + fields[0] + " is not in its trained stage or is limited");
            }
            int[] plan = Ints(Header("vehicle.stage_plan"));
            Require(vessel.currentStage == plan[0] + 1, "wrong_stage", "Staging was changed before launch");
            Require(Math.Abs(VesselInfo.Mass(vessel) / ascent.WetMass - 1) <= 0.1, "wrong_mass",
                    "Wet mass differs from the trained craft by over 10%");
            int[] fuel = Ints(Header("vehicle.fuel_stages"));
            for (int i = 0; i < fuel.Length; i++)
                Require(Math.Abs(VesselInfo.StagePropellant(vessel, fuel[i]) / ascent.FuelMass[i] - 1) <= 0.1,
                        "wrong_fuel", "Stage " + i + " propellant differs from the trained craft by over 10%");
            Write("preflight_passed", "mode", "validated", "vessel", vessel.vesselName, "mass_kg", VesselInfo.Mass(vessel),
                      "target", target.bodyName, "target_altitude_m", targetAltitude,
                      "ascent_policy_sha256", ascentPolicy.Sha256, "capture_policy_sha256", capturePolicy.Sha256);
        }

        private string Header(string key)
        {
            string value;
            if (!ascentPolicy.Header.TryGetValue(key, out value))
                throw new MissionFailure("bad_policy", "Ascent policy lacks " + key);
            return value;
        }

        private static int[] Ints(string text)
        {
            return text.Split(',').Select(x => int.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        }

        // ---------------------------------------------------------------- ascent

        private sealed class Stager
        {
            public readonly int[] Plan;
            public int Transitions, PropulsionIndex = -1;
            private readonly HashSet<int> groups = new HashSet<int>();

            public Stager(int[] plan) { Plan = plan; }

            public List<ModuleEngines> Ready(Vessel vessel)
            {
                return VesselInfo.Engines(vessel).Where(e => VesselInfo.Active(e) && VesselInfo.HasFuel(e)
                                                             && VesselInfo.AvailableThrust(e, vessel) > 1e-3).ToList();
            }

            public bool RegisterReady(Vessel vessel)
            {
                List<ModuleEngines> ready = Ready(vessel);
                if (ready.Count == 0) return false;
                HashSet<int> stages = new HashSet<int>(ready.Select(e => e.part.inverseStage));
                if (stages.Count != 1)
                    throw new MissionFailure("parallel_stages", "Simultaneous engine stages are not supported");
                if (groups.Add(stages.First())) PropulsionIndex++;
                return true;
            }

            public bool Depleted(Vessel vessel)
            {
                List<ModuleEngines> active = VesselInfo.Engines(vessel).Where(VesselInfo.Active).ToList();
                return active.Count > 0 && active.All(e => !VesselInfo.HasFuel(e) || VesselInfo.AvailableThrust(e, vessel) <= 1e-3);
            }
        }

        private bool Activate(Stager stager)
        {
            if (stager.Transitions >= stager.Plan.Length) return false;
            int before = vessel.currentStage, expected = stager.Plan[stager.Transitions];
            if (expected != before - 1)
                throw new MissionFailure("stage_changed", "KSP stage changed outside the autopilot");
            StageManager.ActivateNextStage();
            Vessel owner = controlPart.vessel;
            if (owner != vessel)
            {
                vessel = owner;
                if (FlightGlobals.ActiveVessel != vessel) FlightGlobals.SetActiveVessel(vessel);
                attitude.Attach(vessel);
            }
            if (vessel.currentStage != expected)
                throw new MissionFailure("stage_changed", "Expected stage " + expected + ", received " + vessel.currentStage);
            stager.Transitions++;
            Write("activate_stage", "before", before, "after", vessel.currentStage, "ut", Now);
            return true;
        }

        private IEnumerator Ascent()
        {
            SetPhase("ascent", "Learned ascent to a Kerbin parking orbit");
            controlPart = vessel.GetReferenceTransformPart();
            if (controlPart == null) controlPart = vessel.rootPart;
            int[] fuelStages = Ints(Header("vehicle.fuel_stages"));
            double initialMass = VesselInfo.Mass(vessel);
            double[] initialFuel = fuelStages.Select(s => VesselInfo.StagePropellant(vessel, s)).ToArray();
            Stager stager = new Stager(Ints(Header("vehicle.stage_plan")));
            double startUt = Now;
            attitude.Attach(vessel);
            attitude.Throttle = 0;
            attitude.Point(Up());
            attitude.Throttle = 1.0;
            bool ready = Activate(stager);
            while (ready)
            {
                yield return new WaitUt(Now + StageSettle);
                if (stager.RegisterReady(vessel)) break;
                ready = Activate(stager);
            }
            if (!ready) throw new MissionFailure("no_engine", "Stage plan finished without an ignited engine");
            bool leftGround = false;
            string reason = "step_limit";
            int maxSteps = (int)Math.Ceiling(ascent.MaxTime / ascent.ControlDt) + 2;
            for (int step = 0; step <= maxSteps; step++)
            {
                if (stager.PropulsionIndex >= ascent.StageCount)
                    throw new MissionFailure("extra_stage", "More propulsive stages than the trained craft");
                AscentTelemetry t = SampleAscent(stager.PropulsionIndex, startUt, initialMass, fuelStages, initialFuel);
                double surface = VesselInfo.SurfaceAltitude(vessel);
                leftGround |= vessel.situation == Vessel.Situations.FLYING || surface > 20;
                bool landed = vessel.situation == Vessel.Situations.LANDED || vessel.situation == Vessel.Situations.SPLASHED;
                reason = ascent.TerminalReason(t.Energy, t.Apoapsis, t.Periapsis, surface, landed, leftGround, t.Elapsed);
                float[] observation = ascent.Observation(t);
                Write("telemetry", "step", step, "ut", Now, "altitude", t.Altitude, "apoapsis", t.Apoapsis,
                          "periapsis", t.Periapsis, "pitch", t.Pitch, "stage", t.Stage, "observation", observation);
                if (reason.Length > 0) break;
                if (step == maxSteps) { reason = "step_limit"; break; }
                if (stager.Depleted(vessel))
                {
                    bool staged = false;
                    while (Activate(stager))
                    {
                        yield return new WaitUt(Now + StageSettle);
                        if (stager.RegisterReady(vessel)) { staged = true; break; }
                    }
                    if (!staged) { reason = "out_of_fuel"; break; }
                    continue; // sample the new mass, fuel and stage before inference
                }
                double[] action = ascentPolicy.Act(observation);
                double throttle, pitch;
                AscentContract.Decode(action, out throttle, out pitch);
                attitude.Point(PitchHeading90(pitch));
                attitude.Throttle = throttle;
                Detail = string.Format(CultureInfo.InvariantCulture, "Learned ascent: {0:0.0} km, apoapsis {1:0.0} km",
                                       t.Altitude / 1000, t.Apoapsis / 1000);
                Write("action", "step", step, "ut", Now, "policy_action", action, "throttle", throttle, "pitch", pitch);
                yield return new WaitUt(Now + ascent.ControlDt);
            }
            attitude.Throttle = 0;
            attitude.Release();
            Write("ascent_end", "reason", reason, "ut", Now, "apoapsis", vessel.orbit.ApA, "periapsis", vessel.orbit.PeA);
            if (reason != "orbit_achieved")
                throw new MissionFailure(reason, "Learned ascent ended without orbit: " + reason);
            if (controlPart.vessel != vessel)
                throw new MissionFailure("vessel_changed", "The controlling part left the vessel during ascent");
        }

        private AscentTelemetry SampleAscent(int stage, double startUt, double initialMass, int[] fuelStages, double[] initialFuel)
        {
            Vector3d rWorld = vessel.CoMD - vessel.mainBody.position;
            double radius = rWorld.magnitude;
            Vector3d up = rWorld / radius;
            Vector3d v = vessel.obt_velocity;
            double mass = VesselInfo.Mass(vessel);
            int index = Math.Max(0, Math.Min(stage, fuelStages.Length - 1));
            return new AscentTelemetry
            {
                Altitude = radius - vessel.mainBody.Radius,
                VerticalSpeed = Vector3d.Dot(v, up),
                HorizontalSpeed = Vector3d.Dot(v, VesselInfo.East(vessel)),
                Apoapsis = vessel.orbit.ApA,
                Periapsis = vessel.orbit.PeA,
                Energy = v.sqrMagnitude / 2 - vessel.mainBody.gravParameter / radius,
                DynamicPressure = vessel.dynamicPressurekPa * 1000.0,
                FuelFraction = VesselInfo.StagePropellant(vessel, fuelStages[index]) / initialFuel[index],
                MassFraction = mass / initialMass,
                Pitch = VesselInfo.Pitch(vessel),
                Throttle = attitude.Throttle,
                Elapsed = Now - startUt,
                ThrustToWeight = VesselInfo.AvailableThrust(vessel) / mass / Units.G0,
                Stage = stage,
            };
        }

        private Vector3d Up() { return (vessel.CoMD - vessel.mainBody.position).normalized; }

        private Vector3d PitchHeading90(double pitchDegrees)
        {
            double p = pitchDegrees * Math.PI / 180.0;
            return (Math.Cos(p) * VesselInfo.East(vessel) + Math.Sin(p) * Up()).normalized;
        }

        // -------------------------------------------------------------- transfer

        private sealed class KspEphemeris : IEphemeris
        {
            private readonly Orbit orbit;
            public KspEphemeris(CelestialBody body) { orbit = body.orbit; }

            public void State(double ut, out V3 position, out V3 velocity)
            {
                position = Frames.FromOrbit(orbit.getRelativePositionAtUT(ut));
                velocity = Frames.FromOrbit(orbit.getOrbitalVelocityAtUT(ut));
            }
        }

        private static Body CoreBody(CelestialBody body)
        {
            return new Body
            {
                Mu = body.gravParameter, Radius = body.Radius,
                AtmosphereDepth = body.atmosphere ? body.atmosphereDepth : 0.0,
                SphereOfInfluence = body.sphereOfInfluence,
                SemiMajorAxis = body.orbit != null ? body.orbit.semiMajorAxis : 0.0,
            };
        }

        private void ParkingState(out V3 r, out V3 v, out double ut)
        {
            ut = Now;
            r = Frames.FromOrbit(vessel.orbit.getRelativePositionAtUT(ut));
            v = Frames.FromOrbit(vessel.orbit.getOrbitalVelocityAtUT(ut));
        }

        private double AvailableDeltaV()
        {
            return Kepler.RocketDeltaV(VesselInfo.Mass(vessel), VesselInfo.DryMass(vessel), VesselInfo.SpecificImpulse(vessel));
        }

        private IEnumerator Transfer()
        {
            CelestialBody source = vessel.mainBody;
            IEphemeris ephemeris = new KspEphemeris(target);
            for (int window = 0; window < ParkingWindows; window++)
            {
                SetPhase("transfer_planning", "Searching transfer window " + (window + 1) + " of " + ParkingWindows);
                V3 r, v;
                double epoch;
                ParkingState(out r, out v, out epoch);
                if (window > 0)
                {
                    double energy, e, pe, ap;
                    Kepler.Elements(r, v, source.gravParameter, out energy, out e, out pe, out ap);
                    double a = -source.gravParameter / (2 * energy);
                    double coast = window * 2 * Math.PI * Math.Sqrt(a * a * a / source.gravParameter);
                    Kepler.Propagate(r, v, coast, source.gravParameter, out r, out v);
                    epoch += coast;
                }
                TransferPlan plan = Planner.PlanChildTransfer(CoreBody(source), CoreBody(target), ephemeris, r, v,
                                                              epoch, targetAltitude);
                double available = AvailableDeltaV();
                Write("mission_plan", "window", window, "departure_ut", plan.DepartureUt, "arrival_ut", plan.ArrivalUt,
                          "delta_v", plan.DeltaV, "estimated_total_delta_v", plan.TotalDeltaV, "available_delta_v", available);
                if (available < plan.TotalDeltaV + DeltaVReserve)
                    throw new MissionFailure("insufficient_delta_v", string.Format(CultureInfo.InvariantCulture,
                        "Insufficient delta-v: {0:0} m/s available, {1:0} m/s needed", available, plan.TotalDeltaV + DeltaVReserve));
                EncounterResult result = new EncounterResult();
                yield return PrepareEncounter(plan, ephemeris, source.gravParameter, result);
                if (result.Node == null)
                {
                    Write("mission_window_rejected", "window", window, "reason", result.Reason);
                    continue;
                }
                SetPhase("maneuver", "Executing the transfer burn");
                yield return ExecuteNode(result.Node, source);
                RemoveNode(result.Node);
                yield break;
            }
            throw new MissionFailure("no_encounter", "No verified " + target.bodyName + " encounter in " + ParkingWindows + " windows");
        }

        private sealed class EncounterResult
        {
            public ManeuverNode Node;
            public string Reason = "";
        }

        private struct Trial
        {
            public double AbsError, Scale, Error;
            public V3 Delta;
        }

        /// <summary>Set a node from a core-frame inertial delta-v (inverse of KSP's node basis).</summary>
        private static void SetNodeVector(ManeuverNode node, V3 delta)
        {
            Vector3d position = node.patch.getRelativePositionAtUT(node.UT).xzy;
            Vector3d velocity = node.patch.getOrbitalVelocityAtUT(node.UT).xzy;
            QuaternionD basis = QuaternionD.LookRotation(velocity, Vector3d.Cross(-position, velocity));
            node.DeltaV = QuaternionD.Inverse(basis) * Frames.ToWorld(delta);
            node.solver.UpdateFlightPlan();
        }

        /// <summary>The node encodes the requested burn: within 0.01 m/s plus 0.1%.</summary>
        private static bool NodeMatches(ManeuverNode node, V3 delta)
        {
            return (NodeBurn(node) - delta).Norm <= 0.01 + 1e-3 * delta.Norm;
        }

        /// <summary>Wait (bounded) for KSP's solver to reflect a just-set node, then verify it.</summary>
        private static IEnumerator VerifyNode(ManeuverNode node, V3 delta, string what)
        {
            for (int frame = 0; frame < 10 && !NodeMatches(node, delta); frame++)
            {
                node.solver.UpdateFlightPlan();
                yield return null;
            }
            if (!NodeMatches(node, delta))
                throw new MissionFailure("node_frame", string.Format(CultureInfo.InvariantCulture,
                    "KSP node vector disagrees with the {0} ({1:0.000} m/s difference)", what, (NodeBurn(node) - delta).Norm));
        }

        private static V3 NodeBurn(ManeuverNode node)
        {
            if (node.nextPatch == null) return V3.Zero;
            return Frames.FromOrbit(node.nextPatch.getOrbitalVelocityAtUT(node.UT) - node.patch.getOrbitalVelocityAtUT(node.UT));
        }

        /// <summary>Predicted periapsis altitude at the target, or NaN when the patches miss or pass another body.</summary>
        private double PredictedPeriapsis(ManeuverNode node, out string description)
        {
            List<string> bodies = new List<string>();
            Orbit patch = node.nextPatch;
            for (int i = 0; i < 8 && patch != null; i++)
            {
                bodies.Add(patch.referenceBody.bodyName);
                if (patch.referenceBody == target)
                {
                    bool clean = bodies.All(b => b == target.bodyName || b == vessel.mainBody.bodyName);
                    description = string.Join(">", bodies.ToArray());
                    return clean && !double.IsNaN(patch.PeA) ? patch.PeA : double.NaN;
                }
                bool transition = patch.patchEndTransition == Orbit.PatchTransitionType.ENCOUNTER ||
                                  patch.patchEndTransition == Orbit.PatchTransitionType.ESCAPE;
                patch = transition ? patch.nextPatch : null;
            }
            description = string.Join(">", bodies.ToArray());
            return double.NaN;
        }

        private IEnumerator PrepareEncounter(TransferPlan plan, IEphemeris ephemeris, double mu, EncounterResult result)
        {
            if (plan.DepartureUt < Now + 30)
                throw new MissionFailure("stale_plan", "Transfer plan is stale");
            ManeuverNode node = vessel.patchedConicSolver.AddManeuverNode(plan.DepartureUt);
            ownedNodes.Add(node);
            List<Trial> trials = new List<Trial>();
            bool keep = false;
            try
            {
                // Each trial sets the node, lets the solver run, then reads the patches.
                Func<double, IEnumerator> sample = null;
                double lastError = double.NaN;
                sample = scale => SampleTrial(node, plan, ephemeris, mu, scale, trials, e => lastError = e);
                yield return sample(1.0);
                if (trials.Count == 0 || trials.Min(t => t.AbsError) > PeriapsisTolerance)
                    foreach (double scale in new[] { 0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0 })
                        yield return sample(scale);
                List<Trial> ordered = trials.OrderBy(t => t.Scale).ToList();
                Trial? low = null, high = null;
                double bestBracket = double.PositiveInfinity;
                for (int i = 0; i + 1 < ordered.Count; i++)
                    if (ordered[i].Error * ordered[i + 1].Error <= 0)
                    {
                        double score = Math.Min(ordered[i].AbsError, ordered[i + 1].AbsError);
                        if (score < bestBracket) { bestBracket = score; low = ordered[i]; high = ordered[i + 1]; }
                    }
                if (low.HasValue)
                {
                    double lo = low.Value.Scale, hi = high.Value.Scale, loError = low.Value.Error;
                    for (int i = 0; i < 14; i++)
                    {
                        double mid = (lo + hi) / 2;
                        lastError = double.NaN;
                        yield return sample(mid);
                        if (double.IsNaN(lastError) || Math.Abs(lastError) <= PeriapsisTolerance) break;
                        if (lastError * loError <= 0) hi = mid; else { lo = mid; loError = lastError; }
                    }
                }
                if (trials.Count == 0) { result.Reason = "no_admissible_encounter"; yield break; }
                Trial best = trials.OrderBy(t => t.AbsError).First();
                if (best.AbsError > Math.Max(5000.0, PeriapsisTolerance))
                {
                    result.Reason = string.Format(CultureInfo.InvariantCulture, "periapsis_error_{0:0}_m", best.AbsError);
                    yield break;
                }
                SetNodeVector(node, best.Delta);
                yield return null;
                string route;
                double periapsis = PredictedPeriapsis(node, out route);
                if (double.IsNaN(periapsis) || Math.Abs(periapsis - plan.TargetAltitude) > Math.Max(5000.0, PeriapsisTolerance))
                {
                    // Re-applying the best trial must reproduce its encounter; record what KSP predicted instead.
                    result.Reason = string.Format(CultureInfo.InvariantCulture,
                        "final_node_lost_encounter: scale {0:R}, trial periapsis {1:0} m, final {2:0} m via {3}",
                        best.Scale, best.Error + plan.TargetAltitude, periapsis, route);
                    yield break;
                }
                Write("encounter_node_verified", "offset_scale", best.Scale, "periapsis", periapsis, "route", route,
                          "node_delta_v", node.DeltaV.magnitude, "verified_by", "ksp_patched_conics");
                result.Node = node;
                keep = true;
            }
            finally
            {
                if (!keep) RemoveNode(node);
            }
        }

        private IEnumerator SampleTrial(ManeuverNode node, TransferPlan plan, IEphemeris ephemeris, double mu, double scale,
                                        List<Trial> trials, Action<double> report)
        {
            V3 delta = Planner.ScaledDeparture(plan, ephemeris, mu, scale);
            SetNodeVector(node, delta);
            yield return null;
            yield return VerifyNode(node, delta, "requested burn");
            string route;
            double periapsis = PredictedPeriapsis(node, out route);
            Write("encounter_node_trial", "offset_scale", scale, "delta_v", delta, "route", route, "periapsis", periapsis);
            if (double.IsNaN(periapsis)) yield break;
            double error = periapsis - plan.TargetAltitude;
            trials.Add(new Trial { AbsError = Math.Abs(error), Scale = scale, Error = error, Delta = delta });
            report(error);
        }

        private void RemoveNode(ManeuverNode node)
        {
            if (node != null && node.solver != null && node.solver.maneuverNodes.Contains(node)) node.RemoveSelf();
            ownedNodes.Remove(node);
        }

        // ------------------------------------------------------- node execution

        private Vector3d RemainingBurn(ManeuverNode node) { return node.GetBurnVector(vessel.orbit); }

        private IEnumerator Align(ManeuverNode node)
        {
            double deadline = Time.realtimeSinceStartup + AlignmentTimeout;
            while (true)
            {
                Vector3d remaining = RemainingBurn(node);
                if (remaining.magnitude <= NodeTolerance) yield break;
                attitude.Throttle = 0;
                attitude.Point(remaining);
                if (attitude.Error <= AlignmentTolerance) yield break;
                if (Time.realtimeSinceStartup >= deadline)
                    throw new MissionFailure("alignment_timeout", "Did not align with the burn before the deadline");
                yield return new WaitUt(Now + NodeTick);
            }
        }

        private IEnumerator ExecuteNode(ManeuverNode node, CelestialBody body)
        {
            attitude.Attach(vessel);
            try
            {
                Vector3d remaining = RemainingBurn(node);
                double dv = remaining.magnitude;
                Write("maneuver_start", "node_ut", node.UT, "delta_v", dv, "ut", Now);
                if (dv <= NodeTolerance) yield break;
                double thrust = VesselInfo.AvailableThrust(vessel), isp = VesselInfo.SpecificImpulse(vessel);
                if (thrust <= 0 || isp <= 0) throw new MissionFailure("out_of_fuel", "No active fueled engine");
                double exhaust = isp * Units.G0;
                double duration = VesselInfo.Mass(vessel) * exhaust / thrust * (1 - Math.Exp(-dv / exhaust));
                double burnStart = node.UT - duration / 2;
                yield return Align(node);
                if (burnStart < Now - 5.0) throw new MissionFailure("late_start", "Alignment finished after the burn start");
                if (burnStart - Now > WarpMargin + 5.0)
                {
                    yield return Warp(burnStart - WarpMargin);
                    yield return Align(node);
                }
                if (Now < burnStart) yield return new WaitUt(burnStart);
                double best = RemainingBurn(node).magnitude, progressUt = Now, misalignedSince = double.NaN;
                while (true)
                {
                    if (vessel.mainBody != body) throw new MissionFailure("wrong_body", "SOI changed during the burn");
                    remaining = RemainingBurn(node);
                    dv = remaining.magnitude;
                    if (dv <= NodeTolerance) break;
                    thrust = VesselInfo.AvailableThrust(vessel);
                    if (thrust <= 0 && !Validated)
                    {
                        attitude.Throttle = 0;
                        yield return StageToNextEngines();
                        thrust = VesselInfo.AvailableThrust(vessel);
                    }
                    if (thrust <= 0) throw new MissionFailure("out_of_fuel", "Engine exhausted before completing the node");
                    attitude.Point(remaining);
                    double error = attitude.Error;
                    double throttle = Math.Min(1.0, dv / (thrust / VesselInfo.Mass(vessel)));
                    if (error > AlignmentTolerance)
                    {
                        throttle = 0;
                        if (double.IsNaN(misalignedSince)) misalignedSince = Now;
                        if (Now - misalignedSince > AlignmentTimeout)
                            throw new MissionFailure("alignment_timeout", "Pointing error stayed excessive during the burn");
                    }
                    else
                    {
                        misalignedSince = double.NaN;
                        if (dv < best - Math.Max(0.001, NodeTolerance * 0.1)) { best = dv; progressUt = Now; }
                        if (Now - progressUt > 10.0) throw new MissionFailure("no_progress", "Burn made no progress for 10 s");
                    }
                    if (TimeWarp.CurrentRateIndex != 0) throw new MissionFailure("unexpected_warp", "Time warp during a burn");
                    attitude.Throttle = throttle;
                    Detail = string.Format(CultureInfo.InvariantCulture, "Burning: {0:0.0} m/s remaining", dv);
                    yield return new WaitUt(Now + NodeTick);
                }
                Write("maneuver_end", "remaining_delta_v", dv, "ut", Now);
            }
            finally
            {
                attitude.Throttle = 0;
                attitude.Release();
            }
        }

        private IEnumerator Warp(double ut)
        {
            if (ut < Now - 1e-6) throw new MissionFailure("warp_past", "Warp target is in the past");
            attitude.Throttle = 0;
            try
            {
                // KSP's WarpTo drops to 1x before SOI changes ("orbit event
                // imminent") and when it cannot warp; like kRPC's warp_to,
                // re-issue it until the target. Short remainders run at 1x.
                int refusals = 0;
                double idleSince = double.NaN;
                if (ut > Now) TimeWarp.fetch.WarpTo(ut);
                while (Now < ut)
                {
                    if (TimeWarp.CurrentRateIndex == 0 && ut - Now > 30)
                    {
                        if (double.IsNaN(idleSince)) idleSince = Time.realtimeSinceStartup;
                        else if (Time.realtimeSinceStartup - idleSince > 3)
                        {
                            if (++refusals > 10)
                                throw new MissionFailure("warp_refused", "KSP refused time warp repeatedly");
                            TimeWarp.fetch.WarpTo(ut);
                            idleSince = double.NaN;
                        }
                    }
                    else
                    {
                        idleSince = double.NaN;
                        if (TimeWarp.CurrentRateIndex > 0) refusals = 0;
                    }
                    yield return null;
                }
            }
            finally
            {
                TimeWarp.SetRate(0, true);
            }
        }

        // ------------------------------------------------------- coast/capture

        private IEnumerator CoastToTarget()
        {
            SetPhase("transfer_coast", "Coasting to the " + target.bodyName + " sphere of influence");
            double entry = double.NaN;
            Orbit patch = vessel.orbit;
            for (int i = 0; i < 8 && patch != null; i++)
            {
                Orbit next = patch.patchEndTransition == Orbit.PatchTransitionType.ENCOUNTER ||
                             patch.patchEndTransition == Orbit.PatchTransitionType.ESCAPE ? patch.nextPatch : null;
                if (next != null && next.referenceBody == target) { entry = patch.EndUT; break; }
                patch = next;
            }
            if (double.IsNaN(entry) || entry < Now) throw new MissionFailure("no_encounter", "Patched conics show no encounter after the burn");
            Write("target_soi_coast", "entry_ut", entry, "ut", Now);
            yield return Warp(entry + 2.0);
            // Rails warp can leave a few frames before the SOI switch registers.
            for (int i = 0; i < 50 && vessel.mainBody != target; i++) yield return null;
            if (vessel.mainBody != target) throw new MissionFailure("no_soi", "Vessel did not enter the target SOI");
        }

        private void CaptureState(out V3 r, out V3 v)
        {
            r = Frames.Position(vessel);
            v = Frames.Velocity(vessel);
        }

        private IEnumerator CorrectPeriapsis(double mu, double bodyRadius, double goalRadius)
        {
            const double lead = 180.0;
            if (vessel.orbit.timeToPe < lead + 60)
                throw new MissionFailure("late_correction", "Too close to periapsis for the correction burn");
            V3 r, v;
            CaptureState(out r, out v);
            V3? delta = Planner.PeriapsisCorrection(r, v, mu, bodyRadius, goalRadius, lead);
            Write("periapsis_correction", "delta_v", delta.HasValue ? (object)delta.Value : null, "periapsis", vessel.orbit.PeA);
            if (!delta.HasValue) yield break;
            SetPhase("maneuver", "Correcting the arrival periapsis");
            ManeuverNode node = vessel.patchedConicSolver.AddManeuverNode(Now + lead);
            ownedNodes.Add(node);
            try
            {
                SetNodeVector(node, delta.Value);
                yield return null;
                yield return VerifyNode(node, delta.Value, "correction");
                yield return ExecuteNode(node, target);
            }
            finally
            {
                RemoveNode(node);
            }
        }

        private IEnumerator CoastToCapture(double mu, double radius)
        {
            SetPhase("capture_coast", "Coasting to the capture point");
            V3 r, v, fr, fv;
            CaptureState(out r, out v);
            double dt = Planner.IncomingRadiusTime(r, v, mu, radius, vessel.orbit.timeToPe);
            Kepler.Propagate(r, v, dt, mu, out fr, out fv);
            attitude.Attach(vessel);
            attitude.Throttle = 0;
            attitude.Point(Frames.ToWorld(-fv));
            double deadline = Time.realtimeSinceStartup + 60;
            while (attitude.Error > AlignmentTolerance)
            {
                if (Time.realtimeSinceStartup >= deadline) throw new MissionFailure("alignment_timeout", "Retrograde alignment timed out");
                yield return new WaitUt(Now + NodeTick);
            }
            CaptureState(out r, out v);
            dt = Planner.IncomingRadiusTime(r, v, mu, radius, vessel.orbit.timeToPe);
            Write("capture_coast", "target_ut", Now + dt, "ut", Now);
            yield return Warp(Now + dt);
            if (vessel.mainBody != target) throw new MissionFailure("wrong_body", "SOI changed while coasting to capture");
        }

        private IEnumerator Capture(CaptureConfig config)
        {
            SetPhase("capture", "Learned capture burn into a " + target.bodyName + " orbit");
            V3 r, v;
            CaptureState(out r, out v);
            // Keep the arrival plane through KSP's possible frame rotation: store
            // the conic and re-evaluate its normal in the current frame each step.
            Orbit arrival = new Orbit(vessel.orbit.inclination, vessel.orbit.eccentricity, vessel.orbit.semiMajorAxis,
                                      vessel.orbit.LAN, vessel.orbit.argumentOfPeriapsis, vessel.orbit.meanAnomalyAtEpoch,
                                      vessel.orbit.epoch, vessel.orbit.referenceBody);
            double startUt = Now, initialMass = VesselInfo.Mass(vessel);
            attitude.Attach(vessel);
            attitude.Throttle = 0;
            int streak = 0;
            int maxSteps = (int)Math.Ceiling(config.MaxTime / config.ControlDt) + 2;
            try
            {
                for (int step = 0; step < maxSteps; step++)
                {
                    if (vessel.mainBody != target) throw new MissionFailure("wrong_body", "Vessel left the destination SOI");
                    CaptureState(out r, out v);
                    double ut = Now;
                    V3 goal = V3.Cross(Frames.FromOrbit(arrival.getRelativePositionAtUT(ut)),
                                       Frames.FromOrbit(arrival.getOrbitalVelocityAtUT(ut))).Unit();
                    CaptureTelemetry d = new CaptureTelemetry
                    {
                        R = r, V = v, Direction = Frames.ToCore(Frames.Nose(vessel)), GoalNormal = goal,
                        Mass = VesselInfo.Mass(vessel), InitialMass = initialMass, DryMass = VesselInfo.DryMass(vessel),
                        Thrust = VesselInfo.AvailableThrust(vessel), Isp = VesselInfo.SpecificImpulse(vessel), Time = ut - startUt,
                    };
                    double plane = CaptureContract.PlaneError(r, v, goal);
                    Orbit o = vessel.orbit;
                    List<string> failures = CaptureContract.OrbitFailures(config, o.referenceBody.bodyName, o.ApA, o.PeA,
                                                                           o.eccentricity, o.semiMajorAxis);
                    bool inBand = failures.Count == 0 && plane <= config.PlaneToleranceDeg;
                    float[] observation = CaptureContract.Observation(config, d);
                    Write("capture_telemetry", "step", step, "ut", ut, "periapsis", o.PeA, "apoapsis", o.ApA,
                              "plane_error_deg", plane, "in_band", inBand, "observation", observation);
                    streak = inBand ? streak + 1 : 0;
                    if (streak >= config.SuccessHoldSteps)
                    {
                        Write("capture_end", "success", true, "steps", step + 1, "ut", ut);
                        yield break;
                    }
                    if (r.Norm <= config.SafeRadius) throw new MissionFailure("too_low", "Capture reached the surface clearance");
                    if (r.Norm >= config.SphereOfInfluence) throw new MissionFailure("left_soi", "Vessel left the destination SOI");
                    if (d.Time >= config.MaxTime) throw new MissionFailure("time_limit", "Capture time limit reached");
                    if (d.Mass <= d.DryMass + 1e-6 && !inBand) throw new MissionFailure("out_of_fuel", "Capture exhausted propellant");
                    double[] action = capturePolicy.Act(observation);
                    V3 directionRtn;
                    double throttle = CaptureContract.Decode(config, d, action, out directionRtn);
                    if (inBand) throttle = 0;
                    if (directionRtn.Norm > 0)
                        attitude.Point(Frames.ToWorld(new Rtn(r, v).ToFrame(directionRtn)));
                    if (!(attitude.Error <= AlignmentTolerance)) throttle = 0;
                    attitude.Throttle = throttle;
                    Detail = string.Format(CultureInfo.InvariantCulture, "Learned capture: periapsis {0:0.0} km, apoapsis {1:0.0} km",
                                           o.PeA / 1000, o.ApA / 1000);
                    Write("capture_action", "step", step, "policy_action", action, "throttle", throttle,
                              "pointing_error_deg", attitude.Error);
                    yield return new WaitUt(ut + config.ControlDt);
                }
                throw new MissionFailure("step_limit", "Capture step limit reached");
            }
            finally
            {
                attitude.Throttle = 0;
                attitude.Release();
            }
        }

        private IEnumerator VerifyOrbit(Func<Orbit, List<string>> check)
        {
            SetPhase("orbit_verification", "Coasting one full orbit with the engine off");
            attitude.Detach();
            Orbit o = vessel.orbit;
            List<string> failures = check(o);
            double period = o.period, start = Now;
            if (failures.Count > 0 || !(period > 0))
                throw new MissionFailure("orbit_not_in_band", "Orbit outside target band: " + string.Join(",", failures.ToArray()));
            for (int quarter = 1; quarter <= 4; quarter++)
            {
                yield return Warp(start + period * quarter / 4);
                o = vessel.orbit;
                failures = check(o);
                if (vessel.ctrlState.mainThrottle > 1e-5) failures.Add("throttle_not_zero");
                Write("independent_orbital_period_probe", "quarter", quarter, "ut", Now, "periapsis", o.PeA,
                          "apoapsis", o.ApA, "failures", failures);
                if (failures.Count > 0)
                    throw new MissionFailure("orbit_decayed", "Orbit left the target band during the coast check");
            }
            if (Now - start < period - 1e-3) throw new MissionFailure("short_coast", "Coast check ended early");
        }
    }
}
