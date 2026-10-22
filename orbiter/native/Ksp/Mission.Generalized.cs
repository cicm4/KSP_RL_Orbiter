// Generalized missions: any supported rocket (up to four sequential liquid
// stages) to a chosen Kerbin orbit, or onward to the Mun or Minmus, with the
// vehicle-conditioned GRU ascent and the all-body LSTM capture policies.
// These policies were trained and tested only in simulation (see README).
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
    public sealed class MissionPolicies
    {
        public Policy Ascent, Capture;
        public SequencePolicy GeneralAscent, GeneralCapture;
    }

    public sealed class MissionPlan
    {
        /// <summary>Destination moon, or null to stop in Kerbin orbit.</summary>
        public CelestialBody Target;
        /// <summary>Orbit altitude at the destination (m).</summary>
        public double TargetAltitude;
        /// <summary>Kerbin orbit altitude the generalized ascent aims for (m).</summary>
        public double OrbitAltitude = 100000;
    }

    public sealed partial class Mission
    {
        /// <summary>Turn rate during the generalized capture: inside its trained 5-45 deg/s range.</summary>
        private const double CaptureTurnRate = 0.4;
        private readonly GeneralizedAscentContract generalAscent;
        private CraftReport report;

        public CraftReport Report { get { return report; } }

        /// <summary>Null for a launch, else the phase a mid-flight Run resumes from.</summary>
        public string ResumeFrom { get; private set; }

        /// <summary>Run pressed in flight: continue the mission from where the vessel is.</summary>
        private void ResumePreflight()
        {
            Validated = false;
            Require(vessel.IsControllable, "no_control", "The vessel has no crew or probe control");
            Require(vessel.patchedConicSolver != null, "no_patched_conics",
                    "Maneuver nodes are unavailable (upgrade the Tracking Station or Mission Control)");
            Require(vessel.patchedConicSolver.maneuverNodes.Count == 0, "existing_nodes", "Remove existing maneuver nodes first");
            Require(VesselInfo.Engines(vessel).Count > 0, "no_engine", "The vessel has no engines");
            Orbit o = vessel.orbit;
            if (vessel.mainBody == target)
            {
                Require(o.eccentricity >= 1 || o.timeToPe < o.period / 2, "not_inbound",
                        "Already past the " + target.bodyName + " approach; resume works on the way in");
                Require(o.PeA > 0, "impact_course", "This approach hits " + target.bodyName);
                ResumeFrom = "capture";
            }
            else if (vessel.mainBody == target.referenceBody)
            {
                bool encounter = false;
                for (Orbit patch = o; patch != null; patch = patch.patchEndTransition == Orbit.PatchTransitionType.ENCOUNTER
                                                                 || patch.patchEndTransition == Orbit.PatchTransitionType.ESCAPE
                                                                 ? patch.nextPatch : null)
                    if (patch.referenceBody == target) { encounter = true; break; }
                if (encounter) ResumeFrom = "coast";
                else
                {
                    Require(vessel.situation == Vessel.Situations.ORBITING && o.PeA > (vessel.mainBody.atmosphere ? vessel.mainBody.atmosphereDepth : 0) + 1000,
                            "not_in_orbit", "Launch from the pad, or resume from a stable " + vessel.mainBody.bodyName + " orbit");
                    ResumeFrom = "transfer";
                }
            }
            else throw new MissionFailure("wrong_body", "Resume works from " + target.referenceBody.bodyName + " orbit or on the way to " + target.bodyName);
            if (target != null)
            {
                double low, high;
                GeneralizedCaptureConfig.AltitudeBand(policies.GeneralCapture, target.Radius,
                    target.atmosphere ? target.atmosphereDepth : 0.0, out low, out high);
                Require(targetAltitude >= low - 1 && targetAltitude <= high + 1, "altitude_out_of_band", string.Format(
                    CultureInfo.InvariantCulture, "Choose a {0} orbit between {1:0} and {2:0} km", target.bodyName, low / 1000, high / 1000));
            }
            Write("preflight_passed", "mode", "resume", "resume_from", ResumeFrom, "vessel", vessel.vesselName,
                  "mass_kg", VesselInfo.Mass(vessel), "target", target.bodyName, "target_altitude_m", targetAltitude,
                  "capture_policy_sha256", policies.GeneralCapture.Sha256);
        }

        private IEnumerator GeneralizedResume()
        {
            controlPart = vessel.GetReferenceTransformPart();
            if (controlPart == null) controlPart = vessel.rootPart;
            if (ResumeFrom == "transfer") yield return Transfer();
            if (ResumeFrom == "transfer" || ResumeFrom == "coast") yield return CoastToTarget();
            yield return GeneralizedArrival();
        }

        /// <summary>From inside the target SOI on the way in: correct, coast, capture, verify.</summary>
        private IEnumerator GeneralizedArrival()
        {
            GeneralizedCaptureConfig config = GeneralizedCaptureConfig.For(policies.GeneralCapture, target.bodyName,
                target.gravParameter, target.Radius, target.atmosphere ? target.atmosphereDepth : 0.0,
                target.sphereOfInfluence, targetAltitude);
            double arrival = policies.GeneralCapture.Constant("arrival_radius_ratio") * config.Goal;
            // The all-body policy trained with dry-mass fractions up to 0.4 (at least
            // ~2.5 km/s on board). Craft that arrive nearly empty are flown by the
            // Mun/Minmus capture policy, which trained at a 0.79 dry fraction and has
            // actual KSP flights; otherwise refuse rather than fly out of distribution.
            double dryFraction = VesselInfo.ReachableDryMass(vessel) / VesselInfo.Mass(vessel);
            bool lowFuel = dryFraction > policies.GeneralCapture.Constant("dry_fraction_max") * 1.1;
            CaptureConfig moon = null;
            if (lowFuel)
            {
                Require(target.bodyName == "Mun" || target.bodyName == "Minmus", "low_fuel_capture", string.Format(
                    CultureInfo.InvariantCulture, "Only {0:0}% of the craft's mass is usable fuel; the {1} capture needs more",
                    100 * (1 - dryFraction), target.bodyName));
                moon = CaptureConfig.ForBody(capturePolicy, target.bodyName, target.gravParameter, target.Radius,
                    target.atmosphere ? target.atmosphereDepth : 0.0, target.sphereOfInfluence, targetAltitude);
            }
            Write("capture_policy", "policy", lowFuel ? "moon_capture_validated" : "generalized_capture",
                  "dry_fraction", dryFraction);
            V3 r, v;
            CaptureState(out r, out v);
            if (vessel.orbit.timeToPe >= 240 && r.Norm > arrival) yield return CorrectPeriapsis(config.Mu, config.Radius, config.Goal);
            CaptureState(out r, out v);
            if (r.Norm > arrival) yield return CoastToCapture(config.Mu, arrival);
            if (lowFuel)
            {
                yield return Capture(moon);
                yield return VerifyOrbit(o => CaptureContract.OrbitFailures(moon, o.referenceBody.bodyName, o.ApA, o.PeA,
                                                                            o.eccentricity, o.semiMajorAxis));
            }
            else
            {
                yield return GeneralizedCapture(config);
                yield return VerifyOrbit(o => CaptureOrbitFailures(config, o));
            }
            Succeeded = true;
            SetPhase("succeeded", string.Format(CultureInfo.InvariantCulture,
                "Stable {0:0.0} x {1:0.0} km orbit of {2} verified for one full revolution",
                vessel.orbit.PeA / 1000, vessel.orbit.ApA / 1000, target.bodyName));
        }

        private void GeneralizedPreflight()
        {
            report = CraftAnalyzer.Analyze(vessel, generalAscent, plan.OrbitAltitude);
            if (!report.Ok) throw new MissionFailure("unsupported_craft", report.Errors[0]);
            Require(VesselInfo.Engines(vessel).All(e => !e.EngineIgnited), "engine_running", "Engines are already running");
            Require(report.ActivationStages[0] < vessel.currentStage, "wrong_stage", "The first engine stage has already fired");
            if (target != null)
            {
                double low, high;
                GeneralizedCaptureConfig.AltitudeBand(policies.GeneralCapture, target.Radius,
                    target.atmosphere ? target.atmosphereDepth : 0.0, out low, out high);
                Require(targetAltitude >= low - 1 && targetAltitude <= high + 1, "altitude_out_of_band", string.Format(
                    CultureInfo.InvariantCulture, "Choose a {0} orbit between {1:0} and {2:0} km", target.bodyName, low / 1000, high / 1000));
            }
            CraftSpec c = report.Craft;
            Write("preflight_passed", "mode", "generalized", "vessel", vessel.vesselName, "mass_kg", VesselInfo.Mass(vessel),
                  "target", target != null ? target.bodyName : "Kerbin orbit", "target_altitude_m", targetAltitude,
                  "orbit_altitude_m", plan.OrbitAltitude, "stages", c.Stages.Select(s => new[] { s.DryMass, s.FuelMass, s.ThrustVac, s.IspVac, s.IspSl }),
                  "payload_kg", c.Payload, "drag_area", c.DragArea, "drag_alpha_area", c.DragAlphaArea, "lift_area", c.LiftArea,
                  "front_area", report.FrontArea, "side_area", report.SideArea, "pad_twr", report.PadTwr,
                  "ideal_delta_v", c.IdealDeltaV, "warnings", report.Warnings,
                  "ascent_policy_sha256", policies.GeneralAscent.Sha256, "capture_policy_sha256", policies.GeneralCapture.Sha256);
        }

        private IEnumerator GeneralizedRun()
        {
            yield return GeneralizedAscentFlight();
            if (target == null)
            {
                CraftSpec craft = report.Craft;
                yield return VerifyOrbit(o => KerbinOrbitFailures(craft, o));
                Succeeded = true;
                SetPhase("succeeded", string.Format(CultureInfo.InvariantCulture,
                    "Stable {0:0.0} x {1:0.0} km Kerbin orbit verified for one full revolution", vessel.orbit.PeA / 1000, vessel.orbit.ApA / 1000));
                yield break;
            }
            yield return Transfer();
            yield return CoastToTarget();
            yield return GeneralizedArrival();
        }

        private List<string> KerbinOrbitFailures(CraftSpec craft, Orbit o)
        {
            List<string> failures = new List<string>();
            if (o.referenceBody != vessel.mainBody || o.eccentricity >= 1) failures.Add("not_bound");
            if (!generalAscent.Success(craft, o.eccentricity < 1 ? -1 : 1, o.ApA, o.PeA)) failures.Add("outside_target_band");
            return failures;
        }

        private static List<string> CaptureOrbitFailures(GeneralizedCaptureConfig c, Orbit o)
        {
            List<string> failures = new List<string>();
            if (o.referenceBody.bodyName != c.BodyName) { failures.Add("wrong_body"); return failures; }
            if (!(o.eccentricity >= 0 && o.eccentricity < 1)) failures.Add("not_bound");
            if (Math.Abs(o.PeR - c.Goal) > c.Tolerance || Math.Abs(o.ApR - c.Goal) > c.Tolerance) failures.Add("outside_target_band");
            if (o.PeR <= c.Safe) failures.Add("unsafe_periapsis");
            if (o.ApR >= c.SphereOfInfluence) failures.Add("outside_soi");
            return failures;
        }

        // ------------------------------------------------------------ staging

        private List<ModuleEngines> ReadyEngines()
        {
            return VesselInfo.Engines(vessel).Where(e => VesselInfo.Active(e) && VesselInfo.HasFuel(e)
                                                         && VesselInfo.AvailableThrust(e, vessel) > 1e-3).ToList();
        }

        private bool EnginesDepleted()
        {
            List<ModuleEngines> active = VesselInfo.Engines(vessel).Where(VesselInfo.Active).ToList();
            return active.Count == 0 || active.All(e => !VesselInfo.HasFuel(e) || VesselInfo.AvailableThrust(e, vessel) <= 1e-3);
        }

        /// <summary>Activate stages (decouplers, then ignition) until a fueled engine runs; bounded.</summary>
        private IEnumerator StageToNextEngines()
        {
            for (int attempt = 0; attempt < 6; attempt++)
            {
                if (vessel.currentStage <= 0) break;
                int before = vessel.currentStage;
                StageManager.ActivateNextStage();
                if (controlPart != null && controlPart.vessel != vessel)
                {
                    vessel = controlPart.vessel;
                    if (FlightGlobals.ActiveVessel != vessel) FlightGlobals.SetActiveVessel(vessel);
                    attitude.Attach(vessel);
                }
                Write("activate_stage", "before", before, "after", vessel.currentStage, "ut", Now);
                yield return new WaitUt(Now + StageSettle);
                if (ReadyEngines().Count > 0) yield break;
            }
            if (ReadyEngines().Count == 0) throw new MissionFailure("out_of_fuel", "No further engine stage could be ignited");
        }

        private int CurrentStageIndex()
        {
            List<int> stages = ReadyEngines().Select(e => e.part.inverseStage).Distinct().ToList();
            if (stages.Count != 1) throw new MissionFailure("parallel_stages", "Engines from different stages are running together");
            int index = Array.IndexOf(report.ActivationStages, stages[0]);
            if (index < 0) throw new MissionFailure("unexpected_stage", "An unexpected engine stage ignited");
            return index;
        }

        // ------------------------------------------------------------- ascent

        private IEnumerator GeneralizedAscentFlight()
        {
            CraftSpec craft = report.Craft;
            SequencePolicy policy = policies.GeneralAscent;
            SetPhase("ascent", string.Format(CultureInfo.InvariantCulture, "Learned ascent to a {0:0} km Kerbin orbit", craft.Target / 1000));
            controlPart = vessel.GetReferenceTransformPart();
            if (controlPart == null) controlPart = vessel.rootPart;
            PolicyHistory history = new PolicyHistory(policy.Context, policy.Observations, policy.Actions);
            double[] previous = null;
            // The simulator starts each flight at commanded throttle 0 and pitch 90.
            double startUt = Now, servoPitch = 90, pitchCommand = 90, throttle = 0;
            attitude.Attach(vessel);
            attitude.Point(Up());
            attitude.Throttle = 1.0;
            yield return StageToNextEngines();
            int stage = CurrentStageIndex();
            bool leftGround = false;
            string reason = "step_limit";
            int maxSteps = (int)Math.Ceiling(generalAscent.MaxTime / generalAscent.ControlDt) + 2;
            try
            {
                for (int step = 0; step < maxSteps; step++)
                {
                    GeneralizedAscentTelemetry t = SampleGeneralAscent(stage, startUt, throttle);
                    double surface = VesselInfo.SurfaceAltitude(vessel);
                    leftGround |= vessel.situation == Vessel.Situations.FLYING || surface > 20;
                    bool landed = vessel.situation == Vessel.Situations.LANDED || vessel.situation == Vessel.Situations.SPLASHED;
                    if (generalAscent.Success(craft, t.Energy, t.Apoapsis, t.Periapsis)) { reason = "orbit_achieved"; break; }
                    if (t.Energy >= 0) { reason = "escape"; break; }
                    if (surface < -0.1 || (leftGround && landed)) { reason = "impact"; break; }
                    if (t.Time >= generalAscent.MaxTime - 1e-8) { reason = "time_limit"; break; }
                    if (stage == craft.Stages.Length - 1 && EnginesDepleted()) { reason = "out_of_fuel"; break; }
                    float[] observation = generalAscent.Observation(craft, t);
                    history.Append(observation, previous);
                    double[] action = policy.Act(history.Window);
                    previous = action;
                    GeneralizedAscentContract.Decode(action, out throttle, out pitchCommand);
                    Write("telemetry", "step", step, "ut", Now, "altitude", t.Altitude, "apoapsis", t.Apoapsis,
                          "periapsis", t.Periapsis, "pitch", t.Pitch, "servo_pitch", servoPitch, "stage", stage,
                          "stage_fuel", t.StageFuel, "observation", observation, "policy_action", action,
                          "throttle", throttle, "pitch_command", pitchCommand);
                    Detail = string.Format(CultureInfo.InvariantCulture, "Learned ascent: {0:0.0} km, apoapsis {1:0.0} km, stage {2}/{3}",
                                           t.Altitude / 1000, t.Apoapsis / 1000, stage + 1, craft.Stages.Length);
                    double until = Now + generalAscent.ControlDt, last = Now;
                    while (Now < until)
                    {
                        // The simulator's first-order, rate-limited pitch servo shapes the
                        // command; the attitude controller tracks the shaped setpoint.
                        double dt = Now - last;
                        last = Now;
                        servoPitch = GeneralizedAscentContract.Servo(servoPitch, pitchCommand, craft.AttitudeRate, craft.AttitudeTau, dt);
                        attitude.Point(PitchHeading90(servoPitch));
                        attitude.Throttle = throttle;
                        if (stage < craft.Stages.Length - 1 && EnginesDepleted())
                        {
                            yield return StageToNextEngines();
                            stage = CurrentStageIndex();
                        }
                        yield return null;
                    }
                }
            }
            finally
            {
                attitude.Throttle = 0;
                attitude.Release();
            }
            Write("ascent_end", "reason", reason, "ut", Now, "apoapsis", vessel.orbit.ApA, "periapsis", vessel.orbit.PeA);
            if (reason != "orbit_achieved") throw new MissionFailure(reason, "Learned ascent ended without orbit: " + reason);
            if (controlPart.vessel != vessel) throw new MissionFailure("vessel_changed", "The controlling part left the vessel during ascent");
        }

        private GeneralizedAscentTelemetry SampleGeneralAscent(int stage, double startUt, double commandedThrottle)
        {
            Vector3d rWorld = vessel.CoMD - vessel.mainBody.position;
            double radius = rWorld.magnitude;
            Vector3d up = rWorld / radius, v = vessel.obt_velocity;
            double energy = v.sqrMagnitude / 2 - vessel.mainBody.gravParameter / radius;
            Orbit o = vessel.orbit;
            return new GeneralizedAscentTelemetry
            {
                Altitude = radius - vessel.mainBody.Radius,
                VerticalSpeed = Vector3d.Dot(v, up),
                HorizontalSpeed = Vector3d.Dot(v, VesselInfo.East(vessel)),
                Apoapsis = energy < 0 && o.eccentricity < 1 ? o.ApA : double.PositiveInfinity,
                Periapsis = o.PeA,
                Energy = energy,
                DynamicPressure = vessel.dynamicPressurekPa * 1000.0,
                StageFuel = report.StageFuel(vessel, stage),
                Mass = VesselInfo.Mass(vessel),
                Pitch = VesselInfo.Pitch(vessel),
                Throttle = commandedThrottle,
                Time = Now - startUt,
                AvailableThrust = VesselInfo.AvailableThrust(vessel),
                Stage = stage,
            };
        }

        // ------------------------------------------------------------ capture

        private IEnumerator GeneralizedCapture(GeneralizedCaptureConfig config)
        {
            SequencePolicy policy = policies.GeneralCapture;
            SetPhase("capture", "Learned capture burn into a " + target.bodyName + " orbit");
            Orbit arrival = new Orbit(vessel.orbit.inclination, vessel.orbit.eccentricity, vessel.orbit.semiMajorAxis,
                                      vessel.orbit.LAN, vessel.orbit.argumentOfPeriapsis, vessel.orbit.meanAnomalyAtEpoch,
                                      vessel.orbit.epoch, vessel.orbit.referenceBody);
            double startUt = Now, initialMass = VesselInfo.Mass(vessel);
            PolicyHistory history = new PolicyHistory(policy.Context, policy.Observations, policy.Actions);
            double[] previous = null;
            int hold = 0, maxSteps = (int)Math.Ceiling(config.MaxTime / config.ControlDt) + 2;
            attitude.Attach(vessel);
            attitude.Throttle = 0;
            double savedRate = attitude.MaxRate;
            attitude.MaxRate = CaptureTurnRate;
            try
            {
                for (int step = 0; step < maxSteps; step++)
                {
                    if (vessel.mainBody != target) throw new MissionFailure("wrong_body", "Vessel left the destination SOI");
                    V3 r, v;
                    CaptureState(out r, out v);
                    double ut = Now;
                    V3 goal = V3.Cross(Frames.FromOrbit(arrival.getRelativePositionAtUT(ut)),
                                       Frames.FromOrbit(arrival.getOrbitalVelocityAtUT(ut))).Unit();
                    GeneralizedCaptureTelemetry d = new GeneralizedCaptureTelemetry
                    {
                        R = r, V = v, Direction = Frames.ToCore(Frames.Nose(vessel)).Unit(), GoalNormal = goal,
                        Mass = VesselInfo.Mass(vessel), InitialMass = initialMass, DryMass = VesselInfo.ReachableDryMass(vessel),
                        Thrust = VesselInfo.AvailableThrust(vessel), Isp = VesselInfo.SpecificImpulse(vessel), Time = ut - startUt,
                    };
                    double plane;
                    bool inBand = GeneralizedCaptureContract.InBand(config, r, v, goal, out plane);
                    hold = inBand ? hold + 1 : 0;
                    Orbit o = vessel.orbit;
                    if (hold >= (int)policy.Constant("success_hold_steps"))
                    {
                        Write("capture_end", "success", true, "steps", step + 1, "ut", ut, "periapsis", o.PeA, "apoapsis", o.ApA);
                        yield break;
                    }
                    if (r.Norm <= config.Safe) throw new MissionFailure("too_low", "Capture reached the surface clearance");
                    if (r.Norm >= config.SphereOfInfluence) throw new MissionFailure("left_soi", "Vessel left the destination SOI");
                    if (d.Time >= config.MaxTime) throw new MissionFailure("time_limit", "Capture time limit reached");
                    OrbitState s = new OrbitState(r, v, config.Mu);
                    if (s.Energy >= 0 && V3.Dot(r, v) > 0 && r.Norm > 1.6 * config.Goal)
                        throw new MissionFailure("missed_capture", "The capture burn did not slow the craft enough");
                    if (!(d.Isp > 0) || d.Thrust <= 0 || d.Mass <= d.DryMass + 1e-6)
                    {
                        if (inBand) { yield return new WaitUt(ut + config.ControlDt); continue; }
                        if (vessel.currentStage > 0) { yield return StageToNextEngines(); continue; }
                        throw new MissionFailure("out_of_fuel", "Capture exhausted propellant");
                    }
                    float[] observation = GeneralizedCaptureContract.Observation(config, d);
                    history.Append(observation, previous);
                    double[] action = policy.Act(history.Window);
                    previous = action;
                    V3 directionRtn;
                    double throttle = GeneralizedCaptureContract.Decode(config, d, action, out directionRtn);
                    if (inBand) throttle = 0;
                    Vector3d pointing = Frames.ToWorld(new Rtn(r, v).ToFrame(directionRtn));
                    Write("capture_telemetry", "step", step, "ut", ut, "periapsis", o.PeA, "apoapsis", o.ApA,
                          "plane_error_deg", plane, "in_band", inBand, "observation", observation,
                          "policy_action", action, "throttle", throttle);
                    Detail = string.Format(CultureInfo.InvariantCulture, "Learned capture: periapsis {0:0.0} km, apoapsis {1:0.0} km",
                                           o.PeA / 1000, o.ApA / 1000);
                    // As in training, the engine thrusts along wherever it points while it
                    // turns toward the command (holding fire until aligned stalls small
                    // corrections when only reaction wheels can turn the craft).
                    Write("capture_pointing", "step", step, "pointing_error_deg", attitude.Error);
                    while (Now < ut + config.ControlDt)
                    {
                        attitude.Point(pointing);
                        attitude.Throttle = throttle;
                        yield return null;
                    }
                }
                throw new MissionFailure("step_limit", "Capture step limit reached");
            }
            finally
            {
                attitude.MaxRate = savedRate;
                attitude.Throttle = 0;
                attitude.Release();
            }
        }
    }
}
