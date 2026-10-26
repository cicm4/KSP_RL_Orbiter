// Planet missions: from a Kerbin parking orbit to Moho, Eve, Duna, Dres, Jool
// or Eeloo. A Lambert window search picks the departure, the escape burn is
// solved from the departure hyperbola and refined against KSP's own patched
// conics, heliocentric course corrections aim the arrival periapsis, and the
// capture uses the hand-written feedback law the capture policy learned from,
// split over several periapsis passes when the engine is weak for the body.
using System;
using System.Collections;
using System.Globalization;
using OrbiterRL.Core;
using UnityEngine;

namespace OrbiterRL
{
    public sealed partial class Mission
    {
        /// <summary>Seconds between planning and a heliocentric course correction.</summary>
        private const double CorrectionLead = 600.0;
        /// <summary>Arrival periapsis error (m, at least 2% of the SOI) left to the in-SOI correction.</summary>
        private const double CruiseTolerance = 20000.0;
        /// <summary>Capture in one teacher pass only with thrust above this fraction of local gravity...</summary>
        private const double WeakThrust = 0.8;
        /// <summary>...and a circularizing burn shorter than this many radian-times (goal / circular speed).</summary>
        private const double LongBurn = 0.8, PassBurn = 0.5;
        /// <summary>Apoapsis of the first capture ellipse, as a fraction of the sphere of influence.</summary>
        private const double CaptureApoapsis = 0.5;

        /// <summary>Launch this long before the departure: climb, one parking orbit, planning.</summary>
        private const double LaunchLead = 3600.0;

        private double nominalArrival = double.NaN;
        private InterplanetaryPlan padPlan;

        private static CelestialBody Home { get { return FlightGlobals.GetHomeBody(); } }

        /// <summary>The destination orbits the Sun like Kerbin (a planet, not a moon).</summary>
        private bool Interplanetary
        {
            get { return target != null && target.referenceBody != null && target.referenceBody != Home
                         && target.referenceBody == Home.referenceBody; }
        }

        /// <summary>Delta-v left for maneuvers: KSP's stage simulation when ready, else the active stage.</summary>
        private double AvailableMissionDeltaV()
        {
            VesselDeltaV deltaV = vessel.VesselDeltaV;
            if (deltaV != null && deltaV.IsReady && deltaV.TotalDeltaVVac > 0) return deltaV.TotalDeltaVVac;
            return AvailableDeltaV();
        }

        private InterplanetaryPlan WindowPlan(double epoch, double parkingRadius, V3 parkingNormal)
        {
            CelestialBody source = Home, parent = Home.referenceBody;
            double span = 1.05 * Planner.SynodicPeriod(parent.gravParameter, source.orbit.semiMajorAxis, target.orbit.semiMajorAxis);
            return Planner.PlanInterplanetary(CoreBody(source), CoreBody(target), new KspEphemeris(source), new KspEphemeris(target),
                                              parent.gravParameter, epoch, parkingRadius, targetAltitude, span,
                                              parkingNormal: parkingNormal);
        }

        /// <summary>Normal of the orbit the eastward (heading 90) climb will reach, or of the current orbit.</summary>
        private V3 ParkingNormal()
        {
            if (vessel.situation == Vessel.Situations.ORBITING || vessel.situation == Vessel.Situations.SUB_ORBITAL)
            {
                V3 r, v;
                double ut;
                ParkingState(out r, out v, out ut);
                return V3.Cross(r, v).Unit();
            }
            return V3.Cross(Frames.ToCore(vessel.CoMD - vessel.mainBody.position), Frames.ToCore(VesselInfo.East(vessel))).Unit();
        }

        // The window panel re-checks the craft every second; one search per game day is enough.
        private static string budgetKey;
        private static double budgetValue;

        /// <summary>Delta-v after reaching Kerbin orbit that the trip needs, with a margin for corrections.</summary>
        private double TripBudget(double parkingAltitude)
        {
            string key = string.Format(CultureInfo.InvariantCulture, "{0}/{1:R}/{2:R}/{3}", target.bodyName, targetAltitude,
                                       parkingAltitude, Math.Floor(Now / KSPUtil.dateTimeFormatter.Day));
            if (key == budgetKey) return budgetValue;
            budgetValue = UncachedTripBudget(parkingAltitude);
            budgetKey = key;
            return budgetValue;
        }

        private double UncachedTripBudget(double parkingAltitude)
        {
            double parking = Home.Radius + parkingAltitude;
            double cost = Interplanetary
                ? WindowPlan(Now, parking, ParkingNormal()).TotalDeltaV
                : Planner.MoonBudget(Home.gravParameter, parking, target.orbit.semiMajorAxis, target.gravParameter,
                                     target.Radius + targetAltitude);
            return 1.05 * cost + 50.0;
        }

        // ------------------------------------------------------------ departure

        /// <summary>
        /// Wait for the window on the launchpad: KSP only allows slow time warp in a
        /// low orbit, but any rate on the ground.
        /// </summary>
        private IEnumerator WaitOnPad()
        {
            padPlan = WindowPlan(Now + LaunchLead, Home.Radius + plan.OrbitAltitude, ParkingNormal());
            Write("interplanetary_window", "departure_ut", padPlan.DepartureUt, "arrival_ut", padPlan.ArrivalUt,
                  "ejection_delta_v", padPlan.EjectionDeltaV, "capture_delta_v", padPlan.CaptureDeltaV);
            double launch = padPlan.DepartureUt - LaunchLead;
            if (launch > Now + 600)
            {
                SetPhase("window_wait", string.Format(CultureInfo.InvariantCulture,
                    "Waiting {0:0} days on the pad for the {1} launch window", (launch - Now) / KSPUtil.dateTimeFormatter.Day,
                    target.bodyName));
                yield return Warp(launch);
                for (int i = 0; i < 100 && vessel.packed; i++) yield return null;
                yield return new WaitUt(Now + 3.0); // let physics and terrain settle at 1x
                if (vessel.situation != Vessel.Situations.PRELAUNCH)
                    throw new MissionFailure("not_on_pad", "The rocket left the launchpad while waiting");
            }
        }

        private IEnumerator PlanetTransfer()
        {
            CelestialBody source = vessel.mainBody, parent = source.referenceBody;
            if (source != Home) throw new MissionFailure("wrong_body", "Planet transfers start from a " + Home.bodyName + " orbit");
            SetPhase("transfer_planning", "Finding the next launch window to " + target.bodyName);
            V3 r, v;
            double epoch;
            ParkingState(out r, out v, out epoch);
            double period = vessel.orbit.period;
            InterplanetaryPlan plan = padPlan != null && padPlan.DepartureUt > Now + 0.5 * period + 300 ? padPlan
                : WindowPlan(epoch + period + 600, vessel.orbit.semiMajorAxis, ParkingNormal());
            double available = AvailableMissionDeltaV(), needed = 1.03 * plan.TotalDeltaV + DeltaVReserve;
            Write("interplanetary_plan", "departure_ut", plan.DepartureUt, "arrival_ut", plan.ArrivalUt,
                  "departure_vinf", plan.DepartureVinf, "arrival_vinf", plan.ArrivalVinf, "ejection_delta_v", plan.EjectionDeltaV,
                  "capture_delta_v", plan.CaptureDeltaV, "available_delta_v", available, "long_way", plan.LongWay);
            if (available < needed)
                throw new MissionFailure("insufficient_delta_v", string.Format(CultureInfo.InvariantCulture,
                    "Not enough delta-v for {0}: {1:0} m/s left, about {2:0} m/s needed", target.bodyName, available, needed));
            nominalArrival = plan.ArrivalUt;
            // The escape burn is searched within half a parking orbit of the departure.
            double wake = plan.DepartureUt - period - 300;
            if (wake > Now + 60)
            {
                // Low orbits cap time warp (Kerbin: 50x below 120 km); a long wait there takes real days.
                if (wake - Now > 10 * KSPUtil.dateTimeFormatter.Day && padPlan == null)
                    throw new MissionFailure("window_far", string.Format(CultureInfo.InvariantCulture,
                        "The {0} window opens in {1:0} days and KSP warps slowly in low orbit; launch from the pad instead",
                        target.bodyName, (wake - Now) / KSPUtil.dateTimeFormatter.Day));
                SetPhase("window_wait", string.Format(CultureInfo.InvariantCulture,
                    "Waiting {0:0} days in orbit for the {1} launch window", (wake - Now) / KSPUtil.dateTimeFormatter.Day, target.bodyName));
                yield return Warp(wake);
            }
            SetPhase("transfer_planning", "Aiming the escape burn toward " + target.bodyName);
            ManeuverNode node = null;
            yield return AimEscape(plan, parent, result => node = result);
            // The solved burn, not the estimate, must leave enough for the capture.
            double escape = RemainingBurn(node).magnitude, left = AvailableMissionDeltaV();
            Write("escape_budget", "escape_delta_v", escape, "estimated_ejection_delta_v", plan.EjectionDeltaV,
                  "capture_delta_v", plan.CaptureDeltaV, "available_delta_v", left);
            if (left < escape + 1.03 * plan.CaptureDeltaV + DeltaVReserve)
            {
                RemoveNode(node);
                throw new MissionFailure("insufficient_delta_v", string.Format(CultureInfo.InvariantCulture,
                    "The escape burn needs {0:0} m/s and the capture about {1:0} m/s, but only {2:0} m/s is left",
                    escape, plan.CaptureDeltaV, left));
            }
            SetPhase("maneuver", "Escape burn toward " + target.bodyName);
            try
            {
                yield return ExecuteNode(node, source);
            }
            finally
            {
                RemoveNode(node);
            }
            yield return LeaveSoi();
        }

        private static string PatchChain(Orbit start)
        {
            System.Text.StringBuilder text = new System.Text.StringBuilder();
            Orbit patch = start;
            for (int i = 0; i < 6 && patch != null; i++)
            {
                text.Append(patch.referenceBody.bodyName).Append(':').Append(patch.patchEndTransition)
                    .Append(string.Format(CultureInfo.InvariantCulture, "(e={0:0.###},end={1:0})", patch.eccentricity, patch.EndUT)).Append(' ');
                patch = patch.nextPatch;
            }
            return text.ToString();
        }

        /// <summary>Heliocentric patch after the node's escape, or null when KSP shows none.</summary>
        private static Orbit EscapePatch(Orbit start, CelestialBody parent)
        {
            Orbit patch = start;
            for (int i = 0; i < 4 && patch != null; i++)
            {
                if (patch.referenceBody == parent) return patch;
                patch = patch.patchEndTransition == Orbit.PatchTransitionType.ESCAPE ? patch.nextPatch : null;
            }
            return null;
        }

        /// <summary>True when the escape path enters a moon's sphere of influence before leaving.</summary>
        private static bool MoonFlyby(Orbit start, CelestialBody parent)
        {
            Orbit patch = start;
            for (int i = 0; i < 6 && patch != null && patch.referenceBody != parent; i++)
            {
                if (patch.patchEndTransition == Orbit.PatchTransitionType.ENCOUNTER) return true;
                patch = patch.patchEndTransition == Orbit.PatchTransitionType.ESCAPE ? patch.nextPatch : null;
            }
            return false;
        }

        /// <summary>
        /// Escape node from the departure hyperbola, then shooting: KSP predicts the
        /// heliocentric state after leaving the SOI, a Lambert arc from there to the
        /// arrival point gives the velocity still missing, and that difference is
        /// added to the requested excess velocity until the two agree.
        /// </summary>
        private IEnumerator AimEscape(InterplanetaryPlan plan, CelestialBody parent, Action<ManeuverNode> done)
        {
            CelestialBody source = vessel.mainBody;
            IEphemeris ephemeris = new KspEphemeris(target);
            V3 vinf = plan.DepartureVinf;
            double center = plan.DepartureUt, miss = double.PositiveInfinity;
            int flybyShifts = 0;
            ManeuverNode node = null;
            bool keep = false;
            try
            {
                for (int iteration = 0; iteration < 10; iteration++)
                {
                    V3 r, v;
                    double epoch, burnUt;
                    V3 delta;
                    ParkingState(out r, out v, out epoch);
                    if (!Planner.EjectionBurn(r, v, epoch, source.gravParameter, vinf, center, out burnUt, out delta))
                        throw new MissionFailure("no_escape_geometry", "No escape burn reaches the " + target.bodyName + " departure direction");
                    if (burnUt < Now + 120)
                    {
                        center += vessel.orbit.period;
                        iteration--;
                        if (center > plan.DepartureUt + (3 + flybyShifts) * vessel.orbit.period)
                            throw new MissionFailure("late_window", "The escape burn time already passed");
                        continue;
                    }
                    center = burnUt;
                    if (node != null) RemoveNode(node);
                    node = vessel.patchedConicSolver.AddManeuverNode(burnUt);
                    ownedNodes.Add(node);
                    SetNodeVector(node, delta);
                    yield return null;
                    yield return VerifyNode(node, delta, "escape burn");
                    Orbit helio = EscapePatch(node.nextPatch, parent);
                    for (int frame = 0; frame < 20 && helio == null && !MoonFlyby(node.nextPatch, parent); frame++)
                    {
                        node.solver.UpdateFlightPlan();
                        yield return null;
                        helio = EscapePatch(node.nextPatch, parent);
                    }
                    if (MoonFlyby(node.nextPatch, parent))
                    {
                        // A moon on the way out would bend the escape unpredictably; leave
                        // one parking orbit later, when it has moved on.
                        Write("escape_moon_flyby", "chain", PatchChain(node.nextPatch), "burn_ut", burnUt);
                        if (++flybyShifts > 40)
                            throw new MissionFailure("moon_in_the_way", "A moon stays in the escape path");
                        center = burnUt + vessel.orbit.period;
                        iteration--;
                        continue;
                    }
                    if (helio == null)
                    {
                        Write("escape_patches", "chain", PatchChain(node.nextPatch), "delta_v", delta, "burn_ut", burnUt,
                              "patch_limit", GameSettings.CONIC_PATCH_LIMIT);
                        throw new MissionFailure("no_escape", "KSP predicts no escape from " + source.bodyName + " after the burn");
                    }
                    double exitUt = helio.StartUT;
                    V3 position = Frames.FromOrbit(helio.getRelativePositionAtUT(exitUt));
                    V3 velocity = Frames.FromOrbit(helio.getOrbitalVelocityAtUT(exitUt));
                    V3 goal, goalVelocity, need, unused;
                    ephemeris.State(plan.ArrivalUt, out goal, out goalVelocity);
                    Kepler.Lambert(position, goal, plan.ArrivalUt - exitUt, parent.gravParameter, plan.LongWay, out need, out unused);
                    V3 error = need - velocity;
                    miss = error.Norm;
                    string route;
                    double periapsis = PredictedPeriapsis(node, out route);
                    Write("escape_iteration", "iteration", iteration, "burn_ut", burnUt, "delta_v", delta, "exit_ut", exitUt,
                          "velocity_error", miss, "route", route, "predicted_periapsis", periapsis);
                    if (miss < 0.5) break;
                    vinf = vinf + error;
                }
                if (!(miss < 20.0))
                    throw new MissionFailure("escape_not_converged", string.Format(CultureInfo.InvariantCulture,
                        "The escape burn did not converge ({0:0.0} m/s off)", miss));
                keep = true;
                done(node);
            }
            finally
            {
                if (!keep && node != null) RemoveNode(node);
            }
        }

        private IEnumerator LeaveSoi()
        {
            CelestialBody parent = Home.referenceBody;
            SetPhase("transfer_coast", "Leaving the " + vessel.mainBody.bodyName + " sphere of influence");
            if (vessel.mainBody != parent)
            {
                if (vessel.orbit.patchEndTransition != Orbit.PatchTransitionType.ESCAPE)
                    throw new MissionFailure("no_escape", "The orbit does not leave " + vessel.mainBody.bodyName);
                yield return Warp(vessel.orbit.EndUT + 30);
                for (int i = 0; i < 50 && vessel.mainBody != parent; i++) yield return null;
            }
            if (vessel.mainBody != parent) throw new MissionFailure("no_soi", "The vessel did not reach solar orbit");
        }

        // --------------------------------------------------------------- cruise

        /// <summary>Predicted periapsis at the target from the vessel's own patches (NaN without an encounter).</summary>
        private double ArrivalPeriapsis(out string route, out double entryUt)
        {
            entryUt = double.NaN;
            System.Collections.Generic.List<string> bodies = new System.Collections.Generic.List<string>();
            Orbit patch = vessel.orbit;
            for (int i = 0; i < 8 && patch != null; i++)
            {
                bodies.Add(patch.referenceBody.bodyName);
                if (patch.referenceBody == target)
                {
                    route = string.Join(">", bodies.ToArray());
                    entryUt = patch.StartUT;
                    bool clean = bodies.TrueForAll(b => b == target.bodyName || b == vessel.mainBody.bodyName);
                    return clean ? patch.PeA : double.NaN;
                }
                bool transition = patch.patchEndTransition == Orbit.PatchTransitionType.ENCOUNTER ||
                                  patch.patchEndTransition == Orbit.PatchTransitionType.ESCAPE;
                patch = transition ? patch.nextPatch : null;
            }
            route = string.Join(">", bodies.ToArray());
            return double.NaN;
        }

        /// <summary>Time of closest approach to the target over one solar orbit (resume without a plan).</summary>
        private double ClosestApproach()
        {
            Orbit o = vessel.orbit;
            double period = o.eccentricity < 1 ? o.period : 3.0e7, best = double.PositiveInfinity, bestUt = Now;
            for (int i = 1; i <= 2000; i++)
            {
                double ut = Now + period * i / 2000;
                double distance = (o.getRelativePositionAtUT(ut) - target.orbit.getRelativePositionAtUT(ut)).magnitude;
                if (distance < best) { best = distance; bestUt = ut; }
            }
            return bestUt;
        }

        private IEnumerator Cruise()
        {
            CelestialBody parent = Home.referenceBody;
            if (vessel.mainBody != parent) throw new MissionFailure("wrong_body", "The cruise starts in solar orbit");
            IEphemeris ephemeris = new KspEphemeris(target);
            for (int correction = 0; correction < 3; correction++)
            {
                string route;
                double entry;
                double periapsis = ArrivalPeriapsis(out route, out entry);
                Write("cruise_check", "correction", correction, "predicted_periapsis", periapsis, "route", route, "entry_ut", entry);
                // Near the target a tangential burn at the SOI edge fixes the rest for a few m/s.
                double tolerance = Math.Max(CruiseTolerance, 0.02 * target.sphereOfInfluence);
                if (!double.IsNaN(periapsis) && Math.Abs(periapsis - targetAltitude) <= tolerance) yield break;
                if (correction > 0)
                {
                    // Let the trajectory settle and correct again halfway to the encounter.
                    double halfway = Now + 0.5 * ((double.IsNaN(entry) ? nominalArrival : entry) - Now);
                    if (halfway > Now + 2 * CorrectionLead)
                    {
                        SetPhase("transfer_coast", "Coasting toward " + target.bodyName);
                        yield return Warp(halfway);
                    }
                }
                if (double.IsNaN(nominalArrival) || nominalArrival < Now + 2 * CorrectionLead)
                    nominalArrival = double.IsNaN(entry) ? ClosestApproach() : entry;
                SetPhase("course_correction", "Correcting the course to " + target.bodyName);
                V3 r = Frames.FromOrbit(vessel.orbit.getRelativePositionAtUT(Now));
                V3 v = Frames.FromOrbit(vessel.orbit.getOrbitalVelocityAtUT(Now));
                TransferPlan plan = Planner.PlanCorrection(r, v, Now, parent.gravParameter, CoreBody(target), ephemeris,
                                                           nominalArrival, targetAltitude, CorrectionLead);
                Write("course_correction_plan", "correction", correction, "delta_v", plan.DeltaV, "arrival_ut", plan.ArrivalUt,
                      "arrival_vinf", plan.ArrivalVinf);
                // Below the node executor's resolution; leave it to the in-SOI correction.
                if (plan.DeltaV.Norm < 0.3 && !double.IsNaN(periapsis)) yield break;
                EncounterResult result = new EncounterResult();
                yield return PrepareEncounter(plan, ephemeris, parent.gravParameter, result);
                if (result.Node == null)
                    throw new MissionFailure("no_encounter", "No verified " + target.bodyName + " encounter: " + result.Reason);
                SetPhase("maneuver", "Course correction toward " + target.bodyName);
                try
                {
                    yield return ExecuteNode(result.Node, parent);
                }
                finally
                {
                    RemoveNode(result.Node);
                }
                nominalArrival = plan.ArrivalUt;
            }
            string finalRoute;
            double finalEntry, finalPeriapsis = ArrivalPeriapsis(out finalRoute, out finalEntry);
            if (double.IsNaN(finalPeriapsis))
                throw new MissionFailure("no_encounter", "Course corrections did not produce a " + target.bodyName + " encounter");
        }

        // -------------------------------------------------------------- capture

        private double BurnTime(double deltaV)
        {
            double thrust = VesselInfo.AvailableThrust(vessel), isp = VesselInfo.SpecificImpulse(vessel);
            if (thrust <= 0 || isp <= 0) return double.PositiveInfinity;
            double exhaust = isp * Units.G0;
            return VesselInfo.Mass(vessel) * exhaust / thrust * (1 - Math.Exp(-deltaV / exhaust));
        }

        private double DeltaVForBurn(double seconds)
        {
            double thrust = VesselInfo.AvailableThrust(vessel), isp = VesselInfo.SpecificImpulse(vessel);
            if (thrust <= 0 || isp <= 0) return 0.0;
            double exhaust = isp * Units.G0, fraction = seconds * thrust / (VesselInfo.Mass(vessel) * exhaust);
            return fraction >= 1 ? double.PositiveInfinity : -exhaust * Math.Log(1 - fraction);
        }

        /// <summary>
        /// Retrograde periapsis burns that capture into an ellipse and shrink it,
        /// each short enough to stay near periapsis, until the remaining burn is one
        /// the feedback capture flies well; a cheap apoapsis trim keeps the periapsis
        /// at the goal between passes.
        /// </summary>
        private IEnumerator CapturePasses(GeneralizedCaptureConfig config)
        {
            for (int pass = 0; pass < 16; pass++)
            {
                if (vessel.mainBody != target) throw new MissionFailure("wrong_body", "Vessel left the destination SOI");
                if (VesselInfo.AvailableThrust(vessel) <= 0) yield return StageToNextEngines();
                Orbit o = vessel.orbit;
                double mu = config.Mu, rp = o.PeR;
                double atPeriapsis = Math.Sqrt(mu * (2 / rp - 1 / o.semiMajorAxis));
                double need = atPeriapsis - Math.Sqrt(mu / rp);
                double accel = VesselInfo.AvailableThrust(vessel) / VesselInfo.Mass(vessel), gravity = mu / (config.Goal * config.Goal);
                double burn = BurnTime(need);
                Write("capture_pass_check", "pass", pass, "needed_delta_v", need, "burn_s", burn, "thrust_accel", accel,
                      "gravity", gravity, "periapsis", o.PeA, "apoapsis", o.eccentricity < 1 ? o.ApA : double.PositiveInfinity);
                if (need < 5 || (accel >= WeakThrust * gravity && burn <= LongBurn * config.T)) yield break;
                double first = o.eccentricity >= 1 || o.ApR > CaptureApoapsis * config.SphereOfInfluence
                    ? atPeriapsis - Math.Sqrt(mu * (2 / rp - 2 / (rp + CaptureApoapsis * config.SphereOfInfluence))) : 0.0;
                double step = Math.Min(need, Math.Max(DeltaVForBurn(PassBurn * config.T), first));
                // Stop short of circular: the feedback capture flies the final pass.
                if (need - step < 0.1 * need && step > first) step = Math.Max(first, 0.9 * need);
                double peUt = Now + o.timeToPe;
                if (o.timeToPe < 120 && o.eccentricity < 1) peUt += o.period;
                V3 velocity = Frames.FromOrbit(o.getOrbitalVelocityAtUT(peUt));
                V3 delta = -step * velocity.Unit();
                SetPhase("capture", string.Format(CultureInfo.InvariantCulture,
                    "Capture pass {0}: {1:0} m/s retrograde at periapsis", pass + 1, step));
                ManeuverNode node = vessel.patchedConicSolver.AddManeuverNode(peUt);
                ownedNodes.Add(node);
                try
                {
                    SetNodeVector(node, delta);
                    yield return null;
                    yield return VerifyNode(node, delta, "capture pass");
                    Write("capture_pass", "pass", pass, "delta_v", step, "node_ut", peUt);
                    yield return ExecuteNode(node, target);
                }
                finally
                {
                    RemoveNode(node);
                }
                yield return TrimPeriapsis(config);
            }
        }

        /// <summary>Tangential burn at apoapsis that puts the periapsis back on the goal radius.</summary>
        private IEnumerator TrimPeriapsis(GeneralizedCaptureConfig config)
        {
            Orbit o = vessel.orbit;
            if (!(o.eccentricity < 1) || Math.Abs(o.PeR - config.Goal) <= 2 * config.Tolerance || o.ApR < 1.5 * config.Goal) yield break;
            double apUt = Now + o.timeToAp;
            V3 position = Frames.FromOrbit(o.getRelativePositionAtUT(apUt)), velocity = Frames.FromOrbit(o.getOrbitalVelocityAtUT(apUt));
            double ra = position.Norm, rp = config.Goal;
            double wanted = Math.Sqrt(2 * config.Mu * rp / (ra * (ra + rp)));
            V3 delta = (wanted - velocity.Norm) * velocity.Unit();
            if (delta.Norm < 0.05) yield break;
            SetPhase("capture", "Raising the periapsis back to the target altitude");
            ManeuverNode node = vessel.patchedConicSolver.AddManeuverNode(apUt);
            ownedNodes.Add(node);
            try
            {
                SetNodeVector(node, delta);
                yield return null;
                yield return VerifyNode(node, delta, "periapsis trim");
                Write("periapsis_trim", "delta_v", delta.Norm, "node_ut", apUt, "periapsis", o.PeA);
                yield return ExecuteNode(node, target);
            }
            finally
            {
                RemoveNode(node);
            }
        }

        private IEnumerator PlanetArrival(GeneralizedCaptureConfig config)
        {
            double arrival = policies.GeneralCapture.Constant("arrival_radius_ratio") * config.Goal;
            Write("capture_policy", "policy", "teacher_feedback", "dry_fraction",
                  VesselInfo.ReachableDryMass(vessel) / VesselInfo.Mass(vessel));
            V3 r, v;
            CaptureState(out r, out v);
            if (vessel.orbit.timeToPe >= 240 && r.Norm > arrival) yield return CorrectPeriapsis(config.Mu, config.Radius, config.Goal);
            yield return CapturePasses(config);
            CaptureState(out r, out v);
            Orbit o = vessel.orbit;
            bool inbound = V3.Dot(r, v) < 0 || o.timeToPe < o.timeToAp;
            if (r.Norm > arrival && (o.eccentricity >= 1 || o.ApR > arrival))
            {
                if (!inbound && o.eccentricity < 1) yield return Warp(Now + o.timeToAp + 60);
                yield return CoastToCapture(config.Mu, arrival);
            }
            yield return GeneralizedCapture(config, true);
            yield return VerifyOrbit(ob => CaptureOrbitFailures(config, ob));
        }
    }
}
