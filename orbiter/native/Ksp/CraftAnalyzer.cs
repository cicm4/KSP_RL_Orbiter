// Reads an arbitrary KSP rocket into the generalized policy's CraftSpec.
// KSP's own delta-v simulation (VesselDeltaV, which follows fuel lines,
// crossfeed and the staging order) splits the flight into propulsive phases,
// one per KSP stage that burns fuel. A phase is what the policy calls a
// stage: burn this much propellant at this thrust, then drop this much mass.
// That covers side boosters and asparagus staging as well as plain stacks.
// The ascent policy sees at most four phases; any stages above the ones the
// climb needs ride along as payload and fly the later maneuver nodes.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using OrbiterRL.Core;
using UnityEngine;

namespace OrbiterRL
{
    public sealed class CraftReport
    {
        /// <summary>The ascent's view: its first phases, everything above as payload.</summary>
        public CraftSpec Craft;
        /// <summary>Every propulsive phase of the rocket, with the final payload.</summary>
        public CraftSpec Full;
        public readonly List<string> Errors = new List<string>(), Warnings = new List<string>();
        /// <summary>KSP stage number (inverseStage) of every propulsive phase, in firing order.</summary>
        public int[] PhaseStages;
        public double FrontArea, SideArea, PadTwr;
        /// <summary>True when KSP's delta-v simulation has not produced results yet.</summary>
        public bool Pending;
        /// <summary>Raw stock stage data, logged for checking the phase reconstruction.</summary>
        public readonly List<object> StockStages = new List<object>();

        public bool Ok { get { return Errors.Count == 0; } }
        public int AscentStages { get { return Craft.Stages.Length; } }

        /// <summary>Latest phase whose KSP stage has been activated (-1 before launch).</summary>
        public int PhaseIndex(int currentStage)
        {
            int index = -1;
            for (int i = 0; i < PhaseStages.Length; i++) if (PhaseStages[i] >= currentStage) index = i;
            return index;
        }
    }

    public static class CraftAnalyzer
    {
        // Effective areas fitted to an actual KSP flight of the OrbiterRL Terrier
        // (orbiter/configs/orbiter_rl_terrier_aero.json). Other craft scale them
        // by their own drag-cube frontal and side areas relative to the Terrier's;
        // this is an estimate, not a fit.
        public const double FittedDragArea = 0.7447909856206689, FittedAlphaArea = 6.846020002846322,
                            FittedLiftArea = 4.434034202553299;
        // Drag-cube areas of that stack as measured by this analyzer in KSP 1.12.5
        // (frontal: largest part face along the nose of the single column; side: summed side faces).
        public const double ReferenceFrontArea = 0.9365001916885376, ReferenceSideArea = 16.961489975452423;

        public const double AttitudeRate = 8.0, AttitudeTau = 0.8;
        /// <summary>Vacuum delta-v below which the climb alone is unlikely to reach orbit.</summary>
        public const double OrbitDeltaV = 3800.0;
        /// <summary>Ideal delta-v a launch typically spends reaching a low Kerbin orbit (from flights).</summary>
        public const double AscentDeltaV = 3900.0;
        /// <summary>generalized_env.VehicleEnvelope.minimum_ideal_delta_v: every training rocket had this much.</summary>
        public const double TrainedDeltaV = 4800.0;

        private static readonly HashSet<string> Unsupported = new HashSet<string> { "SolidFuel", "IntakeAir", "XenonGas" };

        private static double Mass(Part part) { return (part.mass + part.GetResourceMass()) * 1000.0; }

        private static double PropellantMass(Part part, ICollection<string> propellants)
        {
            double total = 0;
            foreach (PartResource resource in part.Resources)
                if (propellants.Contains(resource.resourceName))
                    total += resource.amount * resource.info.density * 1000.0;
            return total;
        }

        public static CraftReport Analyze(Vessel vessel, GeneralizedAscentContract contract, double target)
        {
            CraftReport report = new CraftReport();
            List<ModuleEngines> engines = VesselInfo.Engines(vessel);
            if (engines.Count == 0) { report.Errors.Add("The rocket has no engines"); return report; }
            foreach (ModuleEngines e in engines)
                if (e.propellants.Any(p => Unsupported.Contains(p.name)))
                {
                    report.Errors.Add(e.part.partInfo.title + ": solid, jet and ion engines are not supported yet");
                    return report;
                }
            VesselDeltaV deltaV = vessel.VesselDeltaV;
            if (deltaV == null) { report.Errors.Add("KSP's delta-v calculation is unavailable for this vessel"); return report; }
            if (!deltaV.IsReady || deltaV.OperatingStageInfo == null || deltaV.OperatingStageInfo.Count == 0)
            {
                report.Pending = true;
                report.Errors.Add("KSP is still calculating the rocket's delta-v; try again in a moment");
                return report;
            }
            List<DeltaVStageInfo> phases = new List<DeltaVStageInfo>();
            for (int stage = vessel.currentStage - 1; stage >= 0; stage--)
            {
                DeltaVStageInfo info = deltaV.GetStage(stage);
                if (info == null) continue;
                report.StockStages.Add(new object[] { info.stage, info.startMass, info.endMass, info.dryMass, info.fuelMass,
                    info.decoupledMass, info.thrustVac, info.ispVac, info.ispASL, info.deltaVinVac, info.stageBurnTime });
                if (info.thrustVac > 0 && info.ispVac > 0 && (info.startMass - info.endMass) * 1000.0 > 1.0) phases.Add(info);
            }
            if (phases.Count == 0) { report.Errors.Add("No stage of this rocket burns fuel"); return report; }
            report.PhaseStages = phases.Select(p => p.stage).ToArray();

            StageSpec[] specs = new StageSpec[phases.Count];
            for (int i = 0; i < phases.Count; i++)
            {
                DeltaVStageInfo p = phases[i];
                double dropped = i + 1 < phases.Count ? (p.endMass - phases[i + 1].startMass) * 1000.0 : 0.0;
                specs[i] = new StageSpec
                {
                    DryMass = Math.Max(0.0, dropped), FuelMass = (p.startMass - p.endMass) * 1000.0,
                    ThrustVac = p.thrustVac * 1000.0, IspVac = p.ispVac, IspSl = p.ispASL,
                };
            }
            // The last phase keeps the pod and payload; only its engines and tanks are stage mass.
            DeltaVStageInfo last = phases[phases.Count - 1];
            HashSet<string> propellants = new HashSet<string>(last.enginesActiveInStage.Where(e => e.engine != null)
                .SelectMany(e => e.engine.propellants).Where(p => !p.ignoreForIsp).Select(p => p.name));
            HashSet<int> sections = new HashSet<int>(last.enginesActiveInStage.Where(e => e.engine != null)
                .Select(e => VesselInfo.DecoupleStage(e.engine.part)));
            double lastDry = 0;
            foreach (Part part in vessel.parts)
            {
                if (!sections.Contains(VesselInfo.DecoupleStage(part))) continue;
                double propellant = PropellantMass(part, propellants);
                if (part.FindModuleImplementing<ModuleEngines>() != null || propellant > 0) lastDry += Mass(part) - propellant;
            }
            double end = last.endMass * 1000.0;
            lastDry = Math.Min(lastDry, end);
            specs[specs.Length - 1].DryMass = lastDry;
            report.Full = new CraftSpec { Stages = specs, Payload = end - lastDry, Target = target,
                                          AttitudeRate = AttitudeRate, AttitudeTau = AttitudeTau };

            // The climb gets the fewest phases (at least two when the rocket has them)
            // that hold the delta-v the policy trained with; the rest is payload.
            int count = Math.Min(GeneralizedAscentContract.MaxStages, specs.Length);
            for (int k = Math.Min(2, specs.Length); k <= count; k++)
                if (Prefix(specs, report.Full.Payload, k, target).IdealDeltaV >= TrainedDeltaV) { count = k; break; }
            CraftSpec craft = Prefix(specs, report.Full.Payload, count, target);
            EstimateAero(vessel, craft, contract, report);
            report.Craft = craft;
            Screen(craft, contract, report);
            return report;
        }

        /// <summary>The first ``count`` phases, with every later phase's mass as payload.</summary>
        private static CraftSpec Prefix(StageSpec[] phases, double payload, int count, double target)
        {
            double above = payload;
            for (int j = count; j < phases.Length; j++) above += phases[j].DryMass + phases[j].FuelMass;
            return new CraftSpec
            {
                Stages = phases.Take(count).Select(s => new StageSpec
                {
                    DryMass = s.DryMass, FuelMass = s.FuelMass, ThrustVac = s.ThrustVac, IspVac = s.IspVac, IspSl = s.IspSl,
                }).ToArray(),
                Payload = above, Target = target, AttitudeRate = AttitudeRate, AttitudeTau = AttitudeTau,
            };
        }

        private static void EstimateAero(Vessel vessel, CraftSpec craft, GeneralizedAscentContract contract, CraftReport report)
        {
            // Frontal area: the largest face in each column of parts along the nose,
            // summed over the columns (one for a plain stack, more with side boosters).
            Vector3 nose = vessel.ReferenceTransform.up, side = vessel.ReferenceTransform.right;
            Vector3 origin = vessel.rootPart.transform.position;
            List<KeyValuePair<Vector3, double>> columns = new List<KeyValuePair<Vector3, double>>();
            double sideArea = 0;
            foreach (Part part in vessel.parts)
            {
                if (part.DragCubes == null || part.DragCubes.None) continue;
                double face = part.DragCubes.GetCubeAreaDir(part.transform.InverseTransformDirection(nose));
                sideArea += part.DragCubes.GetCubeAreaDir(part.transform.InverseTransformDirection(side));
                Vector3 offset = Vector3.ProjectOnPlane(part.transform.position - origin, nose);
                int match = columns.FindIndex(c => (c.Key - offset).magnitude < 0.3f);
                if (match < 0) columns.Add(new KeyValuePair<Vector3, double>(offset, face));
                else if (face > columns[match].Value) columns[match] = new KeyValuePair<Vector3, double>(columns[match].Key, face);
            }
            // Fins and other small radial parts barely add frontal area; skip their columns.
            double largest = columns.Count > 0 ? columns.Max(c => c.Value) : 0.0;
            double front = columns.Where(c => c.Value >= 0.25 * largest).Sum(c => c.Value);
            report.FrontArea = front;
            report.SideArea = sideArea;
            double tonnes = craft.InitialMass / 1000;
            if (front <= 0 || sideArea <= 0)
            {
                report.Warnings.Add("No drag-cube data; using mid-range aerodynamics");
                front = ReferenceFrontArea;
                sideArea = ReferenceSideArea;
            }
            craft.DragArea = tonnes * contract.Clamp("drag_area_per_tonne", FittedDragArea * front / ReferenceFrontArea / tonnes);
            craft.DragAlphaArea = tonnes * contract.Clamp("drag_alpha_area_per_tonne", FittedAlphaArea * sideArea / ReferenceSideArea / tonnes);
            craft.LiftArea = tonnes * contract.Clamp("lift_area_per_tonne", FittedLiftArea * sideArea / ReferenceSideArea / tonnes);
        }

        /// <summary>Compare with the training distribution; far outside it the policy has never flown.</summary>
        private static void Screen(CraftSpec craft, GeneralizedAscentContract contract, CraftReport report)
        {
            double m0 = craft.InitialMass;
            StageSpec first = craft.Stages[0];
            report.PadTwr = first.ThrustVac * first.IspSl / first.IspVac / m0 / Units.G0;
            if (report.PadTwr < 1.05) report.Errors.Add(string.Format(CultureInfo.InvariantCulture,
                "Launch thrust-to-weight is {0:0.00}; it must be above 1.05", report.PadTwr));
            double dv = craft.IdealDeltaV;
            if (dv < OrbitDeltaV) report.Errors.Add(string.Format(CultureInfo.InvariantCulture,
                "Only {0:0} m/s of vacuum delta-v for the climb; Kerbin orbit needs about 3,800 m/s or more", dv));
            Warn(report, "launch thrust-to-weight", report.PadTwr, contract, "initial_twr");
            for (int i = 0; i < craft.Stages.Length; i++)
            {
                StageSpec s = craft.Stages[i];
                double upper = craft.Payload;
                for (int j = i + 1; j < craft.Stages.Length; j++) upper += craft.Stages[j].DryMass + craft.Stages[j].FuelMass;
                Warn(report, "stage " + (i + 1) + " Isp", s.IspVac, contract, "isp");
                Warn(report, "stage " + (i + 1) + " dry fraction", s.DryMass / (s.DryMass + s.FuelMass), contract, "stage_dry_fraction");
                if (i > 0) Warn(report, "stage " + (i + 1) + " thrust-to-weight", s.ThrustVac / ((upper + s.DryMass + s.FuelMass) * Units.G0), contract, "upper_twr");
            }
            Warn(report, "payload mass (kg)", craft.Payload, contract, "payload");
            if (dv < TrainedDeltaV) report.Warnings.Add(string.Format(CultureInfo.InvariantCulture,
                "{0:0} m/s delta-v for the climb is below the {1:0} m/s the policy trained with", dv, TrainedDeltaV));
        }

        private static void Warn(CraftReport report, string label, double value, GeneralizedAscentContract contract, string name)
        {
            double low = contract.Range(name, false), high = contract.Range(name, true);
            if (value < low * 0.9 || value > high * 1.1)
                report.Warnings.Add(string.Format(CultureInfo.InvariantCulture, "{0} {1:0.##} is outside the trained range {2:0.##}-{3:0.##}",
                                                  label, value, low, high));
        }
    }
}
