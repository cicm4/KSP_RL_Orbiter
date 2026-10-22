// Reads an arbitrary KSP rocket into the generalized policy's CraftSpec:
// up to four sequential liquid stages (engines igniting together, dropped
// with their tanks), plus payload and estimated planar aerodynamics.
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
        public CraftSpec Craft;
        public readonly List<string> Errors = new List<string>(), Warnings = new List<string>();
        /// <summary>KSP activation stage (inverseStage) of each propulsion stage, in firing order.</summary>
        public int[] ActivationStages;
        public double FrontArea, SideArea, PadTwr;
        public readonly List<List<Part>> StageParts = new List<List<Part>>();
        public readonly List<HashSet<string>> StagePropellants = new List<HashSet<string>>();

        /// <summary>Propellant (kg) left in a stage's own tanks that are still attached.</summary>
        public double StageFuel(Vessel vessel, int stage)
        {
            double total = 0;
            foreach (Part part in StageParts[stage])
                if (part != null && part.vessel == vessel)
                    foreach (PartResource resource in part.Resources)
                        if (StagePropellants[stage].Contains(resource.resourceName))
                            total += resource.amount * resource.info.density * 1000.0;
            return total;
        }

        public bool Ok { get { return Errors.Count == 0; } }
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
        // (frontal: largest part face along the nose; side: summed side faces).
        public const double ReferenceFrontArea = 0.9365001916885376, ReferenceSideArea = 16.961489975452423;

        public const double AttitudeRate = 8.0, AttitudeTau = 0.8;

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
            if (vessel.parts.Any(p => p.Modules.Contains("CModuleFuelLine")))
                report.Errors.Add("Fuel lines (asparagus staging) are not supported");
            foreach (ModuleEngines e in engines)
                if (e.propellants.Any(p => Unsupported.Contains(p.name)))
                    report.Errors.Add(e.part.partInfo.title + ": solid, jet and ion engines are not supported");
            List<IGrouping<int, ModuleEngines>> groups = engines.GroupBy(e => e.part.inverseStage)
                .OrderByDescending(g => g.Key).ToList();
            if (groups.Count > GeneralizedAscentContract.MaxStages)
                report.Errors.Add("At most " + GeneralizedAscentContract.MaxStages + " engine stages are supported, found " + groups.Count);
            List<StageSpec> stages = new List<StageSpec>();
            HashSet<Part> counted = new HashSet<Part>();
            int previousDecouple = int.MaxValue;
            foreach (IGrouping<int, ModuleEngines> group in groups)
            {
                int[] decouple = group.Select(e => VesselInfo.DecoupleStage(e.part)).Distinct().ToArray();
                if (decouple.Length != 1)
                {
                    report.Errors.Add("Engines that ignite together must drop together (no parallel boosters)");
                    break;
                }
                int section = decouple[0];
                if (section >= previousDecouple)
                {
                    report.Errors.Add("Stages must fire and drop in order from the bottom up");
                    break;
                }
                previousDecouple = section;
                HashSet<string> propellants = new HashSet<string>(group.SelectMany(e => e.propellants)
                    .Where(p => !p.ignoreForIsp).Select(p => p.name));
                bool last = section < 0 || group == groups[groups.Count - 1];
                double dry = 0, fuel = 0;
                List<Part> members = new List<Part>();
                foreach (Part part in vessel.parts)
                {
                    if (counted.Contains(part) || VesselInfo.DecoupleStage(part) != section) continue;
                    double propellant = PropellantMass(part, propellants);
                    bool propulsive = part.FindModuleImplementing<ModuleEngines>() != null || propellant > 0;
                    // The final stage keeps the pod and payload; only its engines and tanks are stage mass.
                    if (last && !propulsive) continue;
                    counted.Add(part);
                    members.Add(part);
                    fuel += propellant;
                    dry += Mass(part) - propellant;
                }
                // Combined Isp is the thrust-weighted harmonic mean (total thrust / total flow),
                // in vacuum and at 1 atm, where each engine's thrust scales with its Isp.
                double thrust = 0, flowVac = 0, slThrust = 0, slFlow = 0;
                foreach (ModuleEngines e in group)
                {
                    double vac = e.MaxThrustOutputVac(true) * 1000.0;
                    double ispVac = e.atmosphereCurve.Evaluate(0f), ispSl = e.atmosphereCurve.Evaluate(1f);
                    if (vac <= 0 || ispVac <= 0) continue;
                    thrust += vac;
                    flowVac += vac / ispVac;
                    double sl = vac * ispSl / ispVac;
                    slThrust += sl;
                    if (ispSl > 0) slFlow += sl / ispSl;
                }
                double ispV = flowVac > 0 ? thrust / flowVac : 0;
                double ispS = slFlow > 0 ? slThrust / slFlow : 0;
                if (fuel <= 0 || thrust <= 0)
                    report.Errors.Add("Engine stage " + (stages.Count + 1) + " has no fuel or thrust");
                stages.Add(new StageSpec { DryMass = dry, FuelMass = fuel, ThrustVac = thrust, IspVac = ispV, IspSl = ispS });
                report.StageParts.Add(members);
                report.StagePropellants.Add(propellants);
            }
            report.ActivationStages = groups.Select(g => g.Key).ToArray();
            if (!report.Ok) return report;
            double total = vessel.parts.Sum(p => Mass(p));
            double payload = total - stages.Sum(s => s.DryMass + s.FuelMass);
            CraftSpec craft = new CraftSpec
            {
                Stages = stages.ToArray(), Payload = payload, Target = target,
                AttitudeRate = AttitudeRate, AttitudeTau = AttitudeTau,
            };
            EstimateAero(vessel, craft, contract, report);
            report.Craft = craft;
            Screen(craft, contract, report);
            return report;
        }

        private static void EstimateAero(Vessel vessel, CraftSpec craft, GeneralizedAscentContract contract, CraftReport report)
        {
            Vector3 nose = vessel.ReferenceTransform.up, side = vessel.ReferenceTransform.right;
            double front = 0, sideArea = 0;
            foreach (Part part in vessel.parts)
            {
                if (part.DragCubes == null || part.DragCubes.None) continue;
                front = Math.Max(front, part.DragCubes.GetCubeAreaDir(part.transform.InverseTransformDirection(nose)));
                sideArea += part.DragCubes.GetCubeAreaDir(part.transform.InverseTransformDirection(side));
            }
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
            if (dv < 3800) report.Errors.Add(string.Format(CultureInfo.InvariantCulture,
                "Only {0:0} m/s of vacuum delta-v; Kerbin orbit needs about 3,800 m/s or more", dv));
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
            if (dv < 4800) report.Warnings.Add(string.Format(CultureInfo.InvariantCulture,
                "{0:0} m/s delta-v is below the 4,800 m/s the policy trained with", dv));
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
