// Native telemetry with the semantics of the kRPC calls the policies were
// validated with. All masses are kg, thrusts N, pressures Pa.
using System;
using System.Collections.Generic;
using System.Linq;
using OrbiterRL.Core;
using UnityEngine;

namespace OrbiterRL
{
    public static class Frames
    {
        // KSP's Orbit class works in a right-handed frame whose axes are the
        // Unity world axes swizzled xzy (ManeuverNode.GetBurnVector applies
        // .xzy to orbit-frame velocities). The swizzle is its own inverse.
        public static V3 ToCore(Vector3d world) { return new V3(world.x, world.z, world.y); }
        public static Vector3d ToWorld(V3 core) { return new Vector3d(core.X, core.Z, core.Y); }
        public static V3 FromOrbit(Vector3d zup) { return new V3(zup.x, zup.y, zup.z); }

        /// <summary>Vessel CoM relative to the main body, core frame.</summary>
        public static V3 Position(Vessel vessel) { return ToCore(vessel.CoMD - vessel.mainBody.position); }

        /// <summary>Inertial (orbital) velocity, core frame.</summary>
        public static V3 Velocity(Vessel vessel) { return ToCore(vessel.obt_velocity); }

        public static Vector3d Nose(Vessel vessel) { return vessel.ReferenceTransform.up; }
    }

    public static class VesselInfo
    {
        public const string LiquidFuel = "LiquidFuel", Oxidizer = "Oxidizer";

        public static double Mass(Vessel vessel) { return vessel.totalMass * 1000.0; }

        public static float PressureAtm(Vessel vessel)
        {
            return (float)(vessel.staticPressurekPa * PhysicsGlobals.KpaToAtmospheres);
        }

        public static List<ModuleEngines> Engines(Vessel vessel)
        {
            return vessel.FindPartModulesImplementing<ModuleEngines>();
        }

        /// <summary>kRPC Engine.active: ignited, including a flamed-out engine (depletion detection needs it).</summary>
        public static bool Active(ModuleEngines engine) { return engine.EngineIgnited; }

        public static bool HasFuel(ModuleEngines engine)
        {
            if (engine.flameout) return false;
            foreach (Propellant propellant in engine.propellants)
                if (!propellant.ignoreForIsp && propellant.totalResourceAvailable <= 0) return false;
            return true;
        }

        /// <summary>kRPC Engine.available_thrust: full-throttle thrust now, with limiter; 0 without fuel.</summary>
        public static double AvailableThrust(ModuleEngines engine, Vessel vessel)
        {
            if (!HasFuel(engine)) return 0.0;
            return 1000.0 * engine.MaxThrustOutputAtm(true, true, PressureAtm(vessel),
                                                       vessel.atmosphericTemperature, vessel.atmDensity);
        }

        public static double AvailableThrust(Vessel vessel)
        {
            double total = 0;
            foreach (ModuleEngines engine in Engines(vessel))
                if (Active(engine)) total += AvailableThrust(engine, vessel);
            return total;
        }

        /// <summary>kRPC Vessel.specific_impulse: thrust-weighted Isp of active engines.</summary>
        public static double SpecificImpulse(Vessel vessel)
        {
            double thrust = 0, flow = 0;
            float pressure = PressureAtm(vessel);
            foreach (ModuleEngines engine in Engines(vessel))
            {
                if (!Active(engine)) continue;
                double f = AvailableThrust(engine, vessel);
                double isp = engine.atmosphereCurve.Evaluate(pressure);
                if (f > 0 && isp > 0) { thrust += f; flow += f / isp; }
            }
            return flow > 0 ? thrust / flow : 0.0;
        }

        private static double Density(string name)
        {
            PartResourceDefinition definition = PartResourceLibrary.Instance.GetDefinition(name);
            return definition == null ? 0.0 : definition.density * 1000.0;
        }

        private static double Amount(Part part, string name)
        {
            PartResource resource = part.Resources.Get(name);
            return resource == null ? 0.0 : resource.amount;
        }

        /// <summary>Stage that decouples ``part`` (first decoupler on the path to the root), or -1.</summary>
        public static int DecoupleStage(Part part)
        {
            for (Part current = part; current != null; current = current.parent)
                if (current.FindModuleImplementing<ModuleDecouple>() != null ||
                    current.FindModuleImplementing<ModuleAnchoredDecoupler>() != null)
                    return current.inverseStage;
            return -1;
        }

        /// <summary>kg of LiquidFuel+Oxidizer in parts decoupled at ``stage`` (kRPC cumulative=False).</summary>
        public static double StagePropellant(Vessel vessel, int stage)
        {
            double lf = Density(LiquidFuel), ox = Density(Oxidizer), total = 0;
            foreach (Part part in vessel.parts)
                if (DecoupleStage(part) == stage)
                    total += Amount(part, LiquidFuel) * lf + Amount(part, Oxidizer) * ox;
            return total;
        }

        /// <summary>mission_nodes.read_parking_state dry mass: wet mass minus burnable 9:11 LF/Ox.</summary>
        public static double DryMass(Vessel vessel)
        {
            double lf = 0, ox = 0;
            foreach (Part part in vessel.parts) { lf += Amount(part, LiquidFuel); ox += Amount(part, Oxidizer); }
            double usable = Math.Min(lf / 9.0, ox / 11.0) * (9.0 * Density(LiquidFuel) + 11.0 * Density(Oxidizer));
            return Mass(vessel) - usable;
        }

        /// <summary>
        /// Wet mass minus the propellant the running engines can still burn: the
        /// propellant types they use, in their own (not yet decoupled) section.
        /// </summary>
        public static double ReachableDryMass(Vessel vessel)
        {
            List<ModuleEngines> active = Engines(vessel).Where(Active).ToList();
            if (active.Count == 0) return Mass(vessel);
            HashSet<int> sections = new HashSet<int>(active.Select(e => DecoupleStage(e.part)));
            HashSet<string> names = new HashSet<string>(active.SelectMany(e => e.propellants).Where(p => !p.ignoreForIsp).Select(p => p.name));
            double propellant = 0;
            foreach (Part part in vessel.parts)
                if (sections.Contains(DecoupleStage(part)))
                    foreach (PartResource resource in part.Resources)
                        if (names.Contains(resource.resourceName)) propellant += resource.amount * resource.info.density * 1000.0;
            return Mass(vessel) - propellant;
        }

        public static double SurfaceAltitude(Vessel vessel)
        {
            double terrain = vessel.heightFromTerrain;
            return terrain >= 0 ? Math.Min(terrain, vessel.altitude) : vessel.altitude;
        }

        /// <summary>Pitch of the nose above the local horizon, degrees.</summary>
        public static double Pitch(Vessel vessel)
        {
            Vector3d up = (vessel.CoMD - vessel.mainBody.position).normalized;
            return 90.0 - Vector3d.Angle(Frames.Nose(vessel), up);
        }

        /// <summary>Unit east vector (world) from the body's surface rotation velocity.</summary>
        public static Vector3d East(Vessel vessel)
        {
            Vector3d up = (vessel.CoMD - vessel.mainBody.position).normalized;
            Vector3d rotation = vessel.mainBody.getRFrmVel(vessel.CoMD);
            Vector3d east = Vector3d.Exclude(up, rotation);
            if (east.magnitude < 1e-6) throw new InvalidOperationException("East is undefined at the pole");
            return east.normalized;
        }

        public static string Describe(IEnumerable<string> names)
        {
            return string.Join(",", names.OrderBy(x => x, StringComparer.Ordinal).ToArray());
        }
    }
}
