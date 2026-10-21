// Direct-child transfer search (orbiter/mission.py plan_child_transfer) and
// the inbound periapsis correction (mission_nodes.correct_periapsis_node).
// A plan is a candidate; the KSP adapter must confirm it with patched conics.
using System;
using System.Collections.Generic;

namespace OrbiterRL.Core
{
    public interface IEphemeris
    {
        /// <summary>Target position/velocity relative to the departure body.</summary>
        void State(double ut, out V3 position, out V3 velocity);
    }

    /// <summary>Kepler ephemeris from catalog elements (system.py SystemCatalog.state).</summary>
    public sealed class KeplerEphemeris : IEphemeris
    {
        private readonly double mu, a, e, meanAtEpoch, epoch;
        private readonly V3 p, q;

        public KeplerEphemeris(double parentMu, double semiMajorAxis, double eccentricity,
                               double meanAnomalyAtEpoch, double epochUt, V3 basisP, V3 basisQ)
        {
            if (!(eccentricity >= 0 && eccentricity < 1) || !(semiMajorAxis > 0))
                throw new ArgumentException("Catalog ephemerides require bound elliptic body orbits");
            mu = parentMu; a = semiMajorAxis; e = eccentricity;
            meanAtEpoch = meanAnomalyAtEpoch; epoch = epochUt; p = basisP; q = basisQ;
        }

        public void State(double ut, out V3 position, out V3 velocity)
        {
            double mean = Mod(meanAtEpoch + Math.Sqrt(mu / (a * a * a)) * (ut - epoch), 2 * Math.PI);
            double anomaly = e < 0.8 ? mean : Math.PI;
            bool converged = false;
            for (int i = 0; i < 30; i++)
            {
                double difference = (anomaly - e * Math.Sin(anomaly) - mean) / (1 - e * Math.Cos(anomaly));
                anomaly -= difference;
                if (Math.Abs(difference) < 1e-13) { converged = true; break; }
            }
            if (!converged) throw new ArgumentException("Catalog eccentric anomaly did not converge");
            double cosine = Math.Cos(anomaly), sine = Math.Sin(anomaly), beta = Math.Sqrt(1 - e * e);
            double radius = a * (1 - e * cosine);
            position = a * ((cosine - e) * p + beta * sine * q);
            velocity = Math.Sqrt(mu * a) / radius * (-sine * p + beta * cosine * q);
        }

        /// <summary>Python's float modulo (result has the divisor's sign).</summary>
        private static double Mod(double x, double m)
        {
            double r = x % m;
            return r < 0 ? r + m : r;
        }
    }

    public sealed class TransferPlan
    {
        public double DepartureUt, ArrivalUt, ArrivalVinf, CaptureDeltaV, TargetAltitude, TargetPeriapsisRadius;
        public V3 DeparturePosition, DepartureVelocity, DeltaV, AimOffset;
        public bool LongWay;

        public double DepartureDeltaV { get { return DeltaV.Norm; } }
        public double TotalDeltaV { get { return DepartureDeltaV + CaptureDeltaV; } }
    }

    public sealed class Body
    {
        public double Mu, Radius, AtmosphereDepth, SphereOfInfluence, SemiMajorAxis;
    }

    public static class Planner
    {
        public static double CaptureDeltaV(double mu, double periapsis, double vinf)
        {
            if (!(mu > 0) || !(periapsis > 0) || !(vinf >= 0))
                throw new ArgumentException("Invalid capture budget inputs");
            return Math.Sqrt(vinf * vinf + 2 * mu / periapsis) - Math.Sqrt(mu / periapsis);
        }

        /// <summary>numpy.linspace(start, stop, count) including the endpoint.</summary>
        public static double[] Linspace(double start, double stop, int count)
        {
            double[] values = new double[count];
            double step = (stop - start) / (count - 1);
            for (int i = 0; i < count; i++) values[i] = start + i * step;
            values[count - 1] = stop;
            return values;
        }

        private struct Candidate
        {
            public double Cost, DepartureUt, ArrivalUt;
            public V3 Start, StartVelocity;
            public bool LongWay;
            public int Order;
        }

        /// <summary>
        /// plan_child_transfer for a spacecraft orbiting the target's parent.
        /// ``wait`` is the departure search span (one parking period for moons).
        /// </summary>
        public static TransferPlan PlanChildTransfer(Body source, Body target, IEphemeris ephemeris,
                                                     V3 position, V3 velocity, double epoch, double targetAltitude,
                                                     double leadTime = 180.0, int departureSamples = 48,
                                                     int durationSamples = 20)
        {
            double mu = source.Mu, targetMu = target.Mu;
            double periapsis = target.Radius + targetAltitude;
            if (targetAltitude <= target.AtmosphereDepth + 1000 || periapsis >= 0.8 * target.SphereOfInfluence)
                throw new ArgumentException("Target orbit must be safely above the atmosphere and inside its SOI");
            double energy, eccentricity, pe, ap;
            Kepler.Elements(position, velocity, mu, out energy, out eccentricity, out pe, out ap);
            if (energy >= 0) throw new ArgumentException("Departure state must be a bound parking orbit");
            if (pe < source.Radius + source.AtmosphereDepth + 1000)
                throw new ArgumentException("Departure parking orbit must clear the surface and atmosphere");
            double semiMajor = -mu / (2 * energy);
            double period = 2 * Math.PI * Math.Sqrt(semiMajor * semiMajor * semiMajor / mu);
            double hohmann = Math.PI * Math.Sqrt(Math.Pow((semiMajor + target.SemiMajorAxis) / 2, 3) / mu);
            double wait = period;
            List<Candidate> candidates = new List<Candidate>();
            double[] durations = Linspace(0.7 * hohmann, 1.65 * hohmann, durationSamples);
            foreach (double departureUt in Linspace(epoch + leadTime, epoch + leadTime + wait, departureSamples))
            {
                V3 start, startVelocity;
                Kepler.Propagate(position, velocity, departureUt - epoch, mu, out start, out startVelocity);
                foreach (double duration in durations)
                {
                    double arrivalUt = departureUt + duration;
                    V3 finish, finishVelocity;
                    ephemeris.State(arrivalUt, out finish, out finishVelocity);
                    foreach (bool longWay in new[] { false, true })
                    {
                        try
                        {
                            V3 departure, arrival;
                            Kepler.Lambert(start, finish, duration, mu, longWay, out departure, out arrival);
                            double vinf = (arrival - finishVelocity).Norm;
                            double cost = (departure - startVelocity).Norm + CaptureDeltaV(targetMu, periapsis, vinf);
                            if (double.IsNaN(cost)) continue;
                            candidates.Add(new Candidate
                            {
                                Cost = cost, DepartureUt = departureUt, ArrivalUt = arrivalUt, Start = start,
                                StartVelocity = startVelocity, LongWay = longWay, Order = candidates.Count,
                            });
                        }
                        catch (ArgumentException) { }
                    }
                }
            }
            if (candidates.Count == 0)
                throw new ArgumentException("No finite zero-revolution Lambert candidate in the search window");
            // Python's sorted() is stable; keep insertion order on equal cost.
            candidates.Sort((x, y) => x.Cost != y.Cost ? x.Cost.CompareTo(y.Cost) : x.Order.CompareTo(y.Order));
            TransferPlan best = null;
            for (int index = 0; index < Math.Min(12, candidates.Count); index++)
            {
                Candidate c = candidates[index];
                V3 finish, bodyVelocity;
                ephemeris.State(c.ArrivalUt, out finish, out bodyVelocity);
                V3 plane = V3.Cross(finish, bodyVelocity).Unit();
                foreach (int side in new[] { -1, 1 })
                {
                    V3 offset = V3.Zero, departure, arrival;
                    for (int i = 0; i < 3; i++)
                    {
                        Kepler.Lambert(c.Start, finish + offset, c.ArrivalUt - c.DepartureUt, mu, c.LongWay,
                                       out departure, out arrival);
                        V3 relative = arrival - bodyVelocity;
                        double vinf0 = relative.Norm;
                        double impact = periapsis * Math.Sqrt(1 + 2 * targetMu / (periapsis * vinf0 * vinf0));
                        offset = side * impact * V3.Cross(relative, plane).Unit();
                    }
                    Kepler.Lambert(c.Start, finish + offset, c.ArrivalUt - c.DepartureUt, mu, c.LongWay,
                                   out departure, out arrival);
                    double vinf = (arrival - bodyVelocity).Norm;
                    TransferPlan plan = new TransferPlan
                    {
                        DepartureUt = c.DepartureUt, ArrivalUt = c.ArrivalUt, DeparturePosition = c.Start,
                        DepartureVelocity = c.StartVelocity, DeltaV = departure - c.StartVelocity,
                        ArrivalVinf = vinf, CaptureDeltaV = CaptureDeltaV(targetMu, periapsis, vinf),
                        TargetAltitude = targetAltitude, TargetPeriapsisRadius = periapsis,
                        AimOffset = offset, LongWay = c.LongWay,
                    };
                    // min() keeps the first of equal totals.
                    if (best == null || plan.TotalDeltaV < best.TotalDeltaV) best = plan;
                }
            }
            return best;
        }

        /// <summary>Departure velocity of the plan's branch aimed at a scaled offset.</summary>
        public static V3 ScaledDeparture(TransferPlan plan, IEphemeris ephemeris, double mu, double scale)
        {
            V3 target, unused, departure, arrival;
            ephemeris.State(plan.ArrivalUt, out target, out unused);
            Kepler.Lambert(plan.DeparturePosition, target + scale * plan.AimOffset,
                           plan.ArrivalUt - plan.DepartureUt, mu, plan.LongWay, out departure, out arrival);
            return departure - plan.DepartureVelocity;
        }

        /// <summary>
        /// Tangential correction after lead time so the inbound conic reaches
        /// periapsis at ``targetRadius``; null below 0.02 m/s.
        /// </summary>
        public static V3? PeriapsisCorrection(V3 r0, V3 v0, double mu, double bodyRadius,
                                              double targetRadius, double leadTime)
        {
            V3 r, v;
            Kepler.Propagate(r0, v0, leadTime, mu, out r, out v);
            double radius = r.Norm;
            V3 radial = r / radius;
            double vr = V3.Dot(v, radial);
            V3 tangential = v - vr * radial;
            double speed = tangential.Norm;
            if (vr >= 0 || targetRadius >= radius || targetRadius <= bodyRadius)
                throw new ArgumentException("Periapsis correction requires a safe inbound approach outside the target radius");
            double desired = targetRadius * Math.Sqrt((vr * vr + 2 * mu * (1 / targetRadius - 1 / radius))
                                                      / (radius * radius - targetRadius * targetRadius));
            V3 delta = (desired - speed) * tangential / speed;
            if (delta.Norm < 0.02) return null;
            return delta;
        }

        /// <summary>Time until the inbound radius reaches ``radius`` (60-step bisection).</summary>
        public static double IncomingRadiusTime(V3 r, V3 v, double mu, double radius, double timeToPeriapsis)
        {
            if (!(timeToPeriapsis > 0) || double.IsInfinity(timeToPeriapsis) || V3.Dot(r, v) >= 0)
                throw new ArgumentException("Need a finite incoming coast before periapsis");
            if (r.Norm <= radius) return 0.0;
            double low = 0.0, high = timeToPeriapsis;
            V3 pr, pv;
            Kepler.Propagate(r, v, high, mu, out pr, out pv);
            if (pr.Norm > radius) throw new ArgumentException("Inbound conic never reaches the capture radius");
            for (int i = 0; i < 60; i++)
            {
                double middle = (low + high) / 2;
                Kepler.Propagate(r, v, middle, mu, out pr, out pv);
                if (pr.Norm > radius) low = middle; else high = middle;
            }
            return (low + high) / 2;
        }
    }
}
