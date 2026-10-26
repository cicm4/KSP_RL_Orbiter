// Direct-child transfer search (orbiter/mission.py plan_child_transfer), the
// inbound periapsis correction (mission_nodes.correct_periapsis_node), and
// planet-to-planet windows, escape burns and course corrections.
// A plan is a candidate; the KSP adapter must confirm it with patched conics.
using System;
using System.Collections.Generic;
using System.Linq;

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

    public sealed class InterplanetaryPlan
    {
        public double DepartureUt, ArrivalUt, ArrivalVinf, EjectionDeltaV, CaptureDeltaV, TargetAltitude, TargetPeriapsisRadius;
        /// <summary>Heliocentric departure velocity minus the origin body's velocity.</summary>
        public V3 DepartureVinf;
        public bool LongWay;

        public double TotalDeltaV { get { return EjectionDeltaV + CaptureDeltaV; } }
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
                TransferPlan plan = Aimed(c.Start, c.StartVelocity, c.DepartureUt, c.ArrivalUt, c.LongWay, ephemeris,
                                          mu, targetMu, periapsis, targetAltitude);
                // min() keeps the first of equal totals.
                if (best == null || plan.TotalDeltaV < best.TotalDeltaV) best = plan;
            }
            return best;
        }

        /// <summary>
        /// The cheaper of the two B-plane sides: aim the transfer at an offset from
        /// the target so the arrival hyperbola's periapsis is ``periapsis``.
        /// </summary>
        private static TransferPlan Aimed(V3 start, V3 startVelocity, double departureUt, double arrivalUt, bool longWay,
                                          IEphemeris ephemeris, double mu, double targetMu, double periapsis,
                                          double targetAltitude)
        {
            V3 finish, bodyVelocity;
            ephemeris.State(arrivalUt, out finish, out bodyVelocity);
            V3 plane = V3.Cross(finish, bodyVelocity).Unit();
            TransferPlan best = null;
            foreach (int side in new[] { -1, 1 })
            {
                V3 offset = V3.Zero, departure, arrival;
                for (int i = 0; i < 3; i++)
                {
                    Kepler.Lambert(start, finish + offset, arrivalUt - departureUt, mu, longWay, out departure, out arrival);
                    V3 relative = arrival - bodyVelocity;
                    double vinf0 = relative.Norm;
                    double impact = periapsis * Math.Sqrt(1 + 2 * targetMu / (periapsis * vinf0 * vinf0));
                    offset = side * impact * V3.Cross(relative, plane).Unit();
                }
                Kepler.Lambert(start, finish + offset, arrivalUt - departureUt, mu, longWay, out departure, out arrival);
                double vinf = (arrival - bodyVelocity).Norm;
                TransferPlan plan = new TransferPlan
                {
                    DepartureUt = departureUt, ArrivalUt = arrivalUt, DeparturePosition = start,
                    DepartureVelocity = startVelocity, DeltaV = departure - startVelocity,
                    ArrivalVinf = vinf, CaptureDeltaV = CaptureDeltaV(targetMu, periapsis, vinf),
                    TargetAltitude = targetAltitude, TargetPeriapsisRadius = periapsis,
                    AimOffset = offset, LongWay = longWay,
                };
                if (best == null || plan.TotalDeltaV < best.TotalDeltaV) best = plan;
            }
            return best;
        }

        /// <summary>Burn from a circular parking orbit of ``radius`` to a hyperbolic excess speed.</summary>
        public static double EjectionDeltaV(double mu, double radius, double vinf)
        {
            return Math.Sqrt(vinf * vinf + 2 * mu / radius) - Math.Sqrt(mu / radius);
        }

        /// <summary>
        /// EjectionBurn's cost from a circular orbit with the given normal: the
        /// hyperbola must contain the burn radius and the excess velocity, so a
        /// departure out of the orbit plane (declination) tilts it and the burn
        /// pays for the plane change. Unreachable asymptotes cost the full speed sum.
        /// </summary>
        public static double EjectionDeltaV(double mu, double radius, V3 vinf, V3 parkingNormal)
        {
            double speed2 = V3.Dot(vinf, vinf);
            double hyperbolic = Math.Sqrt(speed2 + 2 * mu / radius), circular = Math.Sqrt(mu / radius);
            double cosNu = -1 / (1 + radius * speed2 / mu);
            double sinDec = Math.Min(1.0, Math.Abs(V3.Dot(vinf.Unit(), parkingNormal.Unit())));
            double cosDec = Math.Sqrt(1 - sinDec * sinDec);
            if (cosDec < Math.Abs(cosNu) + 1e-12) return hyperbolic + circular;
            double cosPhi = cosNu / cosDec, sinPhi = Math.Sqrt(Math.Max(0.0, 1 - cosPhi * cosPhi));
            double cosTilt = cosDec * sinPhi / Math.Sqrt(sinDec * sinDec + cosDec * cosDec * sinPhi * sinPhi);
            return Math.Sqrt(Math.Max(0.0, hyperbolic * hyperbolic + circular * circular - 2 * hyperbolic * circular * cosTilt));
        }

        /// <summary>
        /// Launch-window search between sibling bodies (mission.plan_interplanetary_window):
        /// a departure/flight-time Lambert grid, refined around the cheapest cell, with
        /// the Oberth ejection from the parking orbit plus the capture as the cost. With
        /// ``parkingNormal`` the ejection also pays for leaving out of the orbit plane.
        /// Ephemerides give each body's state relative to their common parent.
        /// </summary>
        public static InterplanetaryPlan PlanInterplanetary(Body origin, Body target, IEphemeris originEphemeris,
                                                            IEphemeris targetEphemeris, double parentMu, double epoch,
                                                            double parkingRadius, double targetAltitude, double span,
                                                            int departureSamples = 48, int durationSamples = 24,
                                                            V3? parkingNormal = null)
        {
            double periapsis = target.Radius + targetAltitude;
            if (targetAltitude <= target.AtmosphereDepth + 1000 || periapsis >= 0.8 * target.SphereOfInfluence)
                throw new ArgumentException("Target orbit must be safely above the atmosphere and inside its SOI");
            if (parkingRadius <= origin.Radius + origin.AtmosphereDepth + 1000 || parkingRadius >= origin.SphereOfInfluence)
                throw new ArgumentException("Parking orbit must be above the atmosphere and inside the SOI");
            if (!(span > 0) || departureSamples < 2 || durationSamples < 2)
                throw new ArgumentException("Need a positive search span and at least two samples");
            double nominal = Math.PI * Math.Sqrt(Math.Pow((origin.SemiMajorAxis + target.SemiMajorAxis) / 2, 3) / parentMu);
            InterplanetaryPlan best = null;
            Func<double, double, bool, InterplanetaryPlan> evaluate = (departure, duration, longWay) =>
            {
                V3 r1, v1, r2, v2, leave, arrive;
                originEphemeris.State(departure, out r1, out v1);
                targetEphemeris.State(departure + duration, out r2, out v2);
                try { Kepler.Lambert(r1, r2, duration, parentMu, longWay, out leave, out arrive); }
                catch (ArgumentException) { return null; }
                V3 excess = leave - v1;
                double vinf = (arrive - v2).Norm;
                InterplanetaryPlan plan = new InterplanetaryPlan
                {
                    DepartureUt = departure, ArrivalUt = departure + duration, DepartureVinf = excess, ArrivalVinf = vinf,
                    EjectionDeltaV = parkingNormal.HasValue
                        ? EjectionDeltaV(origin.Mu, parkingRadius, excess, parkingNormal.Value)
                        : EjectionDeltaV(origin.Mu, parkingRadius, excess.Norm),
                    CaptureDeltaV = CaptureDeltaV(target.Mu, periapsis, vinf), LongWay = longWay,
                    TargetAltitude = targetAltitude, TargetPeriapsisRadius = periapsis,
                };
                return double.IsNaN(plan.TotalDeltaV) ? null : plan;
            };
            // Cheapest arc for one departure time over the flight-time grid.
            Func<double, double[], InterplanetaryPlan> cheapest = (departure, durations) =>
            {
                InterplanetaryPlan found = null;
                foreach (double duration in durations)
                    foreach (bool longWay in new[] { false, true })
                    {
                        InterplanetaryPlan plan = evaluate(departure, duration, longWay);
                        if (plan != null && (found == null || plan.TotalDeltaV < found.TotalDeltaV)) found = plan;
                    }
                return found;
            };
            Action<double[], double[]> search = (departures, durations) =>
            {
                foreach (double departure in departures)
                {
                    InterplanetaryPlan plan = cheapest(departure, durations);
                    if (plan != null && (best == null || plan.TotalDeltaV < best.TotalDeltaV)) best = plan;
                }
            };
            double[] coarseDepartures = Linspace(epoch, epoch + span, departureSamples);
            double[] coarseDurations = Linspace(0.6 * nominal, 1.7 * nominal, durationSamples);
            InterplanetaryPlan[] coarse = coarseDepartures.Select(d => cheapest(d, coarseDurations)).ToArray();
            double minimum = coarse.Where(p => p != null).Select(p => p.TotalDeltaV).DefaultIfEmpty(double.NaN).Min();
            if (double.IsNaN(minimum)) throw new ArgumentException("No Lambert launch window found");
            // The earliest departure within 5% of the cheapest: a window opening now
            // beats a marginally cheaper one a synodic period later.
            best = coarse.First(p => p != null && p.TotalDeltaV <= 1.05 * minimum);
            double departureStep = coarseDepartures[1] - coarseDepartures[0];
            double durationStep = coarseDurations[1] - coarseDurations[0];
            double center = best.DepartureUt, flight = best.ArrivalUt - best.DepartureUt;
            search(Linspace(Math.Max(epoch, center - departureStep), Math.Min(epoch + span, center + departureStep), departureSamples),
                   Linspace(Math.Max(0.6 * nominal, flight - durationStep), Math.Min(1.7 * nominal, flight + durationStep), durationSamples));
            return best;
        }

        /// <summary>Synodic period of two bodies orbiting the same parent.</summary>
        public static double SynodicPeriod(double parentMu, double a1, double a2)
        {
            double p1 = 2 * Math.PI * Math.Sqrt(a1 * a1 * a1 / parentMu), p2 = 2 * Math.PI * Math.Sqrt(a2 * a2 * a2 / parentMu);
            double inverse = Math.Abs(1 / p1 - 1 / p2);
            return inverse > 1e-15 ? 1 / inverse : Math.Max(p1, p2);
        }

        /// <summary>
        /// Escape burn from a bound parking orbit whose hyperbola leaves along ``vinf``:
        /// the burn point is where the radius makes the asymptote angle acos(-1/e) with
        /// ``vinf`` in the plane they span, flying tangentially (a periapsis burn).
        /// Searches one parking period centred on ``center``; the cheapest root wins.
        /// </summary>
        public static bool EjectionBurn(V3 r0, V3 v0, double epoch, double mu, V3 vinf, double center,
                                        out double burnUt, out V3 deltaV)
        {
            burnUt = double.NaN;
            deltaV = V3.Zero;
            double energy, e0, pe, ap;
            Kepler.Elements(r0, v0, mu, out energy, out e0, out pe, out ap);
            if (energy >= 0 || !(vinf.Norm > 0)) return false;
            double a = -mu / (2 * energy), period = 2 * Math.PI * Math.Sqrt(a * a * a / mu);
            V3 parkingNormal = V3.Cross(r0, v0).Unit(), direction = vinf.Unit();
            double speed2 = V3.Dot(vinf, vinf);
            Func<double, double> mismatch = ut =>
            {
                V3 r, v;
                Kepler.Propagate(r0, v0, ut - epoch, mu, out r, out v);
                double e = 1 + r.Norm * speed2 / mu;
                double angle = Math.Acos(Math.Max(-1, Math.Min(1, V3.Dot(r.Unit(), direction))));
                return angle - Math.Acos(-1 / e);
            };
            Func<double, V3> burn = ut =>
            {
                V3 r, v;
                Kepler.Propagate(r0, v0, ut - epoch, mu, out r, out v);
                V3 normal = V3.Cross(r, direction);
                if (normal.Norm < 1e-9 * r.Norm) return new V3(double.NaN, double.NaN, double.NaN);
                normal = normal.Unit();
                if (V3.Dot(normal, parkingNormal) < 0) return new V3(double.NaN, double.NaN, double.NaN);
                V3 tangent = V3.Cross(normal, r.Unit());
                return Math.Sqrt(speed2 + 2 * mu / r.Norm) * tangent - v;
            };
            const int samples = 720;
            double start = center - period / 2, step = period / samples, bestCost = double.PositiveInfinity;
            double previous = mismatch(start);
            for (int i = 1; i <= samples; i++)
            {
                double ut = start + i * step, current = mismatch(ut);
                if (previous * current <= 0 && !double.IsNaN(previous) && !double.IsNaN(current))
                {
                    double low = ut - step, high = ut, lowValue = previous;
                    for (int k = 0; k < 60; k++)
                    {
                        double middle = (low + high) / 2, value = mismatch(middle);
                        if (value * lowValue <= 0) high = middle; else { low = middle; lowValue = value; }
                    }
                    double root = (low + high) / 2;
                    V3 dv = burn(root);
                    if (!double.IsNaN(dv.X) && dv.Norm < bestCost) { bestCost = dv.Norm; burnUt = root; deltaV = dv; }
                }
                previous = current;
            }
            return !double.IsNaN(burnUt);
        }

        /// <summary>
        /// Course correction on the way to a body: leave ``lead`` seconds from now
        /// and arrive within 15% of the remaining flight time of ``nominalArrival``,
        /// aimed so the arrival periapsis is at the target altitude.
        /// </summary>
        public static TransferPlan PlanCorrection(V3 position, V3 velocity, double epoch, double mu, Body target,
                                                  IEphemeris ephemeris, double nominalArrival, double targetAltitude,
                                                  double lead, int samples = 41)
        {
            double periapsis = target.Radius + targetAltitude;
            double departureUt = epoch + lead, remaining = nominalArrival - departureUt;
            if (!(remaining > 0)) throw new ArgumentException("The nominal arrival is already past");
            V3 start, startVelocity;
            Kepler.Propagate(position, velocity, lead, mu, out start, out startVelocity);
            double bestCost = double.PositiveInfinity, bestArrival = double.NaN;
            bool bestLong = false;
            foreach (double arrivalUt in Linspace(nominalArrival - 0.15 * remaining, nominalArrival + 0.15 * remaining, samples))
                foreach (bool longWay in new[] { false, true })
                {
                    V3 finish, finishVelocity, departure, arrival;
                    ephemeris.State(arrivalUt, out finish, out finishVelocity);
                    try { Kepler.Lambert(start, finish, arrivalUt - departureUt, mu, longWay, out departure, out arrival); }
                    catch (ArgumentException) { continue; }
                    double cost = (departure - startVelocity).Norm + CaptureDeltaV(target.Mu, periapsis, (arrival - finishVelocity).Norm);
                    if (cost < bestCost) { bestCost = cost; bestArrival = arrivalUt; bestLong = longWay; }
                }
            if (double.IsNaN(bestArrival)) throw new ArgumentException("No correction transfer found");
            return Aimed(start, startVelocity, departureUt, bestArrival, bestLong, ephemeris, mu, target.Mu, periapsis, targetAltitude);
        }

        /// <summary>Hohmann estimate from a circular parking orbit to a moon's orbit plus the capture.</summary>
        public static double MoonBudget(double mu, double parkingRadius, double moonSemiMajorAxis, double moonMu, double goalRadius)
        {
            double r1 = parkingRadius, a2 = moonSemiMajorAxis;
            double departure = Math.Sqrt(mu / r1) * (Math.Sqrt(2 * a2 / (r1 + a2)) - 1);
            double vinf = Math.Sqrt(mu / a2) * (1 - Math.Sqrt(2 * r1 / (r1 + a2)));
            return departure + CaptureDeltaV(moonMu, goalRadius, Math.Abs(vinf));
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
