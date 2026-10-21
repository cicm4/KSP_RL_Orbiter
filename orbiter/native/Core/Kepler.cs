// Two-body propagation and zero-revolution Lambert transfers, mirroring
// orbiter/orbital.py. Vectors share one nonrotating frame, SI units.
using System;

namespace OrbiterRL.Core
{
    public static class Kepler
    {
        public static void Stumpff(double z, out double c, out double s)
        {
            if (Math.Abs(z) < 1e-5)
            {
                c = 0.5 - z / 24 + z * z / 720 - z * z * z / 40320;
                s = 1.0 / 6 - z / 120 + z * z / 5040 - z * z * z / 362880;
                return;
            }
            if (z > 0)
            {
                double root = Math.Sqrt(z);
                c = (1 - Math.Cos(root)) / z;
                s = (root - Math.Sin(root)) / (root * root * root);
                return;
            }
            double r = Math.Sqrt(-z);
            if (r > 700) throw new ArgumentException("Universal anomaly exceeds the hyperbolic numeric range");
            c = (Math.Cosh(r) - 1) / (-z);
            s = (Math.Sinh(r) - r) / (r * r * r);
        }

        public static void Propagate(V3 r0, V3 v0, double dt, double mu, out V3 position, out V3 velocity)
        {
            double radius = r0.Norm;
            if (!(radius > 0) || !(mu > 0) || double.IsNaN(dt) || double.IsInfinity(dt))
                throw new ArgumentException("Invalid propagation input");
            if (dt == 0) { position = r0; velocity = v0; return; }
            double sqrtMu = Math.Sqrt(mu);
            double radial = V3.Dot(r0, v0) / sqrtMu;
            double alpha = 2 / radius - V3.Dot(v0, v0) / mu;
            Func<double, double[]> equation = chi =>
            {
                double z = alpha * chi * chi, c, s;
                Stumpff(z, out c, out s);
                double value = radial * chi * chi * c + (1 - alpha * radius) * chi * chi * chi * s
                               + radius * chi - sqrtMu * dt;
                double derivative = radial * chi * (1 - z * s) + (1 - alpha * radius) * chi * chi * c + radius;
                return new[] { value, derivative };
            };
            int direction = dt > 0 ? 1 : -1;
            double extent = Math.Max(1.0, sqrtMu * Math.Abs(dt) / radius);
            double low = direction > 0 ? 0.0 : -extent, high = direction > 0 ? extent : 0.0;
            bool bracketed = false;
            for (int i = 0; i < 100; i++)
            {
                if (equation(low)[0] <= 0 && 0 <= equation(high)[0]) { bracketed = true; break; }
                if (direction > 0) high *= 2; else low *= 2;
            }
            if (!bracketed) throw new ArgumentException("Could not bracket universal Kepler anomaly");
            double x = (low + high) / 2;
            bool converged = false;
            for (int i = 0; i < 100; i++)
            {
                double[] e = equation(x);
                if (e[0] < 0) low = x; else high = x;
                double candidate = e[1] > 0 ? x - e[0] / e[1] : double.NaN;
                if (!(low < candidate && candidate < high)) candidate = (low + high) / 2;
                if (Math.Abs(candidate - x) <= 1e-11 * Math.Max(1.0, Math.Abs(x))) { x = candidate; converged = true; break; }
                x = candidate;
            }
            if (!converged) throw new ArgumentException("Universal Kepler solve did not converge");
            double cc, ss;
            Stumpff(alpha * x * x, out cc, out ss);
            double f = 1 - x * x / radius * cc, g = dt - x * x * x / sqrtMu * ss;
            position = f * r0 + g * v0;
            double current = position.Norm;
            double fdot = sqrtMu / (current * radius) * (alpha * x * x * x * ss - x);
            double gdot = 1 - x * x / current * cc;
            velocity = fdot * r0 + gdot * v0;
        }

        public static void Lambert(V3 p1, V3 p2, double duration, double mu, bool longWay,
                                   out V3 departure, out V3 arrival)
        {
            if (!(duration > 0) || !(mu > 0) || double.IsInfinity(duration))
                throw new ArgumentException("duration and mu must be positive and finite");
            double r1 = p1.Norm, r2 = p2.Norm;
            if (Math.Min(r1, r2) <= 0) throw new ArgumentException("Lambert endpoints cannot be zero");
            double cosine = Math.Max(-1.0, Math.Min(1.0, V3.Dot(p1, p2) / (r1 * r2)));
            if (1 - Math.Abs(cosine) < 1e-12) throw new ArgumentException("Collinear Lambert endpoints are singular");
            double sine = Math.Sqrt(Math.Max(0.0, 1 - cosine * cosine)) * (longWay ? -1 : 1);
            double a = sine * Math.Sqrt(r1 * r2 / (1 - cosine));
            double sqrtMu = Math.Sqrt(mu);
            Func<double, double[]> transferTime = z =>
            {
                double c, s;
                Stumpff(z, out c, out s);
                if (c <= 0) return new[] { double.NaN, double.NaN };
                double y = r1 + r2 + a * (z * s - 1) / Math.Sqrt(c);
                if (y <= 0) return new[] { double.NaN, y };
                double chi = Math.Sqrt(y / c);
                return new[] { (chi * chi * chi * s + a * Math.Sqrt(y)) / sqrtMu, y };
            };
            Func<double, bool> finite = t => !(double.IsNaN(t) || double.IsInfinity(t));
            double low = -4 * Math.PI * Math.PI, high = 4 * Math.PI * Math.PI - 1e-6;
            if (!finite(transferTime(low)[0]))
            {
                double invalid = low, valid = high;
                for (int i = 0; i < 90; i++)
                {
                    double middle = (invalid + valid) / 2;
                    if (finite(transferTime(middle)[0])) valid = middle; else invalid = middle;
                }
                low = valid;
            }
            double tLow = transferTime(low)[0];
            if (tLow > duration)
            {
                for (int i = 0; i < 20; i++)
                {
                    double candidate = low * 2;
                    double candidateTime = transferTime(candidate)[0];
                    if (!finite(candidateTime)) break;
                    low = candidate;
                    if (candidateTime <= duration) break;
                }
            }
            if (transferTime(low)[0] > duration || transferTime(high)[0] < duration)
                throw new ArgumentException("Requested Lambert duration is outside the bracketed branch");
            double yy = double.NaN;
            bool converged = false;
            for (int i = 0; i < 100; i++)
            {
                double z = (low + high) / 2;
                double[] current = transferTime(z);
                yy = current[1];
                if (!finite(current[0]) || current[0] < duration) low = z; else high = z;
                if (finite(current[0]) && Math.Abs(current[0] - duration) <= 1e-10 * Math.Max(duration, 1.0))
                { converged = true; break; }
            }
            if (!converged) throw new ArgumentException("Lambert solve did not converge");
            double f = 1 - yy / r1, g = a * Math.Sqrt(yy / mu), gdot = 1 - yy / r2;
            if (Math.Abs(g) < 1e-12) throw new ArgumentException("Lambert Lagrange g is singular");
            departure = (p2 - f * p1) / g;
            arrival = (gdot * p2 - p1) / g;
        }

        /// <summary>orbital.elements: energy, eccentricity, periapsis/apoapsis radii.</summary>
        public static void Elements(V3 r, V3 v, double mu, out double energy, out double eccentricity,
                                    out double periapsis, out double apoapsis)
        {
            double radius = r.Norm;
            energy = V3.Dot(v, v) / 2 - mu / radius;
            double h = V3.Cross(r, v).Norm;
            eccentricity = Math.Sqrt(Math.Max(0.0, 1 + 2 * energy * h * h / (mu * mu)));
            double p = h * h / mu;
            periapsis = p / (1 + eccentricity);
            apoapsis = energy < 0 && eccentricity < 1 ? p / (1 - eccentricity) : double.PositiveInfinity;
        }

        public static double RocketDeltaV(double mass, double dryMass, double isp)
        {
            if (!(dryMass > 0) || !(mass >= dryMass) || !(isp > 0))
                throw new ArgumentException("Need wet mass >= positive dry mass and positive Isp");
            return isp * Units.G0 * Math.Log(mass / dryMass);
        }
    }
}
